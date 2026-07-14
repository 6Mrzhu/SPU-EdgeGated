import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
from model import  GlobalDownSample, LocalDownSample, UpSample
from common import query_knn_point, MLP_CONV, fps, MLP_Res, Conv1d, Conv2d
from torch import einsum
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import grouping_operation
import ops
from einops import rearrange
import math
from torch_vertex import DynConv2d
from ops import group


class DifferentiableKNN(nn.Module):
   

    def __init__(self, k=16):
        super(DifferentiableKNN, self).__init__()
        self.k = k

    def forward(self, query_points, support_points):
        """
        
        query_points: (B, M, C)
        support_points: (B, N, C)

       
        indices: (B, M, k)  
        weights: (B, M, k)  
        """
       
        dist = torch.cdist(query_points, support_points)  # (B, M, N)

      
        _, indices = torch.topk(dist, self.k, largest=False)  # (B, M, k)

      
        weights = F.softmax(-dist, dim=-1)  # (B, M, N)
        weights = torch.gather(weights, 2, indices)  # (B, M, k)

        return indices, weights

class UniformInterpolationLayer(nn.Module):
    

    def __init__(self, k=16, num_new_points_per_point=1):
        super(UniformInterpolationLayer, self).__init__()
        self.knn = DifferentiableKNN(k)
        self.k = k
        self.num_new_points_per_point = num_new_points_per_point

    def forward(self, x):
        """
      
        x: (B, N, C)  
        reture：
        (B, N + N*num_new_points_per_point, C) 
        """
        B, N, C = x.shape

       
        rand_directions = torch.randn(B, N, self.num_new_points_per_point, C, device=x.device)
        rand_directions = F.normalize(rand_directions, p=2, dim=-1)  # (B, N, num_new, C)

      
        local_radius = torch.norm(x, p=2, dim=-1, keepdim=True) * 0.1  # (B, N, 1)
        local_radius = local_radius.unsqueeze(2).expand(-1, -1, self.num_new_points_per_point, -1)

      
        new_points = x.unsqueeze(2) + rand_directions * local_radius  # (B, N, num_new, C)
        new_points = new_points.view(B, -1, C)  # (B, N*num_new, C)

       
        indices, weights = self.knn(new_points, x)  # indices: (B, N*num_new, k)

       
        neighbors = torch.gather(
            x.unsqueeze(1).expand(-1, new_points.shape[1], -1, -1),  # (B, N*num_new, N, C)
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N*num_new, k, C)
        )

      
        weights = weights.unsqueeze(-1)  # (B, N*num_new, k, 1)
        interpolated_points = torch.sum(neighbors * weights, dim=2)  # (B, N*num_new, C)

      
        output = torch.cat([x, interpolated_points], dim=1)  # (B, N + N*num_new, C)

        return output

class Transformer(nn.Module):
    """
    

    feed forward of transformer
    Args:
        x: Tensor of features, (B, in_channel, n)
        pos: Tensor of positions, (B, 3, n)

    Returns:
        y: Tensor of features with attention, (B, in_channel, n)

    """

    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class Transformer_extractor(nn.Module):
    """
    Point-wise feature extractor.

    Input:
        points: input points, (B, 3, N_input)
    Output:
        point_feat: ouput feature, (B, dim_feat, N_input)
    """

    def __init__(self, dim_feat, hidden_dim):
        super(Transformer_extractor, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, dim_feat])
        self.mlp_2 = MLP_CONV(in_channel=dim_feat * 2, layer_dims=[dim_feat * 2, dim_feat])
        self.point_transformer = Transformer(dim_feat, dim=hidden_dim)

    def forward(self, points):
        feature_1 = self.mlp_1(points)
        global_feature = torch.max(feature_1, 2, keepdim=True)[0]
        feature_2 = torch.cat([feature_1, global_feature.repeat((1, 1, feature_1.size(2)))], 1)
        feature_3 = self.mlp_2(feature_2)
        point_feat = self.point_transformer(feature_3, points)
        return point_feat


class N2PAttention(nn.Module):
    def __init__(self, k):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = k
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.k_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.v_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(128, 64, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k,
                               'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention @ v,
                        'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x


class N2PAttention1(nn.Module):
    def __init__(self, k):
        super(N2PAttention1, self).__init__()
        self.heads = 4
        self.K = k
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.LeakyReLU(0.2),
                                nn.Conv1d(128, 128, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k,
                               'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention @ v,
                        'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x


class Feature_extractor(nn.Module):
    def __init__(self, dim_feat, hidden_dim):
        super(Feature_extractor, self).__init__()
        self.Encoder = Transformer_extractor(dim_feat, hidden_dim)
        self.LocalDown = LocalDownSample(npts_ds=32, k=16)
        self.GlobalDown = GlobalDownSample(npts_ds=32)
        self.point_transformer = Transformer(in_channel=128, dim=64)
        self.n2f = N2PAttention(k=8)  # 16  #8   #4  #8
        self.n2f1 = N2PAttention1(k=16)  # 8    #16  #8  #8
        self.unsampling = UpSample(inchannels=dim_feat)
        self.gate_fusion = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x:[B,N,3]
        :return: [B,3,4*N]
        '''
        x = x.permute(0, 2, 1).contiguous()
        feature = self.Encoder(x)
        # 2,64,256
        # print("feature:",feature.shape)
        loca_point = self.LocalDown(feature)  # 2,128,128
        # print("loca_point",loca_point.shape)
        up_point_localfeature = self.unsampling(feature, local_feature)  # 2,128,128
        up_point_localfeature = self.n2f1(torch.cat([up_point_localfeature, feature], dim=1))  # b,256,N
        # print("111",up_point_localfeature.shape)

        Gobal_point = self.GlobalDown(feature)
        # print("gobal",Gobal_point.shape)
        up_point_Gobalfeature = self.unsampling(feature, Global_feature)
        up_point_Gobalfeature = self.n2f1(torch.cat([up_point_Gobalfeature, feature], dim=1))  # B 128,N
        # print("220", up_point_Gobalfeature.shape)

        gate = self.gate_fusion(torch.cat([up_point_localfeature, up_point_Gobalfeature], dim=1))
        up_point_feature = gate * up_point_localfeature + (1 - gate) * up_point_Gobalfeature


        return up_point_feature


class up_block(nn.Module):
    def __init__(self, up_ratio=4, in_channels=128):
        super(up_block, self).__init__()
        self.up_ratio = up_ratio
        self.num_heads = 4 
        self.head_dim = in_channels // 4  
        self.qkv_proj = nn.Conv2d(
            in_channels,
            in_channels * 3, 
            kernel_size=1,
            bias=False
        )

        # 
        self.alpha_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid() 
        )

        self.merge_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),  # 2C→C
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        net = input  # b ,128,n
        B, C, N = net.size()
        group_feature = group(net, K=self.up_ratio, group_type="neighbor")  # B,C,N,R
        # print("group",group_feature.shape)
        center_features = net.unsqueeze(-1).expand(-1, -1, -1, self.up_ratio)  # B,C,N,R
        wight_net = torch.cat([center_features, group_feature], dim=1)
        merged_feat = self.merge_proj(wight_net)  # [B, C, N, K]
        # print("merged_feat",merged_feat.shape)
        qkv = self.qkv_proj(merged_feat)  # [B, 3C, N, K]
        # print("qvk",qkv.shape)
        q, k, v = torch.split(qkv, C, dim=1)  # [B, C, N, K]
        # print("q",q.shape)
        q = q.view(B, self.num_heads, self.head_dim, N, self.up_ratio).permute(0, 1, 3, 2, 4)
        k = k.view(B, self.num_heads, self.head_dim, N, self.up_ratio).permute(0, 1, 3, 4, 2)
        v = v.view(B, self.num_heads, self.head_dim, N, self.up_ratio).permute(0, 1, 3, 2, 4)
        # print("v",v.shape)  #[2, 4, 32, 256, 4]
        # print("k",k.shape)  #[2,4,32,256,4]
        attn_scores = torch.matmul(q, k)  # [B, H, C/H, N, K]
        # rint("attn_scores",atres / (self.head_dim ** 0.5)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, C/H, N, K]
        # print("attn_weights",attn_weights.shape)  #[2, 4, 32, 4, 256]
        #[B, H, C/H, N, K] → [B, C, N, K]
        # attn_output = torch.einsum('bhcnk, bhcmk -> bhcdn', attn_weights, v)  # [2,4,32,256,32]
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 1, 3, 2, 4).contiguous()
        # print("attn_output",attn_output.shape)
        attn_output = attn_output.view(B, C, N, self.up_ratio)  # head_dim → [B, C, N, K]
        # print("attn_output1",attn_output.shape)
        attn_agg = attn_output.mean(dim=-1, keepdim=True)  # [B, C, N, 1]
        alpha = self.alpha_proj(attn_agg)  # [B, 1, N, 1]（[0,1]）
        alpha = alpha.expand(-1, -1, -1, self.up_ratio)  # [B, 1, N, up_ratio]
        # print("a",alpha.shape)
        interpolated = alpha * center_features + (1 - alpha) * group_feature
        interpolated = interpolated.reshape(B, C, -1)

        return interpolated



class Upsampling_unit(nn.Module):
    """
    Point upsampling unit

    Input:
        point_feat: input feature, (B, dim_feat, N_input)
        points: input points, (B, 3, N_input)
    Output:
        up_feat: upsampled feature, (B, dim, up_ratio * N_input)
        duplicated_point: upsampled results, (B, 3, up_ratio * N_input)
    """

    def __init__(self, up_ratio=2):
        super(Upsampling_unit, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=256, layer_dims=[128, 64])
        self.mlp_2 = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.duplicated_branch = up_block(up_ratio=up_ratio, in_channels=128)
        self.Encoder = Transformer_extractor(128, 64)
        self.up_ratio = up_ratio
        self.uniform = UniformInterpolationLayer(k=16, num_new_points_per_point=int(self.up_ratio-1))
    def forward(self, point_feat, point):
        duplicated_feat = self.duplicated_branch(point_feat) 
        up_point1 =self.uniform(point).permute(0,2,1).contiguous()
        up_point = self.Encoder(up_point1)
        up_feat = self.mlp_2(torch.cat([duplicated_feat, up_point], dim=1))
        up_feat = torch.relu(up_feat)

        return  up_point1.permute(0,2,1).contiguous(), up_feat


class DenseGenerator(nn.Module):
    def __init__(self, upsacle_factor=4, dim=256):
        super(DenseGenerator, self).__init__()

        self.feature_extractor = Feature_extractor(dim_feat=64, hidden_dim=32)

        self.feature_expansion = Upsampling_unit(up_ratio=upsacle_factor)
        self.coordinate_regression = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1))

    def forward(self, x):
        '''
        :param x:[B,N,3]
        :return: [B,3,4*N]
        '''

        y = self.feature_extractor(x)
        up_point1, up_feat = self.feature_expansion(y, x)
        coord = self.coordinate_regression(up_feat).permute(0, 2, 1).contiguous()
        return coord


class SPUUNet(nn.Module):
    def __init__(self, num_neighbors=16, dim=512):
        super(SPUUNet, self).__init__()
        self.dense = DenseGenerator(upsacle_factor=4, dim=dim)
    def forward(self, point_cloud, gt=None):
        '''
        :param x: [B,N,3]
        :return: [B,4N,3]
        '''

        coord= self.dense(point_cloud)
        # print(feature.shape)
        # offset = self.refine(feature,coord)

        P1 = coord.contiguous()  # example: [16, 512, 3]
        if self.training:
            return P1, gt
        else:
            return P1


if __name__ == "__main__":
    xyz = torch.randn(2, 256, 256).cuda()
    point = torch.randn((2, 256, 3)).cuda()
    model = SPUUNet(num_neighbors=16,dim=512).cuda()
    f = up_block(up_ratio=4, in_channels=256).cuda()
    uniform =UniformInterpolationLayer(k=16, num_new_points_per_point=3)
    feature = Feature_extractor(dim_feat=64, hidden_dim=32).cuda()
    out_point =model(point)
    uniform =uniform(point)
    print(uniform.shape)
