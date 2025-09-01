from collections import defaultdict

from model_loss import ChamferLoss, projection_loss, UniformLoss
from torch.optim.lr_scheduler import ExponentialLR
from uniformLoss.loss import Loss
import torch
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from SPUUNet import SPUUNet
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return  centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
class Model(object):
    def __init__(self, net, phase, opt, writer_tensorboard=None):
        self.net = net
        self.phase = phase
        self.writer_tensorboard = writer_tensorboard
        if self.phase == 'train':
            self.error_log = defaultdict(int)
            self.chamfer_criteria = ChamferLoss()
            self.uniformloss = UniformLoss(loss_name='uniform', alpha=1)
            self.resplusion  =UniformLoss(alpha=1)
            self.old_lr = opt.lr_init
            self.lr = opt.lr_init
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr_init,
                                              betas=(0.9, 0.999))
            self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.7)
            self.decay_step = opt.decay_iter
        self.step = 0

    def set_input(self, input_pc, radius, label_pc=None):
        """`
        :param
            input_pc       Bx3xN
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.radius = radius
        self.R=4
        self.input = input_pc.detach()
        #self.alpha =ALPHA
        B, C, N = input_pc.shape
        downsamplenum = int(N / self.R)  # 假设 R 是 4

        #使用高效的点采样和索引函数
        far_point = farthest_point_sample(xyz=input_pc.permute(0, 2, 1), npoint=downsamplenum)
        input_point = index_points(input_pc.permute(0, 2, 1), far_point)
        #print("input_point",input_point.shape)
        #first stage
        far_point1 = farthest_point_sample(xyz=input_pc.permute(0, 2, 1), npoint=int(downsamplenum*2))
        gt_point = index_points(input_pc.permute(0, 2, 1), far_point1)

        self.input = input_point.detach().cuda()  #B,C,N
        self.gt_point =gt_point.detach().cuda()
        #print("self.gt_point",self.gt_point.shape)
        if label_pc is not None:
            self.gt = label_pc.detach().permute(0, 2, 1).cuda()  #(B,N,C)
            #print("self.gt",self.gt.shape)
        else:
            self.gt = None

        return self.input, self.gt,self.gt_point
    def forward(self):
        if self.gt is not None:
            self.predicted, self.gt = self.net(self.input,self.gt)
        else:
            self.predicted = self.net(self.input)

    def get_lr(self, optimizer):
        """Get the current learning rate from optimizer.
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def optimize(self, steps=None, epoch=None):
        """
        run forward and backward, apply gradients
        """
        self.optimizer.zero_grad()
        self.net.train()
        self.forward()


        P1= self.predicted
        #print("p1",P1.shape)   #B N 3
        #print("p2",P2.shape)
        #cd_1 =Loss().get_cd_loss(P1, self.gt_point)
        cd_1 =Loss().get_cd_loss(P1,self.gt)

        alpha = 0.1

        uniform_2 = self.uniformloss(P1)
        uniform_1 = self.resplusion(P1)

        loss1, loss2, loss3,loss4 =10*cd_1+0.1*uniform_1, cd_1,  uniform_1,uniform_2

        loss =loss1
        losses = [loss1.item(), loss2.item(),loss3.item(),loss4.item()]


        loss.backward()
        self.optimizer.step()

        if steps % self.decay_step == 0 and steps != 0:
            self.lr_scheduler.step()
        lr = self.get_lr(self.optimizer)
        return losses, lr
if __name__ =="__main__":
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SPUUNet(num_neighbors=16)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
  #  net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.train()
    model = Model(net=net, phase='train', opt=type('Options', (object,), {'lr_init': 0.001, 'decay_iter': 1000}))
    point_cloud = torch.rand(2,3, 1024).cuda()
    data_radius = np.ones(shape=(len(point_cloud)))
    data_radius = torch.tensor(data_radius, dtype=torch.float32).cuda()
    label_pc=point_cloud.clone()
    model.set_input(point_cloud, data_radius,label_pc=label_pc)
    total_batch = 5
    epoch =2
    loss, lr = model.optimize(total_batch, epoch)

    print("Losses:", loss)
    print("Learning Rate:", lr)
    print(model)