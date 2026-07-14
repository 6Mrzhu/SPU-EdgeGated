import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(MLP, self).__init__()
        self.mlp =nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
    def forward(self,x):

        x=self.mlp(x)

        return x
if __name__ =="__main__":
    pooler = MLP(in_channels=64, out_channels=128).cuda()
    p1 = torch.randn(2, 256, 64).cuda()
    p2 = pooler(p1.permute(0,2,1))
    print(p2.shape)
