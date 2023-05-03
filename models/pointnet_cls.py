import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # @! [log_softmax](https://www.zhihu.com/question/358069078): softmax 中有取指数操作，可能数值溢出超出 float 范围，故为对 softmax 的结果取对数，值域(负无穷, 0)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    # @ Loss 的实现
    def forward(self, pred, target, trans_feat):
        """
        损失项: nll_loss + mat_diff_loss
        nll_loss: log_softmax + F.nll_loss 取样本的标签对应的预测得分的相反数(得分越低此数值越大)作为单个样本的损失，批量中所有样本损失的均值作为批量的损失
        mat_diff_loss: loss = AA^T - I 旋转矩阵损失函数, 旋转矩阵是正交阵, AA^T=I
        Args:
            pred: (B, num_cls) 每个样本在每个类别上的得分
                torch.Size([24, 40]) elem_val: -3.6188 由 log_softmax 得到
            target: (B,) 每个样本的数字标签
                torch.Size([24]) elem_val: 8
        """
        # pred = torch.tensor([[-1.0,-2,-3],[-1,-2,-3]]); target = torch.tensor([1, 2]), loss=2.5=(2+3)/2
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
