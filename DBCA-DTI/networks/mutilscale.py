import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MultiScaleWeightAdd(nn.Module):
    def __init__(self, numHidden,hidden,Length):
        super(MultiScaleWeightAdd, self).__init__()
        self.w = nn.Parameter(torch.ones((3, numHidden, Length), dtype = torch.float32), requires_grad = True)
        self.epsilon = 0.0001
        self.conv = nn.Conv1d(numHidden, hidden, 1, 1, 0)
        self.swish = nn.SiLU()
        self.relu = nn.ReLU()
    def forward(self, x1, x2, x3):
        w = self.relu(self.w)
        weight = w / (torch.sum(w, dim = 0) + self.epsilon)
        return self.conv(self.swish(weight[0] * x1 + weight[1] * x2 + weight[2] * x3)).permute(0,2,1)

class MEncoder(nn.Module):
    def  __init__(self, hiddenNum, hidden,Length):
        self.weightRes = 1
        self.weightFuse = 1
        super(MEncoder, self).__init__()
        self.hiddenNum = hiddenNum
        self.dropout = 0.05

        self.multiscalefuse = MultiScaleWeightAdd(hiddenNum,hidden,Length)
        self.Relu = nn.ReLU()
        self.Drop = nn.Dropout(0.2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(hiddenNum, hiddenNum * 3, 3, 1, 3 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.GELU(),
            # nn.LeakyReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 3, 1, 3 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.GELU(),
            # nn.LeakyReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum, 3, 1, 3 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.GELU(),
            # nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(hiddenNum, hiddenNum * 3, 5, 1, 5 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.GELU(),
            # nn.LeakyReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 5, 1, 5 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.GELU(),
            # nn.LeakyReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum, 5, 1, 5 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.GELU(),
            # nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(hiddenNum, hiddenNum * 3, 7, 1, 7 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.GELU(),
            # nn.Mish(),
            # nn.LeakyReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 7, 1, 7 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.GELU(),
            # nn.Mish(),
            # nn.LeakyReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum, 7, 1, 7 // 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.GELU(),
            # nn.Mish(),
            # nn.LeakyReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Scale 3
        cnnx3 = self.conv2(x)

        # Scale 5
        cnnx5 = self.conv3(x)

        # Scale 7
        cnnx7 = self.conv4(x)

        # Fuse MultiScale Feature
        multiscaleFeature = self.multiscalefuse(cnnx3, cnnx5, cnnx7)
        return multiscaleFeature
