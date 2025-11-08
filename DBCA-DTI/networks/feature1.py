import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        # self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=o_channel, out_channels=growth_rate, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        # self.bn2 = nn.BatchNorm1d(o_channel)
        # self.drop_rate = 0.1

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        # xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)

        return torch.cat([x, xn], 1)



class Encoder6(nn.Module):
    def __init__(self, inc, outc, length,growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder6, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in
             range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1,
                                  padding=pad1, bias=False)
        self.cross_patch_linear0 = nn.Linear(inc, inc + growth_rate)
        self.cross_patch_linear1 = nn.Linear(inc + growth_rate, inc + growth_rate)
        self.act = nn.ReLU()
        # 改进的全局特征提取部分
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.attention = nn.Sequential(
            nn.Linear(inc, inc//4),  # 压缩通道
            nn.ReLU(),
            nn.Linear(inc//4, inc),  # 恢复通道
            nn.Sigmoid()  # 生成注意力权重
        )
        self.norm = nn.LayerNorm(inc)
    def forward(self, x):
        x = self.conv_in(x)
        # 全局平均池化获取全局上下文
        global_feat = self.global_avg_pool(x).squeeze(-1)  # 形状: [batch, channels]

        # 注意力机制增强重要特征
        attn_weights = self.attention(global_feat).unsqueeze(2)  # 形状: [batch, channels, 1]
        x_attended = x * attn_weights  # 应用注意力权重

        x2 = x_attended.permute(0, 2, 1)  # 转换为 [batch, length, channels]
        x2 = self.norm(x2)  # 归一化
        x1 = self.act(self.cross_patch_linear0(x2))
        x1 = self.act(self.cross_patch_linear1(x1))
        x1 = x1.permute(0, 2, 1)  # 转换回 [batch, channels, length]

        # 局部特征提取
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = x1 + x
        x = self.conv_out(x)
        x = self.relu(x)
        return x

