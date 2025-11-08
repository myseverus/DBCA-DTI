import torch
import torch.nn as nn
# from thop import profile
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

from .bagcamodel import  BAGCA

from .KAA_GAT import KAAGATConv
# from .KAA_CFGAT import KAACFGATConv
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class MolecularGCN(nn.Module):
    def __init__(self, atom_dim, hidden_dim=75, output_dim=75):
        super().__init__()
        # 第一层GCN：处理原子特征
        self.gcn1 = GCNConv(
            in_channels=atom_dim,
            out_channels=hidden_dim,
            improved=False,  # 是否使用改进的自环处理
            cached=False,  # 是否缓存归一化系数
            normalize=True  # 是否对邻接矩阵进行归一化
        )

        # 第二层GCN：聚合特征
        self.gcn2 = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            improved=False,
            cached=False,
            normalize=True
        )

        # 全连接输出层（如果需要）
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, atoms, adjs, edges=None):
        """
        输入形状:
        atoms: [batch_size, num_nodes, atom_dim]
        adjs:  [batch_size, num_nodes, num_nodes] - 邻接矩阵
        edges: 此处未使用，为保持接口兼容性保留
        """
        batch_size, num_nodes = atoms.shape[:2]

        # 合并为批量大图
        x = atoms.view(-1, atoms.size(-1))  # [batch*num_nodes, atom_dim]

        # 生成边索引（GCN使用稀疏邻接表示）
        edge_indices = []
        for i in range(batch_size):
            adj_mask = (adjs[i] > 0.5).float()
            edge_index = adj_mask.nonzero(as_tuple=False).t()  # [2, num_edges]

            # 添加索引偏移，区分不同批次的节点
            edge_index += i * num_nodes
            edge_indices.append(edge_index)

        edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]

        # 第一层GCN，使用激活函数
        x = F.relu(self.gcn1(x, edge_index))

        # 第二层GCN
        x = self.gcn2(x, edge_index)

        # 恢复批次维度
        x = x.view(batch_size, num_nodes, -1)

        return x


class MolecularGAT(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden_dim=75, heads=8, output_dim=75):
        super().__init__()
        self.heads = heads
        # 第一层GAT：处理原子特征
        self.gat1 = KAAGATConv(
            in_channels=atom_dim,
            out_channels=hidden_dim,
            heads=8,  # 多头注意力
            edge_dim=edge_dim,  # 使用边特征
            concat=True,  # 拼接多头结果
            dropout=0.2
        )

        # 第二层GAT：聚合特征
        self.gat2 = KAAGATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,  # 单头输出
            edge_dim=edge_dim,  # 可继续使用边特征
            concat=False,  # 不拼接，直接输出标量
            dropout=0.0
        )

        # 全连接输出层
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, atoms, adjs, edges):
        """
        输入形状:
        atoms: [batch_size, num_nodes, atom_dim]
        adjs:  [batch_size, num_nodes, num_nodes]
        edges: [batch_size, num_nodes, num_nodes, edge_dim]
        """
        batch_size, num_nodes = atoms.shape[:2]

        # 合并为批量大图
        x = atoms.view(-1, atoms.size(-1))  # [batch*num_nodes, atom_dim]

        # 生成边索引和边特征
        edge_indices = []
        edge_attrs = []
        for i in range(batch_size):
            adj_mask = (adjs[i] > 0.5).float()
            edge_index = adj_mask.nonzero(as_tuple=False).t()
            edge_attr = edges[i][edge_index[0], edge_index[1]]  # [num_edges, edge_dim]

            # 添加索引偏移
            edge_index += i * num_nodes
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)

        edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]
        edge_attr = torch.cat(edge_attrs, dim=0)  # [total_edges, edge_dim]

        # 第一层GAT
        # x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = self.gat1(x, edge_index, edge_attr=edge_attr)

        # 第二层GAT
        x = self.gat2(x, edge_index, edge_attr=edge_attr)

        # 恢复批次维度并池化
        x = x.view(batch_size, num_nodes, -1)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        # self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=o_channel, out_channels=growth_rate, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        # self.bn2 = nn.BatchNorm1d(o_channel)
        self.drop_rate = 0.1

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        # xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)

        return torch.cat([x, xn], 1)

class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in
             range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1,
                                  padding=pad1, bias=False)
    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x


class Fusion(nn.Module):
    def __init__(self, hidden1, hidden2, dropout=0.05):
        super(Fusion, self).__init__()
        self.si_L = nn.Sigmoid()
        self.si_S = nn.Sigmoid()
        self.so_f = nn.Sigmoid()
        self.combine = nn.Linear(128 * 4, 128)
        self.ln = nn.LayerNorm(128)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, LM_fea, Sty_fea):

        Sty_fea_norm = Sty_fea * (abs(torch.mean(LM_fea))/abs(torch.mean(Sty_fea)))
        f_h = torch.cat((LM_fea.unsqueeze(1), Sty_fea_norm.unsqueeze(1)), dim=1)
        f_att = torch.mean(f_h, dim=1)
        f_att = self.so_f(f_att)
        fus_fea = torch.cat((LM_fea, Sty_fea, LM_fea * f_att, Sty_fea * f_att), dim=1)
        fus_fea = self.combine(fus_fea)
        return fus_fea

class EnhancedFusion(nn.Module):
    def __init__(self, hidden_size=128, dropout=0.05, use_gate=True, use_residual=True):    #0.05
        super(EnhancedFusion, self).__init__()
        self.hidden_size = hidden_size
        self.use_gate = use_gate
        self.use_residual = use_residual

        # 特征变换层
        self.transform_LM = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        self.transform_Sty = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        # 特征缩放和归一化
        self.scale_LM = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.scale_Sty = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 多尺度特征融合
        self.multi_scale = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),  # 原始特征、乘积、差异、拼接
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 特征选择机制
        self.feature_selection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )

        # 门控机制
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )

        # 最终输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 残差连接层
        if self.use_residual:
            self.residual_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, LM_fea, Sty_fea):
        # 确保输入特征维度正确
        assert LM_fea.size(-1) == self.hidden_size, f"LM feature size {LM_fea.size(-1)} != {self.hidden_size}"
        assert Sty_fea.size(-1) == self.hidden_size, f"Style feature size {Sty_fea.size(-1)} != {self.hidden_size}"

        # 特征变换
        LM_transformed = self.transform_LM(LM_fea)
        Sty_transformed = self.transform_Sty(Sty_fea)

        # 特征缩放和归一化
        LM_scaled = self.scale_LM(LM_transformed)
        Sty_scaled = self.scale_Sty(Sty_transformed)

        # 计算特征间的动态权重 (不使用注意力机制)
        # 使用特征间的相似性和差异性来计算权重
        similarity = F.cosine_similarity(LM_scaled, Sty_scaled, dim=-1, eps=1e-8).unsqueeze(-1)
        difference = torch.abs(LM_scaled - Sty_scaled)

        # 多尺度特征融合
        # 结合原始特征、特征乘积、特征差异和拼接特征
        LM_weighted = LM_scaled * (1 + similarity)
        Sty_weighted = Sty_scaled * (1 + similarity)

        multi_scale_feats = torch.cat([
            LM_weighted,
            Sty_weighted,
            LM_weighted * Sty_weighted,  # 元素级乘法
            difference  # 元素级差异
        ], dim=-1)

        # 通过多尺度融合层
        fused_feats = self.multi_scale(multi_scale_feats)

        # 特征选择机制
        selection_input = torch.cat([LM_weighted, Sty_weighted], dim=-1)
        selection = self.feature_selection(selection_input)
        selected_feats = fused_feats * selection

        # 门控机制
        if self.use_gate:
            gate_input = selection_input
            gating = self.gate(gate_input)
            gated_feats = selected_feats * gating
        else:
            gated_feats = selected_feats

        # 残差连接
        if self.use_residual:
            residual_input = selection_input
            residual = self.residual_proj(residual_input)
            output = gated_feats + residual
        else:
            output = gated_feats

        # 最终输出
        output = self.output(output)

        return output


import torch
import torch.nn as nn


class ImprovedFusion(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(ImprovedFusion, self).__init__()
        # 1. 参数化特征对齐层
        self.align = nn.Linear(hidden_size, hidden_size)

        # 2. 改进的双通道注意力机制
        self.attn_net = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Sigmoid()
        )

        # 3. 融合层使用门控机制
        self.gate = nn.Sequential(
            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.GLU()  # 门控线性单元
        )

        # 4. 增强的归一化层
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)

        # 5. 调整后的正则化
        self.drop = nn.Dropout(p=dropout)

        # 6. 残差连接
        self.res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, LM_fea, Sty_fea):
        # 特征对齐
        aligned_Sty = self.align(Sty_fea)

        # 双通道注意力生成
        attn_input = torch.cat([LM_fea, aligned_Sty], dim=-1)
        attn_weights = self.attn_net(attn_input)
        attn_lm, attn_sty = attn_weights.chunk(2, dim=-1)

        # 注意力加权特征
        weighted_lm = LM_fea * attn_lm
        weighted_sty = aligned_Sty * attn_sty

        # 融合门控
        fusion_input = torch.cat([
            LM_fea,
            aligned_Sty,
            weighted_lm,
            weighted_sty
        ], dim=-1)

        fused = self.gate(fusion_input)

        # 残差连接
        fused = self.res_weight * fused + (1 - self.res_weight) * LM_fea

        # 归一化与正则化
        fused = self.final_norm(fused)
        return self.drop(fused)

class GRL(nn.Module):
    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - 1)
        return -coeff * gradOutput


class Contrast_Fusion(nn.Module):
    def __init__(self, dropout=0.05):
        super(Contrast_Fusion, self).__init__()
        self.so_L = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, LM_fea, Sty_fea):
        LM_att = self.so_L(LM_fea)
        fus_fea = LM_fea + Sty_fea * LM_att
        return fus_fea

class Improved_DT_LeNet(nn.Module):
    def __init__(self, hidden, dropout, classes, layers):
        super(Improved_DT_LeNet, self).__init__()
        self.layers = layers

        # 卷积模块：包含BN和残差连接的完整卷积块
        self.conv_blocks = nn.ModuleList()
        for _ in range(layers):

            self.pooling = nn.AdaptiveAvgPool1d(1)
            # 全连接模块：包含LayerNorm和Dropout
            self.fc_blocks = nn.ModuleList()
            for _ in range(layers):
                block = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout))
            self.fc_blocks.append(block)

            # 分类头
            self.classifier = nn.Sequential(
                nn.Linear(hidden, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, classes)
            )

    def forward(self, x):
        # 输入维度: [batch, seq_len, hidden]
        x = x.permute(0, 2, 1)  # -> [batch, hidden, seq_len]

        # 单一平均池化 (替代双路径池化)
        pooled = self.pooling(x)  # [batch, hidden, 1]
        pooled = pooled.squeeze(-1)  # [batch, hidden]

        # 保存池化特征
        pool_output = pooled.clone()
        # 全连接模块
        for block in self.fc_blocks:
            pooled = block(pooled) + pooled  # 带残差的全连接块

        # 分类输出
        logits = self.classifier(pooled)
        features = self.classifier[:2](pooled)  # 获取128维特征

        return logits, features, pool_output


class DT_LeNet(nn.Module):
    def __init__(self, hidden, dropout, classes, layers):
        super(DT_LeNet, self).__init__()
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(hidden)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.FC_combs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.FC_down = nn.Linear(hidden, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_feature):
        dti_feature = dti_feature.permute(0, 2, 1)  # self.BN(dti_feature.permute(0, 2, 1))
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        _ = dti_feature.clone()
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti, dti_feature, _

from .feature1 import Encoder6
from .mutilscale import MEncoder
class DBCA_DTI(nn.Module):
    def   __init__(self, layer_gnn, device, hidden1=256, hidden2=75, n_layers=1, attn_heads=1,
                 dropout=0.0):
        super(DBCA_DTI, self).__init__()
        '''GNN'''
        self.embed_protein = nn.Embedding(26, hidden2)

        '''LM'''
        self.encoder_p = MEncoder(75,75,1200)

        self.encoder_protein_LM = Encoder6(1024, hidden1, 128, 128,1, groups=64, pad1=7, pad2=3)
        self.encoder_drug = Encoder6(768, hidden1, 128, 128,1, groups=32, pad1=7, pad2=3)
        self.mbca = BAGCA(hidden1,16)
        self.mbca1 = BAGCA(75,15)

        # self.gcn = MolecularGCN(
        #     atom_dim=75,
        #     hidden_dim=75,
        # )

        self.gcn = MolecularGAT(
            atom_dim=75,
            edge_dim=4,
        )

        '''DECISION'''
        self.device = device
        self.layer_gnn = layer_gnn
        self.hidden = hidden1

        self.fusion = EnhancedFusion(128)

        self.FC_out1 = Improved_DT_LeNet(hidden1, 0.00, 2, 1)
        self.FC_out2 = Improved_DT_LeNet(hidden2, 0.00 , 2, 1)

        # self.DTI_feature = nn.ModuleList([nn.Linear(128, 128) for _ in range(2)])
        # self.act = nn.ReLU()
        self.DTI_Pre = nn.Linear(128, 2)
        # self.fusion =  nn.Linear(128*2,128)

    def forward(self, inputs):
        molecule_smiles, molecule_atoms, molecule_adjs, molecule_edges, proteins, protein_LM, molecule_LM = inputs
        # molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """DTI 1D feature with pretrain language model"""
        # proteins_acids_LM:[16,1200,256]、molecule_smiles_LM：[16,100,256]
        # protein:[16,1200,1024]\molecule:[16,100,768]->protein:[16,1200,256]\molecule:[16,100,256]


        proteins_acids_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_smiles_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)

        # [16,1300,256]
        DT_1D_Feature = self.mbca(molecule_smiles_LM, proteins_acids_LM)
        """DTI 2D feature with Graph Nerual Networks"""
        proteins_acids_GNN = torch.zeros((proteins.shape[0], proteins.shape[1], 75), device=self.device)
        DT_2D_Feature = torch.zeros((N, 1300, 75), device=self.device) # b 1300 d 1400
        for i in range(N):
            proteins_acids_GNN[i, :, :] = self.embed_protein(torch.LongTensor(proteins[i].to('cpu').numpy()).cuda())
        # proteins_acids_GNN:[16,1200,75]-> proteins_acids_GNN:[16,1200,75]
        # proteins_acids_GNN = self.encoder_protein_GNN(proteins_acids_GNN.permute(0, 2, 1)).permute(0, 2, 1)
        proteins_acids_GNN = self.encoder_p(proteins_acids_GNN)
        # [32,85,75]、[32,85,85]、[32,85,85,4]
        drug_graph = self.gcn(molecule_atoms, molecule_adjs, molecule_edges)  # [32, 85, 75]
        DT_2D_F  = self.mbca1(proteins_acids_GNN, drug_graph)
        # DT_2D_F = self.Style_Exract(molecule_atoms, molecule_adjs, proteins_acids_GNN, self.layer_gnn)  # [16,1230,75]
        t = DT_2D_F.shape[1]
        if t < 1300:
            DT_2D_Feature[:, 0:t, :] = DT_2D_F
        else:
            DT_2D_Feature = DT_2D_F[:, 0:1300, :]
        # DT_2D_Feature = DT_2D_P_att + DT_2D_D_att

        """Combine the features of two modals"""
        # DT_1D_Feature:【16,1300,256】、DT_2D_Feature：【16, 1300, 75】
        dti1d, dti1d_feature, _ = self.FC_out1(DT_1D_Feature)
        dti2d, dti2d_feature, _ = self.FC_out2(DT_2D_Feature)
        DTI = self.fusion(dti1d_feature, dti2d_feature)
        DTI = self.DTI_Pre(DTI)
        return DTI, dti1d, dti2d
    def predict(self, res):
        if res > 0.5:
            result = 1
        else:
            result = 0
        return result

    def __call__(self, data, epoch=1, train=True):
        # print(np.array(data).shape)
        l1 = 1
        l2 = 1
        l3 = 1


        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        protein_drug_interaction, dti1d, dti2d = self.forward(inputs)  # , dis_invariant

        if train:
            loss1 = F.cross_entropy(protein_drug_interaction, correct_interaction)
            loss2 = F.cross_entropy(dti1d, correct_interaction)
            loss3 = F.cross_entropy(dti2d, correct_interaction)

            return loss1 * l1 + loss2 * l2 + loss3 * l3    # loss4 # + + loss5 * l4###+ loss5 * l4 + loss6 * l5
        else:
            correct_labels = correct_interaction  # .to('cpu').data.numpy().reshape(-1)
            ys1 = F.softmax(protein_drug_interaction * 0.4 + dti1d * 0.33 + dti2d * 0.27, 1)  # .to('cpu').data.numpy()
            ys2 = F.softmax(dti1d, 1)
            ys3 = F.softmax(dti2d, 1)
            return correct_labels, ys1, ys2, ys3 # , dti1d_feature, dti2d_feature, DT_1D_Feature, DT_2D_Feature
