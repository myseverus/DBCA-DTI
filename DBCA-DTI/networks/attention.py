import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from .KAA_GAT import KAAGATConv


class MolecularGAT0(nn.Module):
    def __init__(self, atom_dim, edge_dim, output_dim=75, heads=1, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.output_dim = output_dim

        # 仅使用一层GAT，确保输出维度匹配
        self.gat = KAAGATConv(
            in_channels=atom_dim,  # 输入维度：75
            out_channels=output_dim,  # 输出维度：75（单头情况下）
            heads=heads,  # 注意力头数：1（保证输出维度不扩展）
            edge_dim=edge_dim,  # 边特征维度：4
            concat=False,  # 不拼接多头结果（单头时直接输出）
            dropout=dropout
        )

        # 可选：添加批归一化稳定训练
        # self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, atoms, adjs, edges):
        """
        输入形状:
        atoms: [batch_size, num_nodes, atom_dim] -> [32, 85, 75]
        adjs:  [batch_size, num_nodes, num_nodes] -> [32, 85, 85]
        edges: [batch_size, num_nodes, num_nodes, edge_dim] -> [32, 85, 85, 4]
        输出形状: [32, 85, 75]
        """
        batch_size, num_nodes = atoms.shape[:2]

        # 展平为 [batch*num_nodes, atom_dim] 适应GAT输入格式
        x = atoms.view(-1, atoms.size(-1))  # [32*85, 75]

        # 生成边索引和边特征（批量处理）
        edge_indices = []
        edge_attrs = []
        for i in range(batch_size):
            # 提取第i个样本的邻接矩阵
            adj_mask = (adjs[i] > 0.5).float()  # 二值化邻接矩阵
            edge_index = adj_mask.nonzero(as_tuple=False).t()  # [2, num_edges]

            # 提取对应边的特征
            edge_attr = edges[i][edge_index[0], edge_index[1]]  # [num_edges, 4]

            # 为边索引添加批次偏移（区分不同样本的节点）
            edge_index += i * num_nodes
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)

        # 合并所有批次的边信息
        edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]
        edge_attr = torch.cat(edge_attrs, dim=0)  # [total_edges, 4]

        # 单次GAT处理
        x = self.gat(x, edge_index, edge_attr=edge_attr)  # [32*85, 75]

        # 恢复批次维度 [32, 85, 75]
        x = x.view(batch_size, num_nodes, -1)

        return x


class MolecularGAT1(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden_dim=75, heads=8, output_dim=75):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim

        # 第一层GAT：多头注意力提取特征
        self.gat1 = KAAGATConv(
            in_channels=atom_dim,
            out_channels=hidden_dim,
            heads=heads,  # 多头注意力
            edge_dim=edge_dim,
            concat=True,  # 拼接多头结果，输出维度=hidden_dim*heads
            dropout=0.2
        )

        # 第二层GAT：特征转换
        self.gat2 = KAAGATConv(
            in_channels=hidden_dim * heads,  # 接收多头拼接的维度
            out_channels=hidden_dim,
            heads=heads,  # 再次使用多头
            edge_dim=edge_dim,
            concat=True,  # 拼接多头结果
            dropout=0.2
        )

        # 第三层GAT：输出目标维度
        self.gat3 = KAAGATConv(
            in_channels=hidden_dim * heads,  # 接收第二层多头拼接的维度
            out_channels=output_dim,
            heads=1,  # 单头输出
            edge_dim=edge_dim,
            concat=False,  # 不拼接，直接输出目标维度
            dropout=0.0
        )

    def forward(self, atoms, adjs, edges):
        """
        输入形状:
        atoms: [batch_size, num_nodes, atom_dim]  # [32, 85, 75]
        adjs:  [batch_size, num_nodes, num_nodes]  # [32, 85, 85]
        edges: [batch_size, num_nodes, num_nodes, edge_dim]  # [32, 85, 85, 4]
        输出形状: [32, 85, 75]
        """
        batch_size, num_nodes = atoms.shape[:2]

        # 合并为批量大图（适应PyTorch Geometric的输入格式）
        x = atoms.view(-1, atoms.size(-1))  # [batch*num_nodes, atom_dim] -> [32*85, 75]

        # 生成边索引和边特征（跨批次处理）
        edge_indices = []
        edge_attrs = []
        for i in range(batch_size):
            # 提取邻接矩阵中的边
            adj_mask = (adjs[i] > 0.5).float()  # 二值化邻接矩阵
            edge_index = adj_mask.nonzero(as_tuple=False).t()  # [2, num_edges]

            # 提取对应边的特征
            edge_attr = edges[i][edge_index[0], edge_index[1]]  # [num_edges, edge_dim]

            # 偏移边索引（区分不同批次的节点）
            edge_index += i * num_nodes
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)

        # 合并所有批次的边
        edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]
        edge_attr = torch.cat(edge_attrs, dim=0)  # [total_edges, edge_dim]

        # 三次GAT特征提取
        x = self.gat1(x, edge_index, edge_attr=edge_attr)  # [32*85, 75*8]
        x = self.gat2(x, edge_index, edge_attr=edge_attr)  # [32*85, 75*8]
        x = self.gat3(x, edge_index, edge_attr=edge_attr)  # [32*85, 75]

        # 恢复批次维度
        x = x.view(batch_size, num_nodes, -1)  # [32, 85, 75]

        return x


class MolecularGAT2(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden_dim=75, heads=8, output_dim=75):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim

        # 第1层GAT：多头注意力，提升特征维度
        self.gat1 = KAAGATConv(
            in_channels=atom_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,  # 拼接多头结果
            dropout=0.2
        )

        # 第2层GAT：单头，降低维度到hidden_dim
        self.gat2 = KAAGATConv(
            in_channels=hidden_dim * heads,  # 输入是上一层多头拼接的结果
            out_channels=hidden_dim,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
            dropout=0.2
        )

        # 第3层GAT：再次使用多头注意力
        self.gat3 = KAAGATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=0.2
        )

        # 第4层GAT：输出最终维度
        self.gat4 = KAAGATConv(
            in_channels=hidden_dim * heads,
            out_channels=output_dim,  # 最终输出维度75
            heads=1,
            edge_dim=edge_dim,
            concat=False,
            dropout=0.0
        )


    def forward(self, atoms, adjs, edges):
        """
        输入形状:
        atoms: [batch_size, num_nodes, atom_dim] -> [32, 85, 75]
        adjs:  [batch_size, num_nodes, num_nodes] -> [32, 85, 85]
        edges: [batch_size, num_nodes, num_nodes, edge_dim] -> [32, 85, 85, 4]
        输出形状: [32, 85, 75]
        """
        batch_size, num_nodes = atoms.shape[:2]

        # 合并为批量大图（适应PyTorch Geometric的输入格式）
        x = atoms.view(-1, atoms.size(-1))  # [batch*num_nodes, atom_dim] -> [32*85, 75]

        # 生成边索引和边特征
        edge_indices = []
        edge_attrs = []
        for i in range(batch_size):
            # 提取邻接矩阵中的边
            adj_mask = (adjs[i] > 0.5).float()  # 二值化邻接矩阵
            edge_index = adj_mask.nonzero(as_tuple=False).t()  # [2, num_edges]

            # 提取对应边的特征
            edge_attr = edges[i][edge_index[0], edge_index[1]]  # [num_edges, edge_dim]

            # 添加批次偏移（因为所有图被合并成了一个大图）
            edge_index += i * num_nodes
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)

        # 合并所有批次的边
        edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]
        edge_attr = torch.cat(edge_attrs, dim=0)  # [total_edges, edge_dim]

        # 第1次GAT
        x = self.gat1(x, edge_index, edge_attr=edge_attr)  # [32*85, 75*8]
        # 第2次GAT
        x = self.gat2(x, edge_index, edge_attr=edge_attr)  # [32*85, 75]


        # 第3次GAT
        x = self.gat3(x, edge_index, edge_attr=edge_attr)  # [32*85, 75*8]

        # 第4次GAT
        x = self.gat4(x, edge_index, edge_attr=edge_attr)  # [32*85, 75]


        # 恢复批次维度
        x = x.view(batch_size, num_nodes, -1)  # [32, 85, 75]

        return x


# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = MolecularGAT1(
        atom_dim=75,
        edge_dim=4,
        hidden_dim=75,
        heads=8,
        output_dim=75
    )

    # 创建测试输入
    batch_size = 32
    num_nodes = 85
    atom_dim = 75
    edge_dim = 4

    molecule_atoms = torch.randn(batch_size, num_nodes, atom_dim)  # [32, 85, 75]
    molecule_adjs = torch.randn(batch_size, num_nodes, num_nodes)  # [32, 85, 85]
    molecule_edges = torch.randn(batch_size, num_nodes, num_nodes, edge_dim)  # [32, 85, 85, 4]

    # 前向传播
    output = model(molecule_atoms, molecule_adjs, molecule_edges)

    # 检查输出形状
    print(f"输出形状: {output.shape}")  # 应输出: torch.Size([32, 85, 75])
