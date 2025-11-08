import torch
import torch.nn as nn
from collections import OrderedDict

'''protein sequcence feature extraction module（MCCN）'''
class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class Protein_Seq_Representation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num,out_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, out_dim, kernel_size=3)
            )

        self.linear = nn.Linear(block_num * out_dim, out_dim)

    def forward(self, x):
        # x: torch.Size([512, 900]), After embedding: torch.Size([512, 900, 128])
        # x: [batch, sequence_length], After embedding: [batch, sequence_length, embedding]
        x = self.embed(x).permute(0, 2, 1) 
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x



'''protein k-mer feature extraction module（FCNN）'''
class Protein_Kmer_Representation(nn.Module):
    def __init__(self,hp):
        super(Protein_Kmer_Representation, self).__init__()
        kmer_feature_dim = 8420
        output_dim = hp.module_out_dim

        self.relu = nn.ReLU()
        self.fc_km1 = nn.Linear(kmer_feature_dim, 1024)
        self.fc_km2 = nn.Linear(1024, 512)
        self.fc_km3 = nn.Linear(512, output_dim)

    def forward(self,kmer):
        kmer_1 = self.relu(self.fc_km1(kmer))
        kmer_2 = self.relu(self.fc_km2(kmer_1))
        kmer_3 = self.relu(self.fc_km3(kmer_2))

        return kmer_3

