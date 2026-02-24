# GCN, RGCN, GAT, GGNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import RobertaModel
import dgl
from dgl.nn import GatedGraphConv, GraphConv, GATConv, RelGraphConv

from graph_conv import CustomGraphConvDGL


class CausalVulGNN(torch.nn.Module):
    def __init__(self, num_conv_layers = 2, hidden_dim=128, n_steps=2, n_etypes=5, reduced_size=128, 
                 bert_size=768, dense_size=32, device='cuda:0', dropout=0, num_classes=2):
        super(CausalVulGNN, self).__init__()
        
        self.num_classes = num_classes
        self.device = device

        # 节点特征降维
        self.node_feature_reducer = nn.Linear(bert_size, reduced_size)

        # Dropout 层
        self.dropout = nn.Dropout(p=dropout)
        
        self.bn_feature = nn.BatchNorm1d(reduced_size)
        
        # RGCN 模块
        self.gnn = RelGraphConv(in_feat=reduced_size, out_feat=hidden_dim, num_rels=n_etypes)
        
        # self.gnn = GATConv(in_feats=reduced_size, out_feats=hidden_dim, num_heads=4)

        self.num_conv_layers = num_conv_layers
        # 创建卷积层和批归一化层的模块列表
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        # 添加多个图卷积层和批归一化层
        for i in range(num_conv_layers):
            self.bns_conv.append(nn.BatchNorm1d(hidden_dim))  # 添加批归一化层
            self.convs.append(CustomGraphConvDGL(in_feats=hidden_dim, out_feats=hidden_dim))   # 添加图卷积层
        
        # 注意力机制
        self.edge_att_mlp = nn.Linear(hidden_dim * 2, 2)  # 边注意力
        self.node_att_mlp = nn.Linear(hidden_dim, 2)  # 节点注意力

        self.bn_causal = nn.BatchNorm1d(hidden_dim)
        
        self.bn_spurious = nn.BatchNorm1d(hidden_dim)

        self.causal_convs = CustomGraphConvDGL(in_feats=hidden_dim, out_feats=hidden_dim)
        
        self.spurious_convs = CustomGraphConvDGL(in_feats=hidden_dim, out_feats=hidden_dim)

        
        # 因果分支
        self.causal_fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.causal_fc1 = nn.Linear(hidden_dim, dense_size)
        self.causal_fc2_bn = nn.BatchNorm1d(dense_size)
        self.causal_fc2 = nn.Linear(dense_size, self.num_classes)

        # 上下文虚假特征分支
        self.spurious_fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.spurious_fc1 = nn.Linear(hidden_dim, dense_size)
        self.spurious_fc2_bn = nn.BatchNorm1d(dense_size)
        self.spurious_fc2 = nn.Linear(dense_size, self.num_classes)

        # 联合分支
        self.concat_fc1_bn = nn.BatchNorm1d(hidden_dim * 2)
        self.concat_fc1 = nn.Linear(hidden_dim * 2, dense_size)
        self.concat_fc2_bn = nn.BatchNorm1d(dense_size)
        self.concat_fc2 = nn.Linear(dense_size, self.num_classes)
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, batched_graph):
        # 提取节点和边特征
        features = batched_graph.ndata['features']
        etypes = batched_graph.edata['etype']

        # 节点特征降维
        features = self.node_feature_reducer(features)

        features = self.bn_feature(features)

        # GNN 部分
        x = F.relu(self.gnn(batched_graph, features, etypes))
        
        # print(x.shape)
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(batched_graph, x))

        # 获取边的索引
        edge_index = batched_graph.edges()
        row, col = edge_index[0], edge_index[1]

        # 计算边注意力
        edge_rep = torch.cat([x[row], x[col]], dim=-1)  # 拼接边两端的特征
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)  

        edge_weight_causal = edge_att[:, 0]
        edge_weight_spurious = edge_att[:, 1]

        # 计算节点注意力
        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        # print(node_att.shape)
        # print(node_att)
        # print(node_att[:, 0].view(-1, 1))
        # 因果特征
        x_causal = node_att[:, 0].view(-1, 1) * x
        # 虚假特征
        x_spurious = node_att[:, 1].view(-1, 1) * x
        
        x_causal = F.relu(self.causal_convs(batched_graph, self.bn_causal(x_causal), edge_weight_causal))
        x_spurious = F.relu(self.spurious_convs(batched_graph, self.bn_spurious(x_spurious), edge_weight_spurious))
        
        # 读出特征（全局池化）
        batched_graph.ndata['x_causal'] = x_causal
        batched_graph.ndata['x_spurious'] = x_spurious
        
        x_causal = dgl.mean_nodes(batched_graph, 'x_causal')
        x_spurious = dgl.mean_nodes(batched_graph, 'x_spurious')
        
        out_causal = self.causal_readout_layer(x_causal)
        out_spurious = self.spurious_readout_layer(x_spurious)
        out_context = self.random_readout_layer(x_causal, x_spurious)

        return out_causal, out_spurious, out_context, [x_causal, x_spurious], node_att.cpu().detach().numpy(), edge_att.cpu().detach().numpy()
    
    def causal_readout_layer(self, x):
        x = self.causal_fc1_bn(x)
        x = self.causal_fc1(x)
        x = F.relu(x)
        x = self.causal_fc2_bn(x)
        x = self.causal_fc2(x)
        # out = F.softmax(x, dim=-1)
        return x
    
    def spurious_readout_layer(self, x):
        x = self.spurious_fc1_bn(x)
        x = self.spurious_fc1(x)
        x = F.relu(x)
        x = self.spurious_fc2_bn(x)
        x = self.spurious_fc2(x)
        out = F.log_softmax(x, dim=-1)
        return out
    
    def random_readout_layer(self, x_causal, x_spurious, random_flag=True):
        num = x_spurious.shape[0]
        l = [i for i in range(num)]
        if random_flag:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        # 
        x = torch.cat((x_causal, x_spurious[random_idx]), dim=1)
        x = self.concat_fc1_bn(x)
        x = self.concat_fc1(x)
        x = F.relu(x)
        x = self.concat_fc2_bn(x)
        x = self.concat_fc2(x)
        # out = F.softmax(x, dim=-1)
        return x

