import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class CustomGraphConvDGL(nn.Module):
    def __init__(self, in_feats, out_feats, improved=False, cached=False, bias=True, edge_norm=True, gfn=False):
        """
        完全等效的 GCNConv 实现，基于 DGL。
        :param in_channels: 输入特征维度
        :param out_channels: 输出特征维度
        :param improved: 是否使用改进版 GCN（自环权重加2）
        :param cached: 是否缓存边归一化权重
        :param bias: 是否使用偏置
        :param edge_norm: 是否对边权重进行归一化
        :param gfn: 如果为 True，则跳过消息传递，仅执行线性变换
        """
        super(CustomGraphConvDGL, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        # 权重参数
        self.weight = nn.Parameter(torch.Tensor(self.in_feats, self.out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_result = None

    def norm(self, graph, edge_weight, num_nodes):
        """
        完全等效的边归一化处理。
        :param graph: DGLGraph 图
        :param edge_weight: 边权重
        :param num_nodes: 节点数
        :return: 归一化边权重
        """
        # graph = dgl.add_self_loop(graph)
        with graph.local_scope():
            # 如果没有边权重，初始化为1
            if edge_weight is None:
                edge_weight = torch.ones(graph.num_edges(), device=graph.device)
            # 添加自环
            graph = dgl.add_self_loop(graph)
            loop_weight = torch.full(
                (num_nodes,), 
                2.0 if self.improved else 1.0, 
                dtype=edge_weight.dtype, 
                device=edge_weight.device
            )
            edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

            # 计算归一化的边权重
            graph.edata['w'] = edge_weight
            graph.update_all(
                message_func=fn.copy_e('w', 'm'),
                reduce_func=fn.sum('m', 'deg')
            )
            deg = graph.ndata['deg']
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            graph.ndata['norm'] = deg_inv_sqrt

            # 更新边权重
            graph.apply_edges(
                lambda edges: {'norm_w': edges.src['norm'] * edges.data['w'] * edges.dst['norm']}
            )
            return graph.edata['norm_w']

    def forward(self, graph, x, edge_weight=None):
        """
        前向传播。
        :param graph: DGLGraph 图
        :param x: 节点特征
        :param edge_weight: 边权重
        :return: 更新的节点特征
        """
        x = x @ self.weight
        if self.gfn:
            return x

        # 缓存边归一化权重
        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                norm_w = self.norm(graph, edge_weight, x.size(0))
            else:
                norm_w = edge_weight
            if self.cached:
                self.cached_result = norm_w
        else:
            norm_w = self.cached_result

        # 消息传递
        with graph.local_scope():
            graph = dgl.add_self_loop(graph)
            graph.edata['norm_w'] = norm_w
            graph.ndata['h'] = x
            graph.update_all(
                message_func=fn.u_mul_e('h', 'norm_w', 'm'),
                reduce_func=fn.sum('m', 'h')
            )
            h = graph.ndata['h']

        if self.bias is not None:
            h += self.bias
        return h

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_feats}, {self.out_feats})'


