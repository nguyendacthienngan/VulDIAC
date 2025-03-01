import dgl
import networkx as nx
import torch
import argparse
import glob
import pickle
import os
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


class DiGraphDataEntry:
    def __init__(self, model_name="microsoft/codebert-base"):
        # 初始化 CodeBERT tokenizer
        self.device = 'cuda:2'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.encoder = RobertaModel.from_pretrained(model_name).to(device=self.device)
        self.max_length = 16
        self.node_type_map = {
            'BLOCK': 0, 
            'CALL': 1, 
            'CONTROL_STRUCTURE': 2, 
            'FIELD_IDENTIFIER': 3, 
            'IDENTIFIER': 4, 
            'JUMP_TARGET': 5, 
            'LITERAL': 6, 
            'LOCAL': 7, 
            'MEMBER': 8, 
            'METHOD_PARAMETER_IN': 9, 
            'METHOD_PARAMETER_OUT': 10, 
            'METHOD_RETURN': 11, 
            'RETURN': 12, 
            'TYPE_DECL': 13, 
            'UNKNOWN': 14
        }
        
        self.edge_type_map = {
            'CFG': 0, 
            'DOMINATE': 1, 
            'POST_DOMINATE': 2, 
            'REACHING_DEF': 3, 
        }
        
    def process_code(self, code):
        # 对代码片段进行预处理，生成 tokenizer 的输入张量
        inputs = self.tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        return inputs
    
    def encode_node_type(self, node_type):
        # 使用 one-hot 编码节点类型
        index = self.node_type_map.get(node_type, -1)
        one_hot = torch.zeros(len(self.node_type_map))
        if index != -1:
            one_hot[index] = 1.0
        return one_hot

    def encode_edge_type(self, edge_type):
        # 使用 one-hot 编码边类型
        index = self.edge_type_map.get(edge_type, -1)
        return index

    def networkx_to_dgl(self, nx_graph):
        # 1. 构建节点 ID 映射
        node_mapping = {node_id: idx for idx, node_id in enumerate(nx_graph.nodes())}
        num_nodes = len(node_mapping)
        
        # 2. 初始化 DGL 图并添加节点
        dgl_graph = dgl.DGLGraph()
        dgl_graph.add_nodes(num_nodes)
        
        # 3. 提取并设置节点特征
        input_ids_list, attention_masks, node_types, line_numbers = [], [], [], []
        
        for node_id, data in nx_graph.nodes(data=True):
            code_snippet = data.get('code', '')
            inputs = self.process_code(code_snippet)
            
            # 添加 input_ids 和 attention_mask
            input_ids_list.append(inputs['input_ids'].squeeze(0).tolist())
            attention_masks.append(inputs['attention_mask'].squeeze(0).tolist())
            
            # 添加 node_type 和 line_number
            node_type = data.get('type', 'UNKNOWN')
            node_types.append(self.encode_node_type(node_type).tolist())
            line_numbers.append(float(data.get('line_number', -1)))
            
        features = self.encoder(torch.tensor(input_ids_list).to(device=self.device), torch.tensor(attention_masks).to(device=self.device))
        mean_features = torch.mean(features.last_hidden_state, dim=1)
        # 设置节点特征
        # dgl_graph.ndata['input_ids'] = torch.tensor(input_ids_list)
        # dgl_graph.ndata['attention_mask'] = torch.tensor(attention_masks)
        dgl_graph.ndata['features'] = mean_features.to('cpu')
        dgl_graph.ndata['node_type'] = torch.tensor(node_types)
        dgl_graph.ndata['line_number'] = torch.tensor(line_numbers)

        # 4. 添加边并设置边特征
        src_nodes, dst_nodes, edge_types = [], [], []
        
        for u, v, key, edge_data in nx_graph.edges(keys=True, data=True):
            edge_type = edge_data.get('type', 'UNKNOWN')
            # wo/ REACHING_DEF
            if edge_type == "CDG":
                continue
            
            if edge_type == "DOMINATE" or edge_type == "REACHING_DEF" or edge_type == "POST_DOMINATE":
                continue
            
            # if edge_type == "DOMINATE" or edge_type == "POST_DOMINATE" or edge_type == "REACHING_DEF" or edge_type == "CDG":
            #     continue
            # else:
            src_nodes.append(node_mapping[u])
            dst_nodes.append(node_mapping[v])
            edge_types.append(self.encode_edge_type(edge_type))
            
        
        dgl_graph.add_edges(src_nodes, dst_nodes)
        
        # 设置边特征
        dgl_graph.edata['etype'] = torch.tensor(edge_types)

        return dgl_graph


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_options()
    input_path = args.input
    output_path = args.out
    # 1. 创建一个 DiGraphDataEntry 实例
    graph_transformer = DiGraphDataEntry(model_name="/home/itachi/software/codebert")
    # 2. 从 DOT 文件中读取图数据
    input_path = input_path + "/" if input_path[-1] != "/" else input_path
    output_path = output_path + "/" if output_path[-1] != "/" else output_path
    # 获取当前文件夹中所有以.pkl结尾的文件的路径
    filename = glob.glob(input_path + "/*.pkl")
    
    print('数据集路径：', input_path)
    print('输出文件路径：', output_path)
    print('样本数：', len(filename))
    
    fileS = []
    
    # 遍历当前文件夹中的每个文件
    for file in tqdm(filename):
        file_name = file.split("/")[-1].rstrip(".pkl")
        # filename = sample.split('/')[-1].rstrip(".pkl")
        with open('.././notebook/big_vul_files', 'rb') as f:
            big_vul_files = pickle.load(f)
        
        if file_name not in big_vul_files:
            continue
        
        out_pkl = output_path + file_name + '.pkl'
        # 如果文件已存在则跳过
        if os.path.exists(out_pkl):
            continue
        
        # 根据文件加载数据
        with open(file, 'rb') as f:
            nx_data = pickle.load(f)
        try:
            # 将NetworkX图转换为DGL图
            dgl_data = graph_transformer.networkx_to_dgl(nx_data)
            
        except:
            fileS.append(file)
            print("Error processing file:", file)
            continue
        
        # 保存DGL图
        # out_pkl = output_path + filename + '.pkl'
        with open(out_pkl, 'wb') as f:
            pickle.dump(dgl_data, f)

    print("Error files:", fileS)