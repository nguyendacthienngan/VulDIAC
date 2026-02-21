import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import networkx as nx
import glob
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial
import argparse
import traceback
from cpg.edge import Edge
from cpg.node import Node
import pickle as pkl


def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def get_cpg_graph(cpg):
    edges = [Edge(u, v, attrs) for u, v, attrs in cpg.edges(data=True)]
    node_dict = {node: Node(node, attrs, edges) for node, attrs in cpg.nodes(data=True)}
    new_cpg = nx.MultiDiGraph()
    except_nodes = []
    for node, attrs in node_dict.items():
        node_data = node_dict[node]
        if node_data.code == None:
            except_nodes.append(node)
            continue
        
        if node_data.line_number == None:
            except_nodes.append(node)
            continue
        
        # if node_data.type == 'METHOD':
        #     except_nodes.append(node)
        #     continue
        
        new_cpg.add_node(node, type=node_data.type, line_number=node_data.line_number, code=node_data.code)
    
    for edge in edges:
        if edge.node_in in except_nodes or edge.node_out in except_nodes:
            continue
        
        new_cpg.add_edge(edge.node_in, edge.node_out, type=edge.type)
        
    return new_cpg


def get_cfg_graph(cpg):
    cpg = get_cpg_graph(cpg)
    # 创建一个新的子图用于存储 CFG 子图
    cfg_subgraph = nx.MultiDiGraph()

    # 遍历原图中的节点和边，将与 CFG 相关的节点和边加入 CFG 子图
    # for node, data in cpg.nodes(data=True):
    # 检查该节点是否在 CFG 边的起点或终点
    for u, v, edge_data in cpg.edges(data=True):
        if edge_data.get("type") == "CFG":
            cfg_subgraph.add_node(u, **cpg.nodes[u])
            cfg_subgraph.add_node(v, **cpg.nodes[v])
            cfg_subgraph.add_edge(u, v, **edge_data)  # 添加 CFG 边及其所有数据
    
    for u, v, edge_data in cpg.edges(data=True):
        # if edge_data.get("type") == "CDG":
        #     if u in cfg_subgraph.nodes() and v in cfg_subgraph.nodes():
        #         cfg_subgraph.add_edge(u, v, **edge_data)
                
        if edge_data.get("type") == "REACHING_DEF":
            if u in cfg_subgraph.nodes() and v in cfg_subgraph.nodes():
                cfg_subgraph.add_edge(u, v, **edge_data)
                
        if edge_data.get("type") == "DOMINATE":
            if u in cfg_subgraph.nodes() and v in cfg_subgraph.nodes():
                cfg_subgraph.add_edge(u, v, **edge_data)
                
        if edge_data.get("type") == "POST_DOMINATE":
            if u in cfg_subgraph.nodes() and v in cfg_subgraph.nodes():
                cfg_subgraph.add_edge(u, v, **edge_data)
                
    return cfg_subgraph

def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    args = parser.parse_args()
    return args


def store_in_pkl(dot, out, existing_files):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    if dot_name in existing_files:
        return None
    else:
        try:
            out_pkl = out + dot_name + '.pkl'
            graph = graph_extraction(dot)
            cfg_graph = get_cfg_graph(graph)
            with open(out_pkl, 'wb') as f:
                pkl.dump(cfg_graph, f)
        except Exception as e:
            print("Error processing file:", dot)
            print("Exception message:", e)
            traceback.print_exc()  # 输出完整的错误堆栈信息
            print('This file encountered an error!')
            print('this file got an error!!!')

def main():
    args = parse_options()

    root_input = args.input.rstrip("/")
    root_output = args.out.rstrip("/")

    print("Input root :", root_input)
    print("Output root:", root_output)

    os.makedirs(root_output, exist_ok=True)

    # --------------------------------------------------
    # find all sample folders
    # --------------------------------------------------
    sample_dirs = [
        os.path.join(root_input, d)
        for d in os.listdir(root_input)
        if os.path.isdir(os.path.join(root_input, d))
    ]

    print("Total samples:", len(sample_dirs))

    for sample_dir in sample_dirs:

        sample_name = os.path.basename(sample_dir)
        print(f"\n[Processing] {sample_name}")

        dotfiles = glob.glob(os.path.join(sample_dir, "*.dot"))

        print("dotfiles:", len(dotfiles))

        if len(dotfiles) == 0:
            print("Skip (no dot files)")
            continue

        # output folder per sample
        out_path = os.path.join(root_output, sample_name)
        os.makedirs(out_path, exist_ok=True)

        existing_files = glob.glob(out_path + "/*.pkl")
        existing_files = [
            f.split('/')[-1].split('.pkl')[0]
            for f in existing_files
        ]

        # safer pool size
        pool_size = max(1, os.cpu_count() // 2)
        pool = Pool(pool_size)

        pool.map(
            partial(
                store_in_pkl,
                out=out_path + "/",
                existing_files=existing_files
            ),
            dotfiles
        )

        pool.close()
        pool.join()

    print("\n✅ All samples processed.")
import time

if __name__ == '__main__':
    print("Begin to extract CFG...")
    ts = time.time()
    main()
    print("Time cost: ", time.time() - ts)
    print("CFG extraction done!")
    
    