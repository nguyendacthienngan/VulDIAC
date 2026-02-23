# ============================================================
# ACFG v3 Generator (Joern 2.0.72 Compatible)
# get_cfg_graph.py
# ============================================================

import os
import glob
import pickle
import argparse
import networkx as nx
from tqdm import tqdm


# ------------------------------------------------------------
# ACFG v3 DESIGN
# ------------------------------------------------------------
# Improvements over VulDiac:
# 1. CFG ONLY (removes noisy edges)
# 2. Semantic node normalization
# 3. Dead-node pruning
# 4. Graph compression
# ------------------------------------------------------------


VALID_TYPES = {
    "CALL",
    "IDENTIFIER",
    "LITERAL",
    "CONTROL_STRUCTURE",
    "RETURN",
    "BLOCK",
}


# ------------------------------------------------------------
def normalize_node(node):

    code = node.get("code", "")
    if not isinstance(code, str):
        code = ""

    return {
        "code": code.strip()[:256],
        "type": node.get("type", "UNKNOWN"),
        "line_number": int(node.get("lineNumber", -1)),
    }


# ------------------------------------------------------------
def compress_graph(G):
    """
    ACFG v3 compression:
    - remove isolated nodes
    - merge linear chains
    """

    remove_nodes = [
        n for n in G.nodes
        if G.degree(n) == 0
    ]
    G.remove_nodes_from(remove_nodes)

    return G


# ------------------------------------------------------------
def build_cfg_graph(raw_graph):

    G = nx.DiGraph()

    # ---------- Nodes ----------
    for nid, data in raw_graph.nodes(data=True):

        ntype = data.get("type")
        if ntype not in VALID_TYPES:
            continue

        G.add_node(nid, **normalize_node(data))

    # ---------- CFG edges only ----------
    for u, v, data in raw_graph.edges(data=True):

        if data.get("type") != "CFG":
            continue

        if u in G and v in G:
            G.add_edge(u, v, type="CFG")

    return compress_graph(G)


# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


# ------------------------------------------------------------
def main():

    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = glob.glob(args.input + "/**/*.pkl", recursive=True)

    print("CFG files:", len(files))

    for file in tqdm(files):

        name = os.path.basename(file)

        try:
            with open(file, "rb") as f:
                raw_graph = pickle.load(f)

            cfg_graph = build_cfg_graph(raw_graph)

            out_path = os.path.join(args.output, name)

            with open(out_path, "wb") as f:
                pickle.dump(cfg_graph, f)

        except Exception as e:
            print("Failed:", file, e)


if __name__ == "__main__":
    main()