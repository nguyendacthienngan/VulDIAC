# ============================================================
# ACFG v3 CFG GRAPH GENERATOR
# Joern 2.0.72 + NetworkX 3.x Compatible
# Research-grade + Kaggle-safe
# ============================================================

import os
import glob
import pickle
import argparse
import networkx as nx
from tqdm import tqdm


# ============================================================
# VALID NODE TYPES (ACFG v3)
# ============================================================

VALID_TYPES = {
    "METHOD",
    "METHOD_PARAMETER_IN",
    "METHOD_PARAMETER_OUT",
    "METHOD_RETURN",
    "BLOCK",
    "CALL",
    "IDENTIFIER",
    "LITERAL",
    "CONTROL_STRUCTURE",
    "RETURN",
    "FIELD_IDENTIFIER",
    "JUMP_TARGET",
    "UNKNOWN"
}



# ============================================================
# UTILITIES
# ============================================================
def extract_label(dot_path):
    """
    Extract label from parent folder name.

    sample_77_1/func.dot → 1
    """
    folder = os.path.basename(os.path.dirname(dot_path))

    try:
        label = int(folder.split("_")[-1])
    except:
        label = 0

    return label


def clean_attr(x):
    """Remove DOT quotes safely"""
    if x is None:
        return ""
    return str(x).strip('"')


def normalize_node(attrs):

    code = clean_attr(attrs.get("code", ""))
    typ = clean_attr(attrs.get("type", "UNKNOWN"))

    try:
        line = int(clean_attr(attrs.get("line_number", -1)))
    except:
        line = -1

    return {
        "code": code[:256],
        "type": typ,
        "line_number": line,
    }


# ============================================================
# GRAPH COMPRESSION (OOM PREVENTION)
# ============================================================

def compress_graph(G):
    """
    ACFG v3 compression:
      1. remove isolated nodes
      2. remove tiny useless nodes
    """

    remove_nodes = [
        n for n in G.nodes
        if G.degree(n) == 0
    ]

    G.remove_nodes_from(remove_nodes)

    return G


# ============================================================
# BUILD CFG GRAPH
# ============================================================

def build_cfg_graph(dot_graph):

    G = nx.DiGraph()

    # ---------- Nodes ----------
    for nid, attrs in dot_graph.nodes(data=True):

        ntype = clean_attr(attrs.get("type"))

        if ntype not in VALID_TYPES:
            print(ntype + " is not valid")
            continue

        G.add_node(nid, **normalize_node(attrs))

    # ---------- CFG edges ----------
    for u, v, attrs in dot_graph.edges(data=True):

        etype = clean_attr(attrs.get("type"))

        if etype != "CFG":
            continue

        if u in G and v in G:
            G.add_edge(u, v, type="CFG")

    return compress_graph(G)


# ============================================================
# SAVE GRAPH (NetworkX 3.x SAFE)
# ============================================================

def save_graph(G, out_path):

    if len(G.nodes) == 0:
        return False

    with open(out_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    return True


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = glob.glob(args.input + "/**/*.dot", recursive=True)

    print("DOT files:", len(files))

    saved = 0
    skipped = 0

    for file in tqdm(files):

        # name = os.path.basename(file).replace(".dot", ".pkl")
        # out = os.path.join(args.output, name)

        try:
            # ✅ READ DOT GRAPH
            dot_graph = nx.drawing.nx_pydot.read_dot(file)

            G = build_cfg_graph(dot_graph)

            label = extract_label(file)

            base = os.path.basename(file).replace(".dot", "")
            out_name = f"{base}_{label}.pkl"

            out = os.path.join(args.output, out_name)

            if save_graph(G, out):
                saved += 1
            else:
                skipped += 1

        except Exception as e:
            print("Failed:", file, e)
            skipped += 1

    print("\n==============================")
    print("Saved graphs :", saved)
    print("Skipped      :", skipped)
    print("==============================")


# ============================================================

if __name__ == "__main__":
    main()