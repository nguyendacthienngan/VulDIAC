# ============================================================
# DGL GRAPH BUILDER v3 (ACFG v3 Compatible)
# VulDiac — Joern 2.0.72 SAFE VERSION
# ============================================================

import os
import glob
import argparse
import pickle
import gc

import torch
import dgl
import networkx as nx

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ------------------------------------------------------------
# CONFIG (OOM SAFE)
# ------------------------------------------------------------

MODEL_NAME = "microsoft/codebert-base"

MAX_NODE = 400          # HARD CAP → prevents OOM
MAX_LEN = 64            # token length
BATCH_SIZE = 64         # embedding batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


# ------------------------------------------------------------
# LOAD MODEL (ONCE ONLY)
# ------------------------------------------------------------
print("Loading CodeBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()
model.to(DEVICE)


# ------------------------------------------------------------
# NODE TYPE ENCODING
# ------------------------------------------------------------
TYPE_MAP = {
    "CALL": 1,
    "IDENTIFIER": 2,
    "LITERAL": 3,
    "CONTROL_STRUCTURE": 4,
    "RETURN": 5,
    "BLOCK": 6,
}


def type_id(t):
    return TYPE_MAP.get(t, 0)


# ------------------------------------------------------------
# GRAPH COMPRESSION
# ------------------------------------------------------------
def compress_graph(G):

    if G.number_of_nodes() <= MAX_NODE:
        return G

    # keep highest degree nodes
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    keep = set(n for n, _ in deg[:MAX_NODE])

    return G.subgraph(keep).copy()


# ------------------------------------------------------------
# BERT EMBEDDINGS (BATCHED)
# ------------------------------------------------------------
@torch.no_grad()
def encode_texts(texts):

    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):

        batch = texts[i:i+BATCH_SIZE]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = model(**tokens)
        cls = outputs.last_hidden_state[:, 0, :]

        embeddings.append(cls.cpu())

        del tokens, outputs, cls
        torch.cuda.empty_cache()

    return torch.cat(embeddings, dim=0)


# ------------------------------------------------------------
# BUILD DGL GRAPH
# ------------------------------------------------------------
def build_dgl_graph(nx_g):

    nx_g = compress_graph(nx_g)

    nodes = list(nx_g.nodes())

    if len(nodes) == 0:
        return None

    id_map = {nid: i for i, nid in enumerate(nodes)}

    src = []
    dst = []

    for u, v in nx_g.edges():
        if u in id_map and v in id_map:
            src.append(id_map[u])
            dst.append(id_map[v])

    g = dgl.graph((src, dst), num_nodes=len(nodes))

    # ---------- NODE FEATURES ----------
    texts = []
    types = []
    lines = []

    for n in nodes:
        data = nx_g.nodes[n]

        texts.append(data.get("code", ""))
        types.append(type_id(data.get("type", "")))
        lines.append(data.get("line_number", -1))

    emb = encode_texts(texts)

    g.ndata["feat"] = emb.half()  # fp16 saves VRAM
    g.ndata["type"] = torch.tensor(types)
    g.ndata["line"] = torch.tensor(lines)

    return g


# ------------------------------------------------------------
# SAFE SAVE
# ------------------------------------------------------------
def save_graph(g, path):

    tmp = path + ".tmp"

    with open(tmp, "wb") as f:
        pickle.dump(g, f)

    os.replace(tmp, path)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = glob.glob(args.input + "/**/*.pkl", recursive=True)

    print("CFG graphs:", len(files))

    errors = []

    for file in tqdm(files):

        try:
            with open(file, "rb") as f:
                nx_g = pickle.load(f)

            g = build_dgl_graph(nx_g)

            if g is None:
                continue

            name = os.path.basename(file)
            out = os.path.join(args.output, name)

            save_graph(g, out)

            del g, nx_g
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print("Error:", file, e)
            errors.append(file)

    print("\nFinished.")
    print("Errors:", len(errors))


# ------------------------------------------------------------
if __name__ == "__main__":
    main()