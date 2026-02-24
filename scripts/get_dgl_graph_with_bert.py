# ============================================================
# DGL Builder v3 — VulDiac Compatible (FINAL FIX)
# ============================================================

import os
import glob
import pickle
import argparse
import torch
import dgl
import networkx as nx
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
NODE_TYPE_MAP = {
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

EDGE_TYPE_MAP = {"CFG": 0}


# ------------------------------------------------------------
class DGLBuilder:

    def __init__(self,
                 model_name="microsoft/codebert-base",
                 batch_size=128,
                 max_length=16):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading CodeBERT...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.encoder = RobertaModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

        self.batch_size = batch_size
        self.max_length = max_length

    # --------------------------------------------------------
    def encode_batch(self, codes):

        if len(codes) == 0:
            return torch.zeros((0, 768))

        enc = self.tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        ids = enc["input_ids"]
        mask = enc["attention_mask"]

        outputs = []

        with torch.no_grad():
            for i in range(0, len(ids), self.batch_size):

                out = self.encoder(
                    ids[i:i+self.batch_size].to(self.device),
                    mask[i:i+self.batch_size].to(self.device)
                )

                feat = out.last_hidden_state.mean(dim=1)
                outputs.append(feat.cpu())

        return torch.cat(outputs, dim=0)

    # --------------------------------------------------------
    def node_type_vec(self, t):

        vec = torch.zeros(len(NODE_TYPE_MAP))
        idx = NODE_TYPE_MAP.get(t, 14)
        vec[idx] = 1
        return vec

    # --------------------------------------------------------
    def build(self, nx_graph):

        if len(nx_graph.nodes) == 0:
            return None

        node_map = {n: i for i, n in enumerate(nx_graph.nodes())}
        num_nodes = len(node_map)

        # ---------- Nodes ----------
        codes = []
        node_types = []
        line_numbers = []

        for _, data in nx_graph.nodes(data=True):

            code = data.get("code", "")
            if not isinstance(code, str):
                code = ""

            codes.append(code)
            node_types.append(
                self.node_type_vec(data.get("type", "UNKNOWN"))
            )

            line_numbers.append(
                float(data.get("line_number", -1))
            )

        features = self.encode_batch(codes)

        # ---------- Graph ----------
        g = dgl.graph(([], []), num_nodes=num_nodes)

        g.ndata["features"] = features
        g.ndata["node_type"] = torch.stack(node_types)
        g.ndata["line_number"] = torch.tensor(line_numbers)

        # ---------- Edges ----------
        src, dst = [], []

        for u, v, data in nx_graph.edges(data=True):
            if data.get("type") != "CFG":
                continue
            src.append(node_map[u])
            dst.append(node_map[v])

        if len(src) > 0:
            g.add_edges(src, dst)
            g.edata["etype"] = torch.zeros(len(src), dtype=torch.long)
        else:
            # ⭐ CRITICAL FIX
            g.edata["etype"] = torch.zeros(0, dtype=torch.long)

        return g


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

    print("CFG graphs:", len(files))

    builder = DGLBuilder()

    errors = 0
    saved = 0

    for file in tqdm(files):

        name = os.path.basename(file)

        try:
            with open(file, "rb") as f:
                nx_graph = pickle.load(f)

            g = builder.build(nx_graph)

            if g is None:
                continue

            out = os.path.join(args.output, name)

            with open(out, "wb") as f:
                pickle.dump(g, f)

            saved += 1

        except Exception as e:
            errors += 1
            print("Failed:", file, e)

    print("\nSaved:", saved)
    print("Errors:", errors)


if __name__ == "__main__":
    main()
