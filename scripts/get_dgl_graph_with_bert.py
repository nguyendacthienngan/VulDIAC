# ============================================================
# ACFG v3 → DGL Builder (VulDiac Compatible)
# Joern 2.0.72 + Kaggle Safe
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
# ARGUMENTS
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=32)
    return parser.parse_args()


# ------------------------------------------------------------
# ACFG v3 DGL BUILDER
# ------------------------------------------------------------
class DGLBuilder:

    def __init__(self, batch_size, max_length):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading CodeBERT...")
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )
        self.encoder = RobertaModel.from_pretrained(
            "microsoft/codebert-base"
        ).to(self.device)

        self.encoder.eval()

        self.batch_size = batch_size
        self.max_length = max_length

        self.node_types = {
            'BLOCK':0,'CALL':1,'CONTROL_STRUCTURE':2,
            'IDENTIFIER':3,'LITERAL':4,'RETURN':5,
            'UNKNOWN':6
        }

    # --------------------------------------------------------
    def encode_batch(self, codes):

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

        vec = torch.zeros(len(self.node_types))
        vec[self.node_types.get(t, 6)] = 1
        return vec

    # --------------------------------------------------------
    def build(self, G):

        mapping = {n:i for i,n in enumerate(G.nodes())}
        g = dgl.graph(([],[]), num_nodes=len(mapping))

        codes = []
        types = []
        lines = []

        # ---------- NODE FEATURES ----------
        for _, data in G.nodes(data=True):

            code = data.get("code", "")
            if not isinstance(code, str):
                code = ""

            codes.append(code)

            types.append(
                self.node_type_vec(data.get("type","UNKNOWN"))
            )

            # ⭐ HARD GUARANTEE (FIX)
            line = data.get("line_number", -1)

            try:
                line = int(line)
            except:
                line = -1

            lines.append(float(line))

        if len(codes) == 0:
            return None

        feats = self.encode_batch(codes)

        g.ndata["features"] = feats
        g.ndata["node_type"] = torch.stack(types)

        # ⭐ ALWAYS EXISTS NOW
        g.ndata["line_number"] = torch.tensor(lines)

        # ---------- EDGES ----------
        src, dst = [], []

        for u,v,data in G.edges(data=True):
            if data.get("type") != "CFG":
                continue
            src.append(mapping[u])
            dst.append(mapping[v])

        if len(src) > 0:
            g.add_edges(src,dst)

        return g


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = glob.glob(args.input + "/**/*.pkl", recursive=True)

    print("CFG graphs:", len(files))

    builder = DGLBuilder(
        args.batch_size,
        args.max_length
    )

    for file in tqdm(files):

        try:
            with open(file, "rb") as f:
                G = pickle.load(f)

            dgl_graph = builder.build(G)

            if dgl_graph is None:
                continue

            name = os.path.basename(file)
            out = os.path.join(args.output, name)

            with open(out, "wb") as f:
                pickle.dump(dgl_graph, f)

        except Exception as e:
            print("Failed:", file, e)


if __name__ == "__main__":
    main()
