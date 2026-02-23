# ============================================================
# ACFG v3 → DGL Graph Builder (OOM SAFE)
# get_dgl_graph_with_bert.py
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
class ACFGv3Processor:

    def __init__(self,
                 model="microsoft/codebert-base",
                 batch_size=32,
                 max_length=32):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading CodeBERT...")

        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.encoder = RobertaModel.from_pretrained(model).to(self.device)
        self.encoder.eval()

        self.batch_size = batch_size
        self.max_length = max_length

    # --------------------------------------------------------
    def encode_batch(self, codes):

        # prevent tokenizer crash
        codes = [c if c.strip() else "empty" for c in codes]

        enc = self.tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        feats = []

        with torch.no_grad():
            for i in range(0, len(codes), self.batch_size):

                ids = enc["input_ids"][i:i+self.batch_size].to(self.device)
                mask = enc["attention_mask"][i:i+self.batch_size].to(self.device)

                out = self.encoder(ids, mask)
                emb = out.last_hidden_state.mean(dim=1)

                feats.append(emb.cpu())

        torch.cuda.empty_cache()

        return torch.cat(feats, dim=0)

    # --------------------------------------------------------
    def nx_to_dgl(self, G):

        mapping = {n: i for i, n in enumerate(G.nodes())}

        dgl_g = dgl.graph(([], []), num_nodes=len(mapping))

        codes = []
        lines = []

        for _, data in G.nodes(data=True):
            codes.append(data.get("code", ""))
            lines.append(float(data.get("line_number", -1)))

        features = self.encode_batch(codes)

        dgl_g.ndata["features"] = features
        dgl_g.ndata["line_number"] = torch.tensor(lines)

        src, dst = [], []

        for u, v in G.edges():
            src.append(mapping[u])
            dst.append(mapping[v])

        if len(src):
            dgl_g.add_edges(src, dst)

        return dgl_g


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

    print("Samples:", len(files))

    processor = ACFGv3Processor()

    errors = []

    for file in tqdm(files):

        name = os.path.basename(file)

        try:
            with open(file, "rb") as f:
                nx_graph = pickle.load(f)

            if len(nx_graph.nodes) == 0:
                continue

            dgl_graph = processor.nx_to_dgl(nx_graph)

            out_file = os.path.join(args.output, name)

            with open(out_file, "wb") as f:
                pickle.dump(dgl_graph, f)

        except Exception as e:
            print("Error:", file, e)
            errors.append(file)

    print("Finished.")
    print("Errors:", len(errors))


if __name__ == "__main__":
    main()