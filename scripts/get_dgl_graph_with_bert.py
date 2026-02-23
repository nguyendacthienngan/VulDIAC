import dgl
import networkx as nx
import torch
import argparse
import glob
import pickle
import os
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


# =========================
# Graph Processor
# =========================
class DiGraphDataEntry:

    def __init__(self,
                 model_name="microsoft/codebert-base",
                 batch_size=128,
                 max_length=16):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading CodeBERT...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.encoder = RobertaModel.from_pretrained(model_name).to(self.device)

        # 🔥 IMPORTANT
        self.encoder.eval()

        self.batch_size = batch_size
        self.max_length = max_length

        self.node_type_map = {
            'BLOCK': 0,'CALL': 1,'CONTROL_STRUCTURE': 2,'FIELD_IDENTIFIER': 3,
            'IDENTIFIER': 4,'JUMP_TARGET': 5,'LITERAL': 6,'LOCAL': 7,
            'MEMBER': 8,'METHOD_PARAMETER_IN': 9,'METHOD_PARAMETER_OUT': 10,
            'METHOD_RETURN': 11,'RETURN': 12,'TYPE_DECL': 13,'UNKNOWN': 14
        }

        self.edge_type_map = {
            'CFG': 0,
            'DOMINATE': 1,
            'POST_DOMINATE': 2,
            'REACHING_DEF': 3,
        }

    # =========================
    # Batch CodeBERT Encoding
    # =========================
    def encode_codebert_batch(self, codes):

        encodings = self.tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        outputs = []

        with torch.no_grad():
            for i in range(0, len(input_ids), self.batch_size):

                batch_ids = input_ids[i:i+self.batch_size].to(self.device)
                batch_mask = attention_mask[i:i+self.batch_size].to(self.device)

                out = self.encoder(batch_ids, batch_mask)
                feat = out.last_hidden_state.mean(dim=1)

                outputs.append(feat.cpu())

        return torch.cat(outputs, dim=0)

    # =========================
    def encode_node_type(self, node_type):
        vec = torch.zeros(len(self.node_type_map))
        idx = self.node_type_map.get(node_type, 14)
        vec[idx] = 1
        return vec

    # =========================
    def networkx_to_dgl(self, nx_graph):

        node_mapping = {n: i for i, n in enumerate(nx_graph.nodes())}
        num_nodes = len(node_mapping)

        dgl_graph = dgl.graph(([], []), num_nodes=num_nodes)

        # -------- Collect node info --------
        codes = []
        node_types = []
        line_numbers = []

        for _, data in nx_graph.nodes(data=True):

            code = data.get("code")
            if not isinstance(code, str):
                code = ""

            codes.append(code)
            node_types.append(
                self.encode_node_type(data.get("type", "UNKNOWN"))
            )
            line_numbers.append(float(data.get("line_number", -1)))

        # -------- FAST BERT --------
        features = self.encode_codebert_batch(codes)

        dgl_graph.ndata["features"] = features
        dgl_graph.ndata["node_type"] = torch.stack(node_types)
        dgl_graph.ndata["line_number"] = torch.tensor(line_numbers)

        # -------- Edges --------
        src, dst, etype = [], [], []

        if isinstance(nx_graph, nx.MultiDiGraph):
            edge_iter = nx_graph.edges(keys=True, data=True)
        else:
            edge_iter = ((u, v, None, d)
                         for u, v, d in nx_graph.edges(data=True))

        for u, v, _, data in edge_iter:

            t = data.get("type", "CFG")

            # keep CFG only (VulDiac practice)
            if t != "CFG":
                continue

            src.append(node_mapping[u])
            dst.append(node_mapping[v])
            etype.append(self.edge_type_map["CFG"])

        if len(src) > 0:
            dgl_graph.add_edges(src, dst)
            dgl_graph.edata["etype"] = torch.tensor(etype)
        else:
            dgl_graph.edata["etype"] = torch.zeros(0, dtype=torch.long)

        return dgl_graph


# =========================
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--out", required=True)
    return parser.parse_args()


# =========================
def main():

    args = parse_options()

    input_path = args.input.rstrip("/")
    output_path = args.out.rstrip("/")

    os.makedirs(output_path, exist_ok=True)

    files = glob.glob(input_path + "/**/*.pkl", recursive=True)

    print("Samples:", len(files))

    transformer = DiGraphDataEntry()

    error_files = []

    for file in tqdm(files):

        file_name = os.path.basename(file).replace(".pkl", "")

        # ⭐ Skip global graphs
        if file_name.startswith("_global"):
            continue

        out_file = f"{output_path}/{file_name}.pkl"

        if os.path.exists(out_file):
            continue

        try:
            with open(file, "rb") as f:
                nx_graph = pickle.load(f)

            dgl_graph = transformer.networkx_to_dgl(nx_graph)

            with open(out_file, "wb") as f:
                pickle.dump(dgl_graph, f)

        except Exception as e:
            print("Error:", file, e)
            error_files.append(file)

    print("Finished.")
    print("Errors:", len(error_files))


if __name__ == "__main__":
    main()
