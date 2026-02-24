import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from prettytable import PrettyTable
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score
from tqdm import tqdm
import glob
import dgl
import utils

# from graph_dataloader import DiGraphDataEntry, CustomDGLDataset
from gnn_module import CausalVulGNN

def calculate_binary_metrics(labels, predictions):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    # 计算假阴性率和假阳性率
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    
    # 返回结果
    return {
        'accuracy': format(accuracy * 100, '.3f'),
        'precision': format(precision * 100, '.3f'),
        'recall': format(recall * 100, '.3f'),
        'f1_score': format(f1_score * 100, '.3f'),
        'false_negative_rate': format(fnr * 100, '.3f'),
        'false_positive_rate': format(fpr * 100, '.3f')
    }
    

class CustomDataset(Dataset):
    def __init__(self, x_file, graphs, labels): 
        super().__init__()
        self.filename = x_file
        self.graphs = graphs  # 这是一个DGLGraph对象的列表
        self.labels = labels  # 标签数据def __getitem__(self, i):
    
    def __getitem__(self, idx):
        # return super().__getitem__(idx)
        node_line_number = self.graphs[idx].ndata['line_number']
        return self.filename[idx], self.graphs[idx], node_line_number, self.labels[idx]

    def __len__(self):
        return len(self.graphs)
    
def collate(samples):
    # 将samples中的图、文本数据和标签分别合并
    filenames, graphs, line_numbers, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return filenames, batched_graph, line_numbers, labels


class GNN_Classifier():
    
    def __init__(self, config, dataset_path='devign', model_name='ffmpeg', device='cuda:0', result_save_path="/home/itachi/result/vulgraph/"):
        
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.device = device
        self.model_saved_path = result_save_path + model_name + '_gnn_trained_best_f1_.pt'

        self.checkpoint_path = os.path.join(
            result_save_path,
            "last_checkpoint.pt"
        )

        self.best_model_path = os.path.join(
            result_save_path,
            "best_model.pt"
        )
        self._config_(config)
        self.config = config # Only for saving checkpoint (Optional)
        self._initialize_()
        
    def _config_(self, config):
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.num_conv_layers = config['num_conv_layers']
        self.hidden_dim = config['hidden_dim']
        self.n_steps = config['n_steps']
        self.n_etypes = config['n_etypes']
        self.reduced_size = config['reduced_size']
        self.bert_size = config['bert_size']
        self.dense_size = config['dense_size']
        self.num_classes = config['num_classes']
        self.module = config['module']
        self.lambda_causal = config['lambda_causal']
        self.lambda_spurious = config['lambda_spurious']
        self.lambda_context = config['lambda_context']

        # Dynamically map remaining config keys to object attributes
        for key, value in config.items():
            if not hasattr(self, key):  # Avoid overwriting explicitly set attributes
                setattr(self, key, value)
    
    def _initialize_(self):
        """Initialize the GNN model."""
        # if self.module == 'VulGNN':
        #     self.model = VulGNN(
        #         output_dim=self.hidden_dim,
        #         n_steps=self.n_steps,
        #         n_etypes=self.n_etypes,
        #         reduced_size=self.reduced_size,
        #         bert_size=self.bert_size,
        #         dense_size=self.dense_size,
        #         device=self.device
        #     ).to(self.device)
        # elif self.module == 'CausalGNN':
        #     self.model = CausalVulGNN(
        #         num_conv_layers=self.num_conv_layers,
        #         hidden_dim=self.hidden_dim,
        #         n_steps=self.n_steps,
        #         n_etypes=self.n_etypes,
        #         reduced_size=self.reduced_size,
        #         bert_size=self.bert_size,
        #         dense_size=self.dense_size,
        #         device=self.device
        #     ).to(self.device)
        self.model = CausalVulGNN(
            num_conv_layers=self.num_conv_layers,
            hidden_dim=self.hidden_dim,
            n_steps=self.n_steps,
            n_etypes=self.n_etypes,
            reduced_size=self.reduced_size,
            bert_size=self.bert_size,
            dense_size=self.dense_size,
            device=self.device
        ).to(self.device)
        
    def load_dgl_data(self):
        input_path = self.dataset_path + "/" if self.dataset_path[-1] != "/" else self.dataset_path
        x_file, x_dgl, y_ = [], [], []
        filename = glob.glob(input_path + "/*.pkl")
        for file in tqdm(filename):
            filename = file.split("/")[-1].rstrip(".pkl")
            x_file.append(filename)
            label = int(filename.split("_")[-1])
            y_.append(label)
            with open(file, 'rb') as f:
                dgl_data = pickle.load(f)
            x_dgl.append(dgl_data)
            
        return x_file, x_dgl, y_

    def preparation_data(self):
        # create datasets
        x_file, x_dgl, y_ = self.load_dgl_data()
        self.dataset = CustomDataset(x_file, x_dgl, y_)
        print("数据集长度: ", len(self.dataset))
        torch.manual_seed(self.seed)
        self.train_size = int(0.8 * len(self.dataset))
        self.valid_size = int(0.1 * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.valid_size
        print("train_size: ", self.train_size)
        print("valid_size: ", self.valid_size)
        print("test_size: ", self.test_size)
        train_dataset, valid_dataset, test_dataset = random_split(self.dataset, [self.train_size, self.valid_size, self.test_size])
        # create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs)
        
        if 'bigvul' in self.model_name:
            self.loss_fn_causal = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,17])).float()).to(self.device)
        else:
            self.loss_fn_causal = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,1])).float()).to(self.device)
            
        self.loss_fn_spurious = nn.KLDivLoss(reduction='batchmean').to(self.device)
        
        if 'bigvul' in self.model_name:
            self.loss_fn_context = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,17])).float()).to(self.device)
        else:
            self.loss_fn_context = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,1])).float()).to(self.device)
    
    
    def compute_loss(self, out_causal, out_spurious, out_context, labels, causal_features, spurious_features):
        # uniform_targets = torch.ones_like(out_spurious, dtype=torch.float).to(self.device) / self.num_classes
        uniform_targets = torch.full(
            out_spurious.shape,
            1.0 / self.num_classes,
            device=self.device,
            dtype=out_spurious.dtype
        )

        # numerical stability
        out_spurious = torch.clamp(out_spurious, min=-20, max=0)

        loss_causal = self.loss_fn_causal(out_causal, labels)
        loss_spurious = self.loss_fn_spurious(out_spurious, uniform_targets)
        loss_context = self.loss_fn_context(out_context, labels)
        loss_contrastive = utils.contrastive_loss(causal_features, spurious_features)
        # print(loss_contrastive)
        
        total_loss = self.lambda_causal * loss_causal + self.lambda_spurious * loss_spurious + self.lambda_context * loss_context + self.lambda_contrastive * loss_contrastive
        return total_loss, loss_causal
    
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }

        torch.save(checkpoint, self.checkpoint_path)
        print(f"✅ Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print("⚠️ No checkpoint found. Training from scratch.")
            return 0

        print("🔄 Loading checkpoint...")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"✅ Resumed from epoch {start_epoch}")

        return start_epoch
    

    def train_epoch(self):
        self.model.train()
        total_loss, labels, predictions = 0, [], []
        
        # scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            filename, batched_graph, line_numbers, targets = data
            batched_graph = batched_graph.to(torch.device(self.device))
            # 真实标签
            targets = targets.to(torch.device(self.device))
            outputs_causal, outputs_spurious, outputs_context, c_s_feats, _ , _ = self.model(batched_graph)
            # 平均分布
            # uniform_targets = torch.ones_like(outputs_spurious, dtype=torch.float).to(self.device) / self.num_classes
            if self.module == 'CausalGNN':
                loss, loss_cau = self.compute_loss(out_causal=outputs_causal, 
                                         out_spurious=outputs_spurious, 
                                         out_context=outputs_context,
                                         labels=targets,
                                         causal_features=c_s_feats[0],
                                         spurious_features=c_s_feats[1])
            else:
                loss = self.loss_fn_causal(outputs_causal, targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            predictions.extend(torch.argmax(outputs_causal, dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())
            
        score_dict = calculate_binary_metrics(labels, predictions)
        return total_loss / len(self.train_loader), score_dict


    def evaluation(self):
        print("start evaluating...")
        self.model.eval()
        total_loss, labels, predictions = 0, [], []
        
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for _, data in progress_bar:
                filename, batched_graph, line_number, targets = data
                batched_graph = batched_graph.to(torch.device(self.device))
                targets = targets.to(torch.device(self.device))
                
                outputs_causal, outputs_spurious, outputs_context, c_s_feats, _, _ = self.model(batched_graph)
                
                if self.module == 'CausalGNN':
                    loss, loss_cau = self.compute_loss(out_causal=outputs_causal, 
                                            out_spurious=outputs_spurious, 
                                            out_context=outputs_context,
                                            labels=targets,
                                            causal_features=c_s_feats[0],
                                            spurious_features=c_s_feats[1])
                else:
                    loss = self.loss_fn_causal(outputs_causal, targets)
                
                predictions.extend(torch.argmax(outputs_causal, dim=1).cpu().numpy())
                labels.extend(targets.cpu().numpy())
                total_loss += loss.item()
                
                
        score_dict = calculate_binary_metrics(labels, predictions)
        return total_loss / len(self.valid_loader), score_dict

        
    def test(self):
        # if self.module == 'VulGNN':
        #     self.trained_model = VulGNN(output_dim=self.hidden_dim, n_steps=self.n_steps, n_etypes=self.n_etypes, reduced_size=self.reduced_size, 
        #                           bert_size=self.bert_size, dense_size=self.dense_size, device=self.device)
        # elif self.module == 'CausalGNN':
            # self.trained_model = CausalVulGNN(num_conv_layers=self.num_conv_layers, hidden_dim=self.hidden_dim, n_steps=self.n_steps, n_etypes=self.n_etypes, reduced_size=self.reduced_size, 
            #                       bert_size=self.bert_size, dense_size=self.dense_size, device=self.device)
        self.trained_model = CausalVulGNN(num_conv_layers=self.num_conv_layers, hidden_dim=self.hidden_dim, n_steps=self.n_steps, n_etypes=self.n_etypes, reduced_size=self.reduced_size, 
                            bert_size=self.bert_size, dense_size=self.dense_size, device=self.device)

        self.trained_model.load_state_dict(torch.load(self.model_saved_path))
        self.trained_model.to(self.device)
        self.trained_model.eval()
        filenames, labels, predictions = [], [], []
        atts, line_numbers = [], []
        
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                filename, batched_graph, line_number, targets = data
                batched_graph = batched_graph.to(torch.device(self.device))
                
                # multi_channel_feature = multi_channel_feature.to(torch.device(self.device))
                targets = targets.to(torch.device(self.device))
                outputs_causal, _, _, _, node_att, edge_att= self.trained_model(batched_graph)
                if self.interpreter:
                    filenames.extend(filename)
                    atts.extend([node_att, edge_att])
                    line_numbers.extend(line_number)
                    
                predictions.extend(torch.argmax(outputs_causal, dim=1).cpu().numpy())
                labels.extend(targets.cpu().numpy())
        
        score_dict = calculate_binary_metrics(labels, predictions)
        return score_dict
        
        
    def train(self, resume=False):

        start_epoch = 0

        if resume:
            start_epoch = self.load_checkpoint()
        valid_f1 = 0.0
        train_table = PrettyTable(['type', 'epoch', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'FNR', 'FPR'])
        valid_table = PrettyTable(['type', 'epoch', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'FNR', 'FPR'])
        for epoch in range(start_epoch, self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.train_epoch()
            train_table.add_row(["tra", str(epoch + 1), format(train_loss, '.4f')] + [train_score[j] for j in train_score])
            print(train_table)
            val_loss, val_score = self.evaluation()

            # ✅ SAVE EVERY EPOCH (important for Kaggle)
            self.save_checkpoint(epoch)

            valid_table.add_row(["valid", str(epoch + 1), format(val_loss, '.4f')] + [val_score[j] for j in val_score])
            print(valid_table)
            print("\n")
            if float(val_score['f1_score']) > valid_f1:
                valid_f1 = float(val_score['f1_score'])
                torch.save(self.model.state_dict(), self.model_saved_path)

        test_score = self.test()
        
        for key, value in test_score.items():
            print(f"{key}: {value}")