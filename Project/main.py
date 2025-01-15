import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, SAGEConv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp
import pandas as pd

class LipophilicityGNN(torch.nn.Module):
    def __init__(self, n_feat, h_ch=64, n_lay=3, dp=0.2):
        super(LipophilicityGNN, self).__init__()
        self.n_lay = n_lay
        self.dp = dp
        self.conv_f = GCNConv(n_feat, h_ch)
        self.convs = torch.nn.ModuleList([GCNConv(h_ch, h_ch) for _ in range(n_lay - 1)])
        self.skips = torch.nn.ModuleList([nn.Linear(h_ch, h_ch) for _ in range(n_lay - 1)])
        self.lin1 = nn.Linear(h_ch, h_ch // 2)
        self.lin2 = nn.Linear(h_ch // 2, 1)
        
    def forward(self, x, e_idx, b):
        h = F.relu(self.conv_f(x, e_idx))
        h = F.dropout(h, p=self.dp, training=self.training)
        
        for i in range(self.n_lay - 1):
            h_n = F.relu(self.convs[i](h, e_idx))
            h_s = self.skips[i](h)
            h = h_n + h_s
            h = F.dropout(h, p=self.dp, training=self.training)
        
        h = global_mean_pool(h, b)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dp, training=self.training)
        return self.lin2(h)

def load_prep_data():
    ds = MoleculeNet(root='data/lipophilicity', name='ESOL')
    dl = list(ds)
    idx = list(range(len(ds)))
    tr_idx, ts_idx = train_test_split(idx, test_size=0.2, random_state=42)
    tr_load = DataLoader([ds[i] for i in tr_idx], batch_size=64, shuffle=True)
    ts_load = DataLoader([ds[i] for i in ts_idx], batch_size=64)
    return tr_load, ts_load, ds

def train_mod(m, tr_load, opt, dev):
    m.train()
    t_loss = 0
    for d in tr_load:
        d = d.to(dev)
        opt.zero_grad()
        out = m(d.x, d.edge_index, d.batch)
        loss = F.mse_loss(out, d.y.view(-1, 1))
        loss.backward()
        opt.step()
        t_loss += loss.item() * d.num_graphs
    return t_loss / len(tr_load.dataset)

def eval_mod(m, load, dev):
    m.eval()
    pred = []
    act = []
    with torch.no_grad():
        for d in load:
            d = d.to(dev)
            out = m(d.x, d.edge_index, d.batch)
            pred.extend(out.cpu().numpy())
            act.extend(d.y.cpu().numpy())
    pred = np.array(pred).flatten()
    act = np.array(act)
    r2 = r2_score(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))
    mae = mean_absolute_error(act, pred)
    return r2, rmse, mae

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr_load, ts_load, ds = load_prep_data()
    m = LipophilicityGNN(n_feat=ds[0].x.shape[1], h_ch=64, n_lay=3, dp=0.2).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    ep = 100
    tr_loss = []
    ts_met = []
    
    for e in range(ep):
        tr_l = train_mod(m, tr_load, opt, dev)
        r2, rmse, mae = eval_mod(m, ts_load, dev)
        tr_loss.append(tr_l)
        ts_met.append((r2, rmse, mae))
        
        if (e + 1) % 10 == 0:
            print(f'Epoch {e+1:03d}, Loss: {tr_l:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(tr_loss)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    plt.subplot(132)
    plt.plot([m[0] for m in ts_met])
    plt.title('R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    
    plt.subplot(133)
    plt.plot([m[1] for m in ts_met])
    plt.title('RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
