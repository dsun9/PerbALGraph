from collections import Counter
from itertools import permutations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from infovgae.model import VGAE
from model import GNN_Backbone
from sklearn.metrics import f1_score
from torch_geometric.utils import negative_sampling, to_networkx
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAveragePrecision,
    MulticlassF1Score,
)
from tqdm import tqdm
from utils import preprocess_graph, sparse_to_tuple


def train_gnn(cfg, fabric, params, data, labeled_nodes, epochs, lr):
    g_model = GNN_Backbone(**params)
    g_opt = torch.optim.Adam(g_model.parameters(), lr=lr, weight_decay=1e-4)
    g_model, g_opt = fabric.setup(g_model, g_opt)
    g_model.train()

    g_opt.zero_grad()
    node_criterion = torch.nn.CrossEntropyLoss()
    link_criterion = torch.nn.BCEWithLogitsLoss()
    with tqdm(total=epochs, ncols=100, desc='GNN Training', leave=False) as pbar:
        for epoch_g in range(epochs):
            z, out_cls = g_model(data.x, data.edge_index)
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index, num_nodes=data.num_nodes,
                num_neg_samples=data.train_edge_index.size(1)
            )
            edge_index = torch.cat(
                [data.train_edge_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                torch.ones((data.train_edge_index.size(1),)),
                torch.zeros((neg_edge_index.size(1),))
            ], dim=0).to(fabric.device)
            out_link = g_model.decode(z, edge_index).view(-1)
            loss = 1 * link_criterion(out_link, edge_label)
            if labeled_nodes:
                loss += 0.1 * node_criterion(out_cls[labeled_nodes], data.y[labeled_nodes] - 1)
            fabric.backward(loss)
            g_opt.step()
            g_opt.zero_grad()
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
    return g_model

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def train_infovgae(cfg, fabric, data, labeled_nodes, epochs, lr):
    g_model = VGAE(data.num_actor, data.num_asser, data.x.shape[1], 32, cfg.model.num_classes)
    g_opt = torch.optim.Adam(g_model.parameters(), lr=lr, weight_decay=1e-4)
    g_model, g_opt = fabric.setup(g_model, g_opt)
    g_model.train()

    adj_matrix = nx.adjacency_matrix(to_networkx(data, to_undirected=True, remove_self_loops=True))
    adj_matrix.eliminate_zeros()
    adj_matrix[adj_matrix > 1] = 1
    adj_matrix = adj_matrix.tolil()
    cnt = 0
    for k1 in labeled_nodes:
        for k2 in labeled_nodes:
            if k1 != k2 and data.y[k1] == data.y[k2]:
                adj_matrix[k1, k2] = 1
                adj_matrix[k2, k1] = 1
                cnt += 1
    print(f"Added {cnt} edges")

    adj_N = adj_matrix.shape[0]
    adj_train = adj_matrix
    adj_norm = preprocess_graph(adj_train)
    features = sp.coo_matrix(data.x.cpu())
    pos_weight = float(adj_N * adj_N - adj_train.sum()) / adj_train.sum() * 5.0
    print("Pos weight: {}".format(pos_weight))
    norm = adj_N * adj_N / float((adj_N * adj_N - adj_train.sum()) * 2)

    adj_norm = torch.sparse_coo_tensor(*sparse_to_tuple(adj_norm), dtype=torch.float32).to_dense()
    adj_label = torch.sparse_coo_tensor(*sparse_to_tuple(adj_train + sp.eye(adj_N)), dtype=torch.float32).to_dense()
    features = torch.sparse_coo_tensor(*sparse_to_tuple(features), dtype=torch.float32)

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    
    adj_norm = fabric.to_device(adj_norm)
    adj_label = fabric.to_device(adj_label)
    features = fabric.to_device(features)
    weight_tensor = fabric.to_device(weight_tensor)
    
    g_opt.zero_grad()
    for epoch in range(epochs):
        embed, _, _ = g_model.encode(adj_norm, features)
        A_pred = g_model.decode(embed)

        reconstruct_loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1), weight=weight_tensor)

        train_acc = get_acc(A_pred, adj_label)
        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Rec_loss: {reconstruct_loss.item():.4f}, Link_acc: {train_acc.item():.4f}")

        fabric.backward(reconstruct_loss)
        g_opt.step()
        g_opt.zero_grad()
    return g_model, adj_norm, features

@torch.no_grad()
def eval_gnn(fabric, g_model, data):
    g_model.eval()
    out = g_model(data.x, data.edge_index)[1]
    pred = out[data.test_asser_nodes]
    label = data.y[data.test_asser_nodes] - 1

    all_metrics = {}
    all_metrics['Acc(micro)'] = MulticlassAccuracy(num_classes=g_model.num_class, average='micro')
    all_metrics['Acc(macro)'] = MulticlassAccuracy(num_classes=g_model.num_class, average='macro')
    all_metrics['Acc(weighted)'] = MulticlassAccuracy(num_classes=g_model.num_class, average='weighted')
    all_metrics['AP(macro)'] = MulticlassAveragePrecision(num_classes=g_model.num_class, average='macro')
    all_metrics['AP(weighted)'] = MulticlassAveragePrecision(num_classes=g_model.num_class, average='weighted')
    all_metrics['F1(micro)'] = MulticlassF1Score(num_classes=g_model.num_class, average='micro')
    all_metrics['F1(macro)'] = MulticlassF1Score(num_classes=g_model.num_class, average='macro')
    all_metrics['F1(weighted)'] = MulticlassF1Score(num_classes=g_model.num_class, average='weighted')
    to_compare = 'F1(micro)'
    for v in all_metrics.values():
        fabric.to_device(v)

    results = {}
    for k, v in all_metrics.items():
        results[k] = v(pred, label).item()
    return results, results[to_compare]

@torch.no_grad()
def eval_infovgae(fabric, g_model, data, adj_norm, features):
    g_model.eval()
    out, _, _ = g_model.encode(adj_norm, features)
    orig_out = out.cpu().detach().numpy()
    
    label = data.y.cpu().numpy()[data.test_asser_nodes] - 1
    perm = list(permutations(range(g_model.num_class)))
    pred_perm = [np.argmax(orig_out[:, p], axis=1)[data.test_asser_nodes] for p in perm]
    
    pred = None
    f1 = None
    for pi, p in zip(perm, pred_perm):
        f1_p = f1_score(label, p, average="macro")
        if f1 is None or f1_p > f1:
            pred = p
            f1 = f1_p
    assert pred is not None

    pred = torch.zeros((len(pred), g_model.hidden2_dim)).scatter_(1, torch.tensor(pred).view(-1, 1), 1)
    label = torch.LongTensor(label)

    all_metrics = {}
    all_metrics['Acc(micro)'] = MulticlassAccuracy(num_classes=g_model.num_class, average='micro')
    all_metrics['Acc(macro)'] = MulticlassAccuracy(num_classes=g_model.num_class, average='macro')
    all_metrics['Acc(weighted)'] = MulticlassAccuracy(num_classes=g_model.num_class, average='weighted')
    all_metrics['AP(macro)'] = MulticlassAveragePrecision(num_classes=g_model.num_class, average='macro')
    all_metrics['AP(weighted)'] = MulticlassAveragePrecision(num_classes=g_model.num_class, average='weighted')
    all_metrics['F1(micro)'] = MulticlassF1Score(num_classes=g_model.num_class, average='micro')
    all_metrics['F1(macro)'] = MulticlassF1Score(num_classes=g_model.num_class, average='macro')
    all_metrics['F1(weighted)'] = MulticlassF1Score(num_classes=g_model.num_class, average='weighted')
    to_compare = 'F1(micro)'

    results = {}
    for k, v in all_metrics.items():
        results[k] = v(pred, label).item()
    return results, results[to_compare]
    

def test_gnn(cfg, fabric, data, final_labeled):
    not_masked = set(torch.where(data.y != 0)[0].tolist())
    assert len(set(final_labeled) & not_masked) == len(final_labeled)
    final_labeled = sorted(set(final_labeled))
    gnn_params = {
        "in_channels": data.x.size(1),
        "hidden_channels": cfg.model.hidden_channels,
        "out_channels": cfg.model.out_channels,
        "layer_type": cfg.model.layer_type,
        "num_layers": cfg.model.num_layers,
        "num_class": cfg.model.num_classes,
    }
    g_model = train_gnn(cfg, fabric, gnn_params, data, final_labeled, cfg.train.gnn_epochs, 0.01)
    results, compare_metric = eval_gnn(fabric, g_model, data)
    cls_dist = list(
        dict(
            sorted(Counter(data.y[list(final_labeled)].detach().cpu().numpy().tolist()).most_common())
        ).values()
    )
    return results, cls_dist, compare_metric


def test_infovgae(cfg, fabric, data, final_labeled):
    not_masked = set(torch.where(data.y != 0)[0].tolist())
    assert len(set(final_labeled) & not_masked) == len(final_labeled)
    final_labeled = sorted(set(final_labeled))
    g_model, adj_norm, features = train_infovgae(cfg, fabric, data, final_labeled, cfg.train.gnn_epochs, 0.2)
    results, compare_metric = eval_infovgae(fabric, g_model, data, adj_norm, features)
    cls_dist = list(
        dict(
            sorted(Counter(data.y[list(final_labeled)].detach().cpu().numpy().tolist()).most_common())
        ).values()
    )
    return results, cls_dist, compare_metric
