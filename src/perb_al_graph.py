import logging
import math
from itertools import chain
from pathlib import Path

import hydra
import igraph as ig
import lightning as L
import networkx as nx
import numpy as np
import pandas as pd
import torch
from lightning.fabric.loggers import CSVLogger
from model import GNN_Backbone
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.utils import (
    add_random_edge,
    dropout_edge,
    dropout_path,
    negative_sampling,
    to_networkx,
)
from tqdm import tqdm
from utils import JSD, perc

torch.set_float32_matmul_precision('medium')
log = logging.getLogger(__name__)
RELATIVE_ROOT = (Path(__file__).parent / "..").resolve()
CONFIG_PATH = str((RELATIVE_ROOT / "conf").resolve())
DATA_FOLDER = RELATIVE_ROOT / "data"

def setup(cfg):
    data = torch.load(DATA_FOLDER / cfg.dataset.name / "processed" / "data.pt")
    cache_folder = DATA_FOLDER / cfg.dataset.name / "cache"
    G_cache = cache_folder / "G.pkl"
    G_cache.parent.mkdir(parents=True, exist_ok=True)
    if G_cache.exists():
        log.info(f"Loading G from cache \"{G_cache.relative_to(RELATIVE_ROOT)}\"")
        G = pd.read_pickle(G_cache)
    else:
        log.info(f"Generating G and saving to cache \"{G_cache.relative_to(RELATIVE_ROOT)}\"")
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        pd.to_pickle(G, G_cache)
    log.info(str(G))
    
    data.all_nodes = set(range(data.num_asser))
    data.not_masked = set(torch.where(data.y != 0)[0].tolist())

    comp_set_cache = cache_folder / "G_comp.pkl"
    if comp_set_cache.exists():
        log.info(f"Loading comp_sets from cache \"{comp_set_cache.relative_to(RELATIVE_ROOT)}\"")
        comp_sets = pd.read_pickle(comp_set_cache)
    else:
        log.info(f"Generating comp_sets and saving to cache \"{comp_set_cache.relative_to(RELATIVE_ROOT)}\"")
        comp_sets = [c for c in nx.connected_components(G)]
        comp_sets.sort(key=lambda x: len(x), reverse=True)
        pd.to_pickle(comp_sets, comp_set_cache)
    log.info(f"{len(comp_sets)} connected components found")

    return data, G, comp_sets

def generate_init_set_centrality(cfg, N_init, data, G, comp_sets):
    if N_init == 0:
        return [], sorted(data.train_asser_nodeset & data.not_masked)
    cache_folder = DATA_FOLDER / cfg.dataset.name / "cache"
    targets = [N_init * len(set(c) & data.train_asser_nodeset & data.not_masked) / len(data.train_asser_nodeset & data.not_masked) for c in comp_sets]
    new_l = [int(v) for v in targets]
    while sum(new_l) < N_init:
        residuals = [t - v for t,v in zip(targets, new_l)]
        index = residuals.index(max(residuals))
        new_l[index] += 1
    candidate_comps = [(c_idx, l) for c_idx, l in enumerate(new_l)]
    log.info(f"Selected {len(candidate_comps)}: {candidate_comps}")

    metrics = []
    for c_idx, _ in candidate_comps:
        metric_cache = cache_folder / f"metric_{cfg.init_selection.centrality}_{c_idx}.pkl"
        if metric_cache.exists():
            log.info(f"Loading metric from cache \"{metric_cache.relative_to(RELATIVE_ROOT)}\"")
            metrics.append(
                [(k, v) for k, v in pd.read_pickle(metric_cache) if k in data.train_asser_nodeset & data.not_masked]
            )
        else:
            if len(comp_sets) == 1:
                g = G
            else:
                g = G.subgraph(comp_sets[c_idx])
            log.info("Convert to igraph")
            ig_g = ig.Graph.from_networkx(g)
            log.info(f"Calculating metric for {cfg.init_selection.centrality} on component {c_idx}")
            if cfg.init_selection.centrality == 'betweenness':
                metric = sorted(zip(g.nodes(), ig_g.betweenness(directed=False)), key=lambda x: x[1], reverse=True)
            elif cfg.init_selection.centrality == 'pagerank':
                metric = sorted(zip(g.nodes(), ig_g.pagerank(directed=False)), key=lambda x: x[1], reverse=True)
            elif cfg.init_selection.centrality == 'degree':
                metric = sorted(zip(g.nodes(), ig_g.degree(loops=False)), key=lambda x: x[1], reverse=True)
            elif cfg.init_selection.centrality == 'harmonic':
                metric = sorted(zip(g.nodes(), ig_g.harmonic_centrality()), key=lambda x: x[1], reverse=True)
            else:
                raise ValueError(f"Unknown centrality {cfg.init_selection.centrality}")
            pd.to_pickle(metric, metric_cache)
            metrics.append([(k, v) for k, v in metric if k in data.train_asser_nodeset & data.not_masked])

    initial_nodes = list(chain(*[[x[0] for x in metrics[i][:candidate_comps[i][1]]] for i in range(len(candidate_comps))]))
    unlabeled_nodes = list((data.train_asser_nodeset & data.not_masked) - set(initial_nodes))
    return sorted(initial_nodes), sorted(unlabeled_nodes)

@torch.no_grad()
def ours_selectBatch(cfg, g_model, data, G, output, pool, selected, B, epoch, basef=0.95):
    g_model.eval()
    gamma = np.random.beta(1, 1.005 - basef**epoch)
    alpha = 1 - gamma
    random_gs = [
        dropout_edge(data.edge_index, p=0.2, force_undirected=True)[0] for _ in range(cfg.selection.perturb_num_e)
    ] + [
        dropout_path(data.edge_index, p=0.1, num_nodes=data.num_nodes)[0] for _ in range(cfg.selection.perturb_num_p)
    ] + [
        add_random_edge(data.edge_index, p=0.2, num_nodes=data.num_nodes)[0] for _ in range(cfg.selection.perturb_num_m)
    ]
    def my_bet(g):
        bet = ig.Graph.from_networkx(g).betweenness(vertices=data.train_asser_nodes, directed=False)
        assert len(bet) == len(data.train_asser_nodes)
        return dict(zip(data.train_asser_nodes, bet))
    centrality_method = {
        'pagerank': nx.pagerank,
        'betweenness': my_bet
    }

    centralities = []
    outs = [output]
    centrality = centrality_method[cfg.selection.centrality](G)
    arr = np.zeros(data.num_nodes)
    arr[list(centrality.keys())] = list(centrality.values())
    centralities.append(arr)

    if cfg.selection.centrality_var:
        for perturb_ei in random_gs:
            g = to_networkx(Data(x=data.x, edge_index=perturb_ei), to_undirected=True, remove_self_loops=True)
            centrality = centrality_method[cfg.selection.centrality](g)
            arr = np.zeros(data.num_nodes)
            arr[list(centrality.keys())] = list(centrality.values())
            centralities.append(arr)

    for perturb_ei in random_gs:
        outs.append(g_model(data.x, perturb_ei)[1])
    out_cls = torch.stack(outs, dim=0).detach().cpu().numpy().transpose((1, 0, 2)) # (node, trial, class_dist)

    perturb_scores = JSD(out_cls)
    centrality_var = np.var(centralities, axis=0) if cfg.selection.centrality_var else centralities[0]
    scores = alpha * perc(perturb_scores) + gamma * perc(centrality_var)

    selected_scores = scores[pool]
    ret = []
    for idx in np.argsort(-selected_scores)[:B]:
        ret.append(pool[idx])
    g_model.train()
    return ret

def generate_ours_onepass(cfg, fabric, params, N_init, N_budget, data, G, comp_sets):
    g_model = GNN_Backbone(**params)
    g_opt = torch.optim.Adam(g_model.parameters(), lr=0.01, weight_decay=1e-2)
    g_model, g_opt = fabric.setup(g_model, g_opt)
    g_model.train()

    initial_nodes, unlabeled_nodes = generate_init_set_centrality(cfg, min(N_init, N_budget // 2), data, G, comp_sets)
    N_budget_left = N_budget - min(N_init, N_budget // 2)
    
    select_every = math.ceil(cfg.init_selection.gnn_epochs * cfg.exp.N_append / N_budget_left)
    epochs = select_every * math.ceil(N_budget_left / cfg.exp.N_append)

    g_opt.zero_grad()
    node_criterion = torch.nn.CrossEntropyLoss()
    link_criterion = torch.nn.BCEWithLogitsLoss()
    with tqdm(total=epochs, ncols=100, desc='GNN Training', leave=True) as pbar:
        for epoch in range(epochs):
            real_epoch = epoch
            z, out_cls = g_model(data.x, data.edge_index)
            
            if real_epoch >= 0 and (real_epoch + 1) % select_every == 0:
                selected = ours_selectBatch(
                    cfg, g_model, data, G,
                    output=out_cls,
                    pool=unlabeled_nodes,
                    selected=initial_nodes,
                    B=cfg.exp.N_append,
                    epoch=(real_epoch + 1) // select_every - 1
                )
                for s in selected:
                    if cfg.exp.dedup:
                        if max(np.dot(data.x_dedup[s].cpu().numpy(), data.x_dedup[n].cpu().numpy()) for n in initial_nodes) > cfg.exp.dedup_threshold:
                            unlabeled_nodes.remove(s)
                            continue
                    if len(initial_nodes) < N_budget:
                        initial_nodes.append(s)
                        unlabeled_nodes.remove(s)

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
            loss = 0.1 * link_criterion(out_link, edge_label)
            loss += node_criterion(out_cls[initial_nodes], data.y[initial_nodes] - 1)
            fabric.backward(loss)
            g_opt.step()
            g_opt.zero_grad()
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
            if len(initial_nodes) >= N_budget:
                break
    return sorted(initial_nodes), sorted(unlabeled_nodes)

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    from common import test_gnn, test_infovgae

    # Configure Fabric
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).resolve()
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    fabric = L.Fabric(
        accelerator=cfg.accelerator,
        devices=cfg.device,
        loggers=CSVLogger(str(output_dir), name="logs"),
    )
    fabric.launch()

    data, G, comp_sets = setup(cfg)
    data = fabric.to_device(data)
    fabric.seed_everything(cfg.dataset.rand_set * 440383)
    gnn_params = {
        "in_channels": data.x.size(1),
        "hidden_channels": cfg.model.hidden_channels,
        "out_channels": cfg.model.out_channels,
        "layer_type": cfg.model.layer_type,
        "num_layers": cfg.model.num_layers,
        "num_class": cfg.model.num_classes,
    }
    final_selected, final_unlabeled = generate_ours_onepass(cfg, fabric, gnn_params, cfg.exp.N_init, cfg.exp.N_budget, data, G, comp_sets)
    log.info(f"Final selected {len(final_selected)}: {final_selected}")

    outresult = []
    for model_seed in range(cfg.train.seed, cfg.train.seed + 10):
        fabric.seed_everything(model_seed * 39499)
        eval_results, cls_dist, _ = test_infovgae(cfg, fabric, data, final_selected)
        eval_results_str = ', '.join([f"{k}= {v:.6f}" for k, v in eval_results.items()])
        log.info(f"Final Testing, selected {len(final_selected)}, dist {cls_dist}, {eval_results_str}")
        outresult.append(eval_results)
    print(pd.DataFrame(outresult))

    outresult = []
    for model_seed in range(cfg.train.seed, cfg.train.seed + 10):
        fabric.seed_everything(model_seed * 79319)
        eval_results, cls_dist, _ = test_gnn(cfg, fabric, data, final_selected)
        eval_results_str = ', '.join([f"{k}= {v:.6f}" for k, v in eval_results.items()])
        log.info(f"Final Testing, selected {len(final_selected)}, dist {cls_dist}, {eval_results_str}")
        outresult.append(eval_results)
    print(pd.DataFrame(outresult))

if __name__ == "__main__":
    main()
