import argparse
from collections import Counter
from copy import deepcopy
from glob import glob
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import (
    add_remaining_self_loops,
    remove_isolated_nodes,
    subgraph,
    to_undirected,
)


def encode(docs, tokenizer, max_len):
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, 
                                               max_length=max_len, padding='max_length',
                                               return_attention_mask=True, truncation=True, 
                                               return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

def create_dataset(docs, tokenizer, max_len=512, num_cpus=16):
    print("Converting texts into tensors.")
    chunk_size = ceil(len(docs) / num_cpus)
    chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
    results = Parallel(n_jobs=num_cpus)(delayed(encode)(docs=chunk, tokenizer=tokenizer, max_len=max_len) for chunk in chunks)
    input_ids = torch.cat([result[0] for result in results])
    attention_masks = torch.cat([result[1] for result in results])
    data = TensorDataset(input_ids, attention_masks)
    return data

def subgraph_bipartite(subset, edge_index, num_nodes):
    sub_edge_index, _, sub_node_mask = remove_isolated_nodes(
        subgraph(subset, edge_index, num_nodes=num_nodes)[0],
        num_nodes=num_nodes
    )
    return sub_edge_index, sub_node_mask

def main(input_path):
    data = pd.read_csv(input_path)
    # data.drop(columns=['Unnamed: 0'], inplace=True)
    data['label']=data.apply(lambda x: x['manual_label'] if not pd.isna(x['manual_label']) else x['gpt_label'], axis=1)

    label_set = set(data.label.unique())
    if 'unused' in label_set:
        label_set.remove('unused')
    label_set = sorted(label_set)
    label_map = {'unused': 0} | {k: i + 1 for i, k in enumerate(label_set)}
    msg2label = data[['message_id', 'label']].set_index('message_id').label.map(label_map).to_dict()
    msg2content = data[['message_id', 'text']].set_index('message_id').text.to_dict()
    msg2ismanual = data[['message_id', 'is_gt']].set_index('message_id').is_gt.to_dict()
    msg2istest = data[['message_id', 'label', 'is_gt']].set_index('message_id').apply(lambda x: x.label != 'unused' and x.is_gt == 1, axis=1).to_dict()
    act2label = data.groupby('actor_id').label.apply(lambda x: label_map[sorted(Counter(x).most_common(), key=lambda x: (x[1],x[0]))[0][0]]).to_dict()

    assertions = data.groupby('index_text').message_id.apply(set).to_dict() # {assertion: {message_id}}
    act2id = {k: i for i, k in enumerate(data.actor_id.unique())} # {actor: id}
    id2act = {i: k for k, i in act2id.items()} # {id: actor}
    asser2id = {k: i + len(act2id) for i, k in enumerate(data.index_text.unique())} # {assertion: id}
    id2asser = {i: k for k, i in asser2id.items()} # {id: assertion}

    x_onehot = torch.FloatTensor(np.diag(np.ones(len(asser2id) + len(act2id))))
    edges = set()
    for _, r in data.iterrows():
        edges.add((asser2id[r['index_text']], act2id[r['actor_id']]))
    edges = torch.LongTensor(sorted(edges)).T
    y = [act2label[id2act[i]] for i in range(len(id2act))] + [msg2label[next(iter(assertions[id2asser[i + len(act2id)]]))] for i in range(len(id2asser))]
    is_manual = torch.BoolTensor([0 for _ in range(len(id2act))] + [msg2ismanual[next(iter(assertions[id2asser[i + len(act2id)]]))] for i in range(len(id2asser))])
    is_asser = torch.BoolTensor([False for _ in range(len(id2act))] + [True for _ in range(len(id2asser))])
    is_test = torch.BoolTensor([False for _ in range(len(id2act))] + [msg2istest[next(iter(assertions[id2asser[i + len(act2id)]]))] for i in range(len(id2asser))])
    is_train = torch.BoolTensor([False for _ in range(len(id2act))] + [not msg2istest[next(iter(assertions[id2asser[i + len(act2id)]]))] for i in range(len(id2asser))])
    pool = torch.where(is_train)[0].tolist()
    actor_pool = list(range(len(act2id)))
    assert set(pool) & set(torch.where(is_test)[0].tolist()) == set()
    
    gdata = Data(
        x = x_onehot,
        x_dedup = x_onehot,
        edge_index = to_undirected(add_remaining_self_loops(edges)[0]),
        orig_edge_index = edges,
        y = torch.LongTensor(y),
        y_is_manual = is_manual,
        y_is_asser = is_asser,
        test_asser_nodes = torch.where(is_test)[0].tolist(),
        test_asser_nodeset = set(torch.where(is_test)[0].tolist()),
        test_asser_mask = is_test,
        train_asser_nodes = pool,
        train_asser_nodeset = set(pool),
        train_asser_mask = is_train,
        orig_id = [id2act[i] for i in range(len(id2act))] + [id2asser[i + len(act2id)] for i in range(len(id2asser))],
        orig_msgid = [None for _ in range(len(id2act))] + [assertions[id2asser[i + len(act2id)]] for i in range(len(id2asser))],
        orig_msg = [None for _ in range(len(id2act))] + [[msg2content[m] for m in assertions[id2asser[i + len(act2id)]]] for i in range(len(id2asser))],
        num_asser = len(asser2id),
        num_actor = len(act2id),
    )
    
    test_edge_index, test_nodes_mask = subgraph_bipartite(actor_pool + gdata.test_asser_nodes, gdata.edge_index, gdata.num_nodes)
    gdata.test_edge_index = test_edge_index
    gdata.test_nodes_mask = test_nodes_mask
    train_edge_index, train_nodes_mask = subgraph_bipartite(actor_pool + gdata.train_asser_nodes, gdata.edge_index, gdata.num_nodes)
    gdata.train_edge_index = train_edge_index
    gdata.train_nodes_mask = train_nodes_mask

    sp_pt = int(len(pool) * 0.1)
    gdata['val_sppt'] = sp_pt
    for i in range(100):
        np.random.seed(i)
        pool_t = deepcopy(pool)
        np.random.shuffle(pool_t)
        gdata[f'train_randset_{i}'] = pool_t
    (input_path.parent.parent / 'processed').mkdir(parents=True, exist_ok=True)
    torch.save(gdata, input_path.parent.parent / 'processed' / 'data.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_csv', type=str, required=True, help='specify input raw data csv file')
    args = parser.parse_args()
    main(Path(args.raw_csv))
