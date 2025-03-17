# PerbALGraph

This repository contains the implementation of our paper, **"Perturbation-based Graph Active Learning for Semi-Supervised Belief Representation Learning."** It presents a novel active learning framework for semi-supervised belief representation learning that efficiently selects nodes to label by leveraging graph perturbations.

## Overview

**PerbALGraph** improves semi-supervised belief representation learning by:
- **Perturbing the Graph:** Generating multiple perturbed versions of the input graph (using edge dropping, noisy edge addition, and path dropping) to reveal hidden structural signals.
- **Measuring Instability:** Quantifying changes in model predictions (via Jensen-Shannon divergence) across perturbed graphs to identify nodes with uncertain predictions.
- **Evaluating Sensitivity:** Using variances in centrality metrics (like PageRank and Betweenness) to assess a node's structural importance.
- **Score Combination & De-duplication:** Integrating instability and sensitivity scores to select the most informative nodes while avoiding redundant labels.

These components work together to maximize performance under a limited labeling budget, as demonstrated in extensive experiments with models like GCN and SGVGAE.

## Key Components

- **Graph Perturbation:** Applies tailored perturbations to generate diverse graph views.
- **Active Learning Strategy:** Selects nodes based on a combined performance variance score.

## Getting Started

### Installation

Using virturlenv/conda/uv to create a Python 3.10 environment and install the required packages.

### Run the Algorithm

Simply execute the following command in the `src` folder:

```bash
python perb_al_graph.py
```

## Configuration
The `conf/config.yaml` file stores all configurable parameters. Modify this file to adjust hyperparameters, perturbation settings, and other experimental options.

# Dataset
Due to privacy issues, the dataset used in our experiments is not publicly released. A sample CSV file is included to illustrate the expected format and folder structure. We will consider releasing the dataset when policy permits. To process the data CSV, run the following in the `data` folder:

```bash
python process_csv.py --raw_csv <Path to CSV>
```
