# PerbALGraph

This repository contains the implementation of our paper, **"Perturbation-based Graph Active Learning for Semi-Supervised Belief Representation Learning."** It presents a novel active learning framework for graph-structured data that efficiently selects nodes to label by leveraging graph perturbations.

## Overview

**PerbALGraph** improves semi-supervised belief representation learning by:
- **Perturbing the Graph:** Generating multiple perturbed versions of the input graph (using edge dropping, noisy edge addition, and path dropping) to reveal hidden structural signals.
- **Measuring Instability:** Quantifying changes in model predictions (via Jensen-Shannon divergence) across perturbed graphs to identify nodes with uncertain predictions.
- **Evaluating Sensitivity:** Using centrality metrics (like PageRank and Betweenness) to assess a node’s structural importance.
- **Score Combination & De-duplication:** Integrating instability and sensitivity scores to select the most informative nodes while avoiding redundant labels.

These components work together to maximize performance under a limited labeling budget, as demonstrated in extensive experiments with models like GCN and SGVGAE.

## Key Components

- **Graph Perturbation:** Applies tailored perturbations to generate diverse graph views.
- **Active Learning Strategy:** Selects nodes based on a combined performance variance score.
- **De-duplication Module:** Prevents redundant labeling by ensuring diversity among selected nodes.
- **Experimental Validation:** Evaluated on multiple social network datasets (datasets are withheld due to privacy concerns; a sample CSV is provided).

## Getting Started

### Installation

Using virturlenv/conda/uv to create a Python 3.10 environment and install the required packages.

### Running the Model

Simply execute the following command to run the active learning process:

```bash
python perb_al_graph.py
```

## Configuration
All configurable parameters are stored in the conf/config.yaml file. Modify this file to adjust hyperparameters, perturbation settings, and other experimental options.

# Dataset
Due to privacy issues, the dataset used in our experiments is not publicly released. A sample CSV file is included to illustrate the expected format. We will consider releasing the dataset when policy permits.