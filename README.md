# Link prediction without learning

This repository provides the implementation and supplementary material of the following paper:

**Link prediction without learning**  
*Simon Delarue, Thomas Bonald, Tiphaine Viard*  
European Conference on Artificial Intelligence, 2024.

>**Abstract.** Link prediction is a fundamental task in machine learning for graphs. Recently, Graph Neural Networks (GNNs) have gained in popularity and have become the default approach for solving this type of task. Despite the considerable interest for these methods, simple topological heuristics persistently emerge as competitive alternatives to GNNs. In this study, we show that this phenomenon is not an exception and that GNNs do not consistently establish a performance standard for link prediction on graphs. For this purpose, we identify several limitations in the current GNN evaluation methodology, such as the lack of variety in benchmark dataset characteristics and the limited use of diverse baselines outside of neural methods. In particular, we highlight that integrating feature information into topological heuristics remains a little-explored path. In line with this observation, we propose a simple non-neural model that leverages local structure, node feature, and graph feature information within a weighted combination. Experiments conducted on large variety of networks indicate that the proposed approach outperforms existing state-of-the-art GNNs and increases generalisation ability. Contrasting with GNNs, our approach does not rely on any learning process and therefore achieves superior results without sacrificing efficiency, showcasing a reduction of one to three orders of magnitude in computation time.

## Requirements

Combination: `torch=2.0.0` + `torch_geometric==2.4.0` + `torch_scatter==2.1.1`

```shell
python -m pip install -r requirements.txt 
```

## Usage

Available options are the following:
```
--dataset         Graph dataset {cora, pubmed, ...}
--randomstate     Random state seed (default=8)
--k               Number of random splits (e.g. 3)
--model           Model name. Available models are {MODEL}_LP
```
Available values for `MODEL` are:  
- Enhanced topological heuristics: `ECN`, `EAA`, `ERA`.  
- Topological heuristics: `CN`, `AA`, `RA`.  
- GNNs: `GCN`, `GAT`, `GRAPHSAGE`, `GAE`, `VGAE`, `SEAL`, `NEOGNN`.

### Example

To run experiment for the **Enhanced Resource Allocation (ERA)** model on `Cora` dataset, use:
```
python src/main.py --dataset=cora --randomstate=8 --k=3 --model=ERA_LP
```

### Results

Results for all evaluation metrics are saved within `src/runs/` directory.

## Supplementary material

`supplementary_material/appendices/` contains all appendices with detailed information about datasets, models, hyperparameters, and additional experiment results.

`supplementary_material/scopus/` contains detailed information about all the reviewed literature articles and baseline counting.

## Reference

If you find the code useful, please cite our paper.

```
To be updated.
```

