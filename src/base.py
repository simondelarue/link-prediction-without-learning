from abc import ABC, abstractmethod
import numpy as np
import os
import pickle
from scipy import sparse
import time

from sknetwork.data import Bunch

from torch_geometric.data import Data
import torch_geometric.transforms as T

from rand_link_split import RandLinkSplit


class BaseDataset:
    """Base class for Dataset."""

    def __init__(self, dataset: str, random_state: int, k: int, model, undirected: bool):
        self.name = dataset
        self.random_state = random_state
        self.data = self.get_data(dataset, undirected)
        self.k = k
        self.undirected = undirected
        self.model = model.lower()
    
    def link_split(self, fold: int, test_ratio: float):
        """Transform data to get train, validation and test splits on edges.
        """
        # Negative to positive edge ratio in the whole graph
        n_possible_edges = self._get_nb_nodes() * (self._get_nb_nodes() - 1)
        n_neg_edges = n_possible_edges - self._get_nb_edges()
        ratio_neg_to_pos = round(n_neg_edges / self._get_nb_edges(), 1)       

        # Ratio of negative pairs in training is set to be the same as in the original graph
        if self.model in ['gae_lp', 'vgae_lp', 'seal_lp', 'neognn_lp']:
            transform = T.Compose([
                RandLinkSplit(num_val=0.05, num_test=test_ratio,
                            is_undirected=self.undirected,
                            split_labels=True,
                            add_negative_train_samples=True,
                            neg_sampling_ratio=1,
                            seed=fold,
                            disjoint_train_ratio=0.0)])
        else:
            transform = T.Compose([
                RandLinkSplit(num_val=0.05, num_test=test_ratio,
                            is_undirected=self.undirected,
                            add_negative_train_samples=True,
                            neg_sampling_ratio=1,
                            seed=fold,
                            disjoint_train_ratio=0.0)])
        return transform
    
    def get_netset(self, dataset: str, pathname: str, use_cache: bool = True):
        """Get data in Netset format (scipy.sparse CSR matrices). Save data in Bunch format if use_cache is set to False.
        
        Parameters
        ----------
        dataset: str
            Dataset name
        pathname: str
            Path to data.
        use_cache: bool (default=True)
            If True, use cached data (if existing).

        Returns
        -------
            Bunch object.
        """
        if os.path.exists(os.path.join(pathname, dataset)) and use_cache:
            with open(os.path.join(pathname, dataset), 'br') as f:
                graph = pickle.load(f)
            print(f'Loaded dataset from {os.path.join(pathname, dataset)}')
        else:
            print(f'Building netset data...')
            # Convert dataset to NetSet format (scipy CSR matrices)
            graph = self.to_netset(dataset)

            # Save Netset dataset
            with open(os.path.join(pathname, dataset), 'bw') as f:
                pickle.dump(graph, f)
            print(f'Netset data saved in {os.path.join(pathname, dataset)}')
        
        self.netset = graph
        
        return self.netset
    
    def to_netset(self, dataset: str):
        """Convert data into Netset format and return Bunch object."""
        # nodes and edges
        rows = np.asarray(self.data.edge_index[0])
        cols = np.asarray(self.data.edge_index[1])
        data = np.ones(len(rows))
        n = len(self.data.y)
        adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        # Features
        if dataset.startswith('ogbn'):
            biadjacency = sparse.csr_matrix(np.array(self.data.x))
        else:
            biadjacency = sparse.csr_matrix(np.array(self.data.x))

        # Node information
        labels = np.array(self.data.y)

        graph = Bunch()
        graph.adjacency = adjacency.astype(bool)
        graph.biadjacency = biadjacency
        graph.labels_true = labels

        return graph
    
    def _to_custom_data(self, dataset):
        """Convert Dataset format from Pytorch to a modifiable Data object."""
        data = Data(x=dataset.x,
               edge_index=dataset.edge_index,
               num_classes=dataset.num_classes,
               y=dataset.y,
               train_mask=dataset.train_mask,
               val_mask=dataset.val_mask,
               test_mask=dataset.test_mask)
        
        return data
    
    def _get_nb_nodes(self):
        """Get number of nodes in the graph."""
        return self.data.x.shape[0]
    
    def _get_nb_edges(self):
        """Get number of edges in the graph."""
        return len(self.data.edge_index[0])

class BaseModel(ABC):
    """Base class for models."""

    def __init__(self, name: str):
        self.name = name
        self.train_loader = None
        self.timeout = time.time() + 60*60*24 # 24hour timeout

    @abstractmethod
    def fit_predict(self, dataset, train_idx: np.ndarray = None):
        pass
