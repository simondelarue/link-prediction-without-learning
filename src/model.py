import numpy as np
from scipy import sparse
from scipy.spatial.distance import jensenshannon
import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.neighbors import KernelDensity

import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GAE, VGAE
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import torch.nn.functional as F

from base import BaseModel
from sota_model import SEAL, NeoGNNLP

"""
Models developed in PyTorchGeometric library are available here.
For state-of-the-art algorithms not avaiblable in PyTorchGeometric, see sota_models.py file.
"""


def get_model(model: str, dataset = None, train_idx : np.ndarray = None, **kwargs) -> BaseModel:
    """Get model."""
    # models developped in torch_geometric for link prediction
    if model.lower() in ['gcn_lp', 'gat_lp', 'graphsage_lp']:
        return GNNLP(model.lower(), dataset, **kwargs)
    # GAE and VAGE for link prediction
    elif model.lower() in ['gae_lp', 'vgae_lp']:
        return GAELP(model.lower(), dataset, **kwargs)
    # Topological heuristics
    elif model.lower() in ['cn_lp', 'ecn_lp',
                           'aa_lp', 'eaa_lp',
                           'ra_lp', 'era_lp']:
        return Heuristics(model.lower(), dataset, **kwargs)
    elif model.lower() in ['seal_lp']:
        return SEAL(model.lower(), dataset, **kwargs)
    elif model.lower() in ['neognn_lp']:
        return NeoGNNLP(model.lower(), dataset, **kwargs)
    else:
        raise ValueError(f'Unknown model: {model}.')


class GCNEncoder(torch.nn.Module):
    """GCN Encoder."""
    def __init__(self, dataset, out_channels: int = 128):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
    

class VariationalGCNEncoder(torch.nn.Module):
    """Variational GCN Encoder."""
    def __init__(self, dataset, out_channels: int = 128):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    

class GCNLP(torch.nn.Module):
    """GCN Model for link prediction."""
    def __init__(self, dataset, hidden_channels: int = 32, out_channels: int = 16):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]] # embeddings of all src nodes
        dst = z[edge_label_index[1]] # embeddings of all dst nodes
        r = (src * dst).sum(dim=-1) # dot product as in original GAE
        return r
    
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    

class GraphSageLP(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 256, out_channels: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=dataset.num_features, out_channels=hidden_channels, aggr='max')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='max')

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]] # embeddings of all src nodes
        dst = z[edge_label_index[1]] # embeddings of all dst nodes
        r = (src * dst).sum(dim=-1) # dot product as in original GAE
        return r


class GATLP(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 16, out_channels: int = 64, n_heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=n_heads)
        self.conv2 = GATConv(hidden_channels * n_heads, out_channels)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]] # embeddings of all src nodes
        dst = z[edge_label_index[1]] # embeddings of all dst nodes
        r = (src * dst).sum(dim=-1) # dot product as in original GAE
        return r
    

class AAPredictor():
    """Adamic Adar Index predictor (or Ressource Allocation if use_log is set to False)."""
    def __init__(self, dataset, use_log: bool = True, undirected: bool = False, custom: bool = False) -> None:
        self.dataset = dataset
        self.undirected = undirected
        self.use_log = use_log
        self.custom = custom
        self.cos_sim = None

    def __call__(self, g) -> np.ndarray:
        return self.encode(g)
    
    def feature_similarity(self, features, edge_index):
        cos_sim = cosine_similarity(features)
        res = []
        for src, dst in zip(*edge_index):
            feat_sim_e = cos_sim[src, dst]
            res.append(feat_sim_e)
        return res
    
    def pos_neg_similarity(self, g):
        """Compute feature similarity on positive and negative samples."""
        pos_mask = g.edge_label == 1
        neg_mask = g.edge_label == 0
        pos_edge_label_index = g.edge_label_index[:, pos_mask]
        neg_edge_label_index = g.edge_label_index[:, neg_mask]

        pos_cos_sim = self.feature_similarity(g.x, pos_edge_label_index)
        neg_cos_sim = self.feature_similarity(g.x, neg_edge_label_index)

        return pos_cos_sim, neg_cos_sim
    
    def estimate_probs(self, values):
        model = KernelDensity(bandwidth=.01)
        sample = np.array(values).reshape((len(values), 1))
        model.fit(sample)

        x = np.asarray([val for val in np.arange(0, 1, 0.01)])
        x = x.reshape((len(x), 1))
        probs = model.score_samples(x)
        probs = np.exp(probs)

        return probs
    
    def encode(self, g) -> np.ndarray:
        """Retrieve neighbors and degree for each node. If 'custom' parameter
        is set to True, use feature information."""
        adj = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.x.shape[0]).tocsr()

        if self.custom:
            pos_sim, neg_sim = self.pos_neg_similarity(g)
            probs_pos = self.estimate_probs(pos_sim)
            probs_neg = self.estimate_probs(neg_sim)
            self.js_distance = jensenshannon(probs_pos, probs_neg)

        if not self.undirected:
            if self.custom:
                # Augment adjacency with feature similarity
                self.cos_sim = sparse.csr_matrix(cosine_similarity(g.x))

            # Find neighbors in (augmented) data
            adj_lil = adj.tolil()
            neighbors_out = {idx: set(neighbs) for idx, neighbs in enumerate(adj_lil.rows)}
            neighbors_in = {idx: set(neighbs) for idx, neighbs in enumerate(adj_lil.T.rows)}

            # In and out degrees
            self.out_degs = adj.astype(bool).dot(np.ones(adj.shape[1]))
            self.in_degs = adj.astype(bool).T.dot(np.ones(adj.shape[0]))
            
            return neighbors_in, neighbors_out
    
    def decode(self, g, neighbors_in, neighbors_out):
        """Compute Adamic Adar index if 'use_log' is set to True, else Ressource Allocation.
        If 'custom' parameter is set to True, feature information is used."""
        # Cosine similarity for all pairs of nodes
        if self.custom:
            cos_sim = np.asarray(self.cos_sim.todense())

        indexes = []
        feat_sims = []
        cc_sims = []
        for e_idx, (src, dst) in enumerate(zip(*g.edge_label_index)):
            # Common neighbors defined as intersection between out-neighbors of src node and in-neighbors of dst node
            common_neighbors = np.asarray(list(neighbors_out.get(src.item()).intersection(neighbors_in.get(dst.item()))))
            index = 0
            feat_idx = 0
            feat_e = 0
            
            if len(common_neighbors) > 0:
                # Divide by the degree of the common neighbor
                assert len(self.out_degs[common_neighbors]) == len(self.in_degs[common_neighbors])
                len_neighbs_cc = self.out_degs[common_neighbors] + self.in_degs[common_neighbors]
                # If use_log -> Adamic Adar, otherwise, Ressource Allocation
                if self.use_log:
                    index = np.sum(1 / np.log(len_neighbs_cc))
                else:
                    index = np.sum(1 / len_neighbs_cc)
            indexes.append(index)

            if self.custom:
                if len(common_neighbors) > 0:
                    # src -> common_neighb and common_neighb -> dst feature similarities
                    feat_sim_src_cc = cos_sim[src.item(), common_neighbors]
                    feat_sim_dst_cc = cos_sim[common_neighbors, dst.item()]
                    feat_idx = feat_sim_src_cc.dot(feat_sim_dst_cc.T)
                cc_sims.append(feat_idx)

                # src -> dst feature similarity
                feat_e = cos_sim[src, dst]
                feat_sims.append(feat_e)

        if self.custom:
            neighbors_weight = normalize(np.asarray(indexes).reshape(1, -1))[0]
            feature_weight = normalize(np.asarray(feat_sims).reshape(1, -1))[0] \
                + normalize(np.asarray(cc_sims).reshape(1, -1))[0]
            return (1 - self.js_distance) * neighbors_weight + self.js_distance * feature_weight
        else:
            return np.asarray(indexes)
    

class CNPredictor():
    """Common Neighbors predictor."""
    def __init__(self, dataset, undirected: bool = False, custom: bool = False) -> None:
        self.dataset = dataset
        self.undirected = undirected
        self.custom = custom
        self.cos_sim = None

    def __call__(self, g) -> np.ndarray:
        return self.encode(g)
    
    def feature_similarity(self, features, edge_index):
        cos_sim = cosine_similarity(features)
        res = []
        for src, dst in zip(*edge_index):
            feat_sim_e = cos_sim[src, dst]
            res.append(feat_sim_e)
        return res
    
    def pos_neg_similarity(self, g):
        """Compute feature similarity on positive and negative samples."""
        pos_mask = g.edge_label == 1
        neg_mask = g.edge_label == 0
        pos_edge_label_index = g.edge_label_index[:, pos_mask]
        neg_edge_label_index = g.edge_label_index[:, neg_mask]

        pos_cos_sim = self.feature_similarity(g.x, pos_edge_label_index)
        neg_cos_sim = self.feature_similarity(g.x, neg_edge_label_index)

        return pos_cos_sim, neg_cos_sim
    
    def estimate_probs(self, values):
        model = KernelDensity(bandwidth=.01)
        sample = np.array(values).reshape((len(values), 1))
        model.fit(sample)

        x = np.asarray([val for val in np.arange(0, 1, 0.01)])
        x = x.reshape((len(x), 1))
        probs = model.score_samples(x)
        probs = np.exp(probs)

        return probs
    
    def encode(self, g) -> np.ndarray:
        """Retrieve neighbors of each node. If 'custom' parameter is set to True,
        use feature information."""
        adj = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.x.shape[0]).tocsr()

        if self.custom:
            pos_sim, neg_sim = self.pos_neg_similarity(g)
            probs_pos = self.estimate_probs(pos_sim)
            probs_neg = self.estimate_probs(neg_sim)
            self.js_distance = jensenshannon(probs_pos, probs_neg)

        if not self.undirected:
            if self.custom:
                # Augment adjacency with feature similarity
                self.cos_sim = sparse.csr_matrix(cosine_similarity(g.x))

            # Find neighbors in (augmented) data
            adj_lil = adj.tolil()
            neighbors_out = {idx: set(neighbs) for idx, neighbs in enumerate(adj_lil.rows)}
            neighbors_in = {idx: set(neighbs) for idx, neighbs in enumerate(adj_lil.T.rows)}
            
            return neighbors_in, neighbors_out
        else:
            pass

    def decode(self, g, neighbors_in, neighbors_out):
        """Compute length of common neighbors. If 'custom' parameter is set to True,
        feature information is used."""
        len_common_neighbs = []
        feat_sim = []
        for src, dst in zip(*g.edge_label_index):
            if not self.undirected:
                common_neighbors = neighbors_out.get(src.item()).intersection(neighbors_in.get(dst.item()))
                len_common_neighbs.append(len(common_neighbors))
                if self.custom:
                    # use feature information
                    feat_e = self.cos_sim[src, dst]
                    feat_sim.append(feat_e)
            else:
                pass
        if self.custom:
            neighbors_weight = normalize(np.asarray(len_common_neighbs).reshape(1, -1))[0]
            feature_weight = normalize(np.asarray(feat_sim).reshape(1, -1))[0]
            return (1 - self.js_distance) * neighbors_weight + self.js_distance * feature_weight
        else:
            return np.asarray(len_common_neighbs)


class Heuristics(BaseModel):
    """Topological heuristics for link prediction."""
    def __init__(self, name: str, dataset, **kwargs):
        super(Heuristics, self).__init__(name)
        if name.lower() == 'cn_lp':
            self.alg = CNPredictor(dataset.data)
        elif name.lower() == 'ecn_lp':
            self.alg = CNPredictor(dataset.data, custom=True)
        elif name.lower() == 'aa_lp':
            self.alg = AAPredictor(dataset.data, use_log=True)
        elif name.lower() == 'eaa_lp':
            self.alg = AAPredictor(dataset.data, use_log=True, custom=True)
        elif name.lower() == 'ra_lp':
            self.alg = AAPredictor(dataset.data, use_log=False)
        elif name.lower() == 'era_lp':
            self.alg = AAPredictor(dataset.data, use_log=False, custom=True)
        else:
            raise Exception('Unkown Heuristics.')

    def fit_predict(self, dataset, train_g, val_g, test_g, **kwargs) -> np.ndarray:
        """Predict validation and test edges."""
        # Common neighbors and degree information from training data
        neighbors_in, neighbors_out = self.alg(train_g)
        
        # Val and test predictions
        out_val = self.alg.decode(val_g, neighbors_in, neighbors_out)
        out_test = self.alg.decode(test_g, neighbors_in, neighbors_out)
        
        # Time constraint
        if time.time() > self.timeout:
            return -1, -1
        
        return out_val, out_test


class GAELP(BaseModel):
    """GAE and VAGE models for link prediction."""
    def __init__(self, name: str, dataset, **kwargs):
        super(GAELP, self).__init__(name)
        self.n_epochs = 200
        if name.lower() == 'gae_lp':
            self.alg = GAE(GCNEncoder(dataset.data))
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=0)
        elif name.lower() == 'vgae_lp':
            self.alg = VGAE(VariationalGCNEncoder(dataset.data))
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=0)

    def train(self, train_g):
        """Training function.
        
        Parameters
        ----------
        train_g: Custom Dataset object for training.
        
        Returns
        -------
            Loss. 
        """
        self.alg.train()
        self.optimizer.zero_grad()
        z = self.alg.encode(train_g.x, train_g.edge_index)
        # In the following line, if negative edge indexes are not given, negative sampling is performed at each call (i.e. at each epoch), which drastically improves the performance
        # Loss (recon_loss()) for GAE is the binary cross entropy (see torch_geometric doc).
        loss = self.alg.recon_loss(z, train_g.pos_edge_label_index, None)
        if self.name == 'vgae_lp':
            loss = loss + (1 / train_g.num_nodes) * self.alg.kl_loss()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def test(self, data):
        """Test function.
        
        Parameters
        ----------
        data: Torch Data object.
        
        Returns
        -------
            Loss.
        """
        self.alg.eval()

        z = self.alg.encode(data.x, data.edge_index)
        
        pos_y = z.new_ones(data.pos_edge_label_index.size(1))
        neg_y = z.new_zeros(data.neg_edge_label_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.alg.decoder(z, data.pos_edge_label_index, sigmoid=True)
        neg_pred = self.alg.decoder(z, data.neg_edge_label_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        return pred
    
    def fit_predict(self, dataset, train_g, val_g, test_g, **kwargs) -> np.ndarray:
        """Fit algorithm on training data and predict node labels.
        
        Parameters
        ----------
        dataset
            Custom Dataset object.
            
        Returns
        -------
            Array of predicted node labels.
        """
        # Train model
        for epoch in range(1, self.n_epochs + 1):
            loss = self.train(train_g)
            out_val = self.test(val_g)
            out_test = self.test(test_g)

            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            
            # Time constraint
            if time.time() > self.timeout:
                return -1, -1

        return out_val, out_test


class GNNLP(BaseModel):
    """GNN model for link prediction class."""
    def __init__(self, name: str, dataset, **kwargs):
        super(GNNLP, self).__init__(name)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # Transform features if needed
        #dataset = self.transform_data(dataset, **kwargs)

         # Initialize model
        if name == "gcn_lp":
            self.alg = GCNLP(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.n_epochs = 200
        elif name == 'gat_lp':
            self.alg = GATLP(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.05, weight_decay=5e-4)
            self.n_epochs = 100
        elif name == 'graphsage_lp':
            self.alg = GraphSageLP(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.n_epochs = 100

    def train(self, train_g) -> torch.Tensor:
        """Training function.
        
        Parameters
        ----------
        train_g: Custom Dataset object for training.
        
        Returns
        -------
            Loss. 
        """
        self.alg.train()

        self.optimizer.zero_grad()
        z = self.alg.encode(train_g.x, train_g.edge_index)
        out = self.alg.decode(z, train_g.edge_label_index).view(-1)
        loss = self.criterion(out, train_g.edge_label)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def test(self, data):
        """Test function.
        
        Parameters
        ----------
        data: Torch Data object.
        
        Returns
        -------
            Loss.
        """
        self.alg.eval()
        z = self.alg.encode(data.x, data.edge_index)
        out = self.alg.decode(z, data.edge_label_index).view(-1).sigmoid()
        return out.detach().numpy()
    
    def fit_predict(self, dataset, train_g, val_g, test_g, **kwargs) -> np.ndarray:
        """Fit algorithm on training data and predict node labels.
        
        Parameters
        ----------
        dataset
            Custom Dataset object.
            
        Returns
        -------
            Array of predicted node labels.
        """
        # Train model
        for epoch in range(1, self.n_epochs + 1):
            loss = self.train(train_g)
            out_val = self.test(val_g)
            out_test = self.test(test_g)

            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            
            # Time constraint
            if time.time() > self.timeout:
                return -1, -1

        return out_val, out_test
