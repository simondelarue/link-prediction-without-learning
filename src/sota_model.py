from itertools import chain
from base import BaseModel
import math
import numpy as np
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
import time

import torch
import torch_sparse
from torch.nn import Parameter
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d, ModuleList
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, SortAggregation
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
import torch_geometric.transforms as T
from rand_link_split import RandLinkSplit


# SEAL framework ; code implemented from both https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py
# and https://github.com/facebookresearch/SEAL_OGB
class DGCNN(torch.nn.Module):
    """DGCN model used as backbone model for SEAL."""
    def __init__(self, train_dataset, hidden_channels, num_layers, GNN=GCNConv, k=0.6,
                 mlp_channels=128):
        super().__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = int(max(10, k))

        self.convs = ModuleList()
        self.convs.append(GNN(train_dataset.num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.pool = SortAggregation(k)
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.mlp = MLP([dense_dim, mlp_channels, 1], dropout=0.5, norm=None)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = self.pool(x, batch)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        return self.mlp(x)


class SEALDataset(InMemoryDataset):
    """SEAL Dataset"""
    def __init__(self, dataset, train_g, val_g, test_g, num_hops: int = 2,
                 split: str = 'train'):
        self.data = dataset.data
        self.train_data = train_g
        self.val_data = val_g
        self.test_data = test_g
        self.num_hops = num_hops
        super().__init__(f'../tmp/{dataset.name.capitalize()}')
        index = ['train', 'val', 'test'].index(split)
        self.load(self.processed_paths[index])

    @property
    def processed_file_names(self):
        return ['SEAL_train_data.pt', 'SEAL_val_data.pt', 'SEAL_test_data.pt']

    def extract_enclosing_subgraphs(self, edge_index, edge_label_index, y):
        data_list = []
        
        print(f'edge lab index: {len(edge_label_index[0])}')

        for i, (src, dst) in tqdm(enumerate(edge_label_index.t().tolist())):
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hops, edge_index, num_nodes=None,
                directed=True, relabel_nodes=True)
            src, dst = mapping.tolist()

            # Remove target link from the subgraph.
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]

            # Calculate node labeling.
            z = self.drnl_node_labeling(sub_edge_index, src, dst,
                                        num_nodes=sub_nodes.size(0))
            
            # In process() we solely rely on graph structure features, not on
            # explicit features. Therefore we drop them to save some memory.
            #data = Data(x=self.data.x[sub_nodes], z=z,
            #            edge_index=sub_edge_index, y=y)
            data = Data(x=None, z=z,
                        edge_index=sub_edge_index, y=y)
            if i % int(len(edge_label_index[0]) / 10) == 0 and i > 0:
                print(data)
            data_list.append(data)

        return data_list

    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
        if num_nodes <= 1:
            return torch.tensor([0]).to(torch.long)
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        if src == dst:
            return torch.zeros(num_nodes).to(torch.long)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                                indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                                indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.

        self._max_z = max(int(z.max()), self._max_z)

        return z.to(torch.long)

    def process(self):
        #transform = RandomLinkSplit(num_val=0.05, num_test=0.1,
        #                            is_undirected=True, split_labels=True)
        #train_data, val_data, test_data = transform(self.data)

        self._max_z = 0

        # Collect a list of subgraphs for training, validation and testing:
        train_pos_data_list = self.extract_enclosing_subgraphs(
            self.train_data.edge_index, self.train_data.pos_edge_label_index, 1)
        train_neg_data_list = self.extract_enclosing_subgraphs(
            self.train_data.edge_index, self.train_data.neg_edge_label_index, 0)
        print('train done')

        val_pos_data_list = self.extract_enclosing_subgraphs(
            self.val_data.edge_index, self.val_data.pos_edge_label_index, 1)
        val_neg_data_list = self.extract_enclosing_subgraphs(
            self.val_data.edge_index, self.val_data.neg_edge_label_index, 0)
        print('val done')

        test_pos_data_list = self.extract_enclosing_subgraphs(
            self.test_data.edge_index, self.test_data.pos_edge_label_index, 1)
        test_neg_data_list = self.extract_enclosing_subgraphs(
            self.test_data.edge_index, self.test_data.neg_edge_label_index, 0)
        print('test done')

        # Convert node labeling to one-hot features.
        for data in chain(train_pos_data_list, train_neg_data_list,
                          val_pos_data_list, val_neg_data_list,
                          test_pos_data_list, test_neg_data_list):
            # We solely learn links from structure, dropping any node features:
            data.x = F.one_hot(data.z, self._max_z + 1).to(torch.float)

        print('Saving SEAL subgraphs...')
        train_data_list = train_pos_data_list + train_neg_data_list
        self.save(train_data_list, self.processed_paths[0])
        val_data_list = val_pos_data_list + val_neg_data_list
        self.save(val_data_list, self.processed_paths[1])
        test_data_list = test_pos_data_list + test_neg_data_list
        self.save(test_data_list, self.processed_paths[2])

    
class SEAL(BaseModel):
    """SEAL Model."""
    def __init__(self, name: str, dataset, **kwargs):
        super(SEAL, self).__init__(name)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.n_epochs = 50
        self.alg = None  # Initialized after preprocessing
        self.num_hops = 1 if dataset.name in ['cs', 'photo', 'wikivitals',
                                              'wikivitals-fr', 'wikischools',
                                              'wikivitals+', 'ogbn-arxiv'] else 2

    def preprocess(self, dataset, train_g, val_g, test_g):
        """Preprocess data using SEAL framework."""        
        self.train_dataset = SEALDataset(dataset, train_g, val_g, test_g, num_hops=self.num_hops, split='train')
        self.val_dataset = SEALDataset(dataset, train_g, val_g, test_g, num_hops=self.num_hops, split='val')
        self.test_dataset = SEALDataset(dataset, train_g, val_g, test_g, num_hops=self.num_hops, split='test')

        # Loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32)

    def train(self):
        """Training function."""
        self.alg.train()
        total_loss = 0

        print('Iterating over batches...')
        for data in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            out = self.alg(data.x, data.edge_index, data.batch)
            loss = self.criterion(out.view(-1), data.y.to(torch.float))
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs

        return total_loss / len(self.train_dataset)
    
    def test(self, loader):
        """Test function."""
        self.alg.eval()

        y_pred, y_true = [], []
        for data in loader:
            logits = self.alg(data.x, data.edge_index, data.batch)
            y_pred.append(logits.view(-1))
            y_true.append(data.y.view(-1).to(torch.float))

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        return y_pred.detach().numpy()
    
    def fit_predict(self, dataset, train_g, val_g, test_g, **kwargs):
        print('preprocessing...')
        self.preprocess(dataset, train_g, val_g, test_g)

        # Initialize model using SEAL training datasets
        self.alg = DGCNN(train_dataset=self.train_dataset, hidden_channels=16, num_layers=2, mlp_channels=32)
        self.optimizer = torch.optim.Adam(params=self.alg.parameters(), lr=0.0001)
        
        print('Training...')
        # Train model
        for epoch in range(1, self.n_epochs + 1):
            loss = self.train()
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
            # Time constraint
            if time.time() > self.timeout:
                return -1
        print('Done!')

        # Validation and Test
        out_val = self.test(self.val_loader)
        out_test = self.test(self.test_loader)

        return out_val, out_test
    

# From https://github.com/seongjunyun/Neo-GNNs/blob/main/models.py
class NeoGNN(torch.nn.Module):
    def __init__(self, name, dataset, hidden_channels: int = 128, out_channels: int = 64, num_layers: int = 3, dropout: float = 0):
        super(NeoGNN, self).__init__()

        in_channels = dataset.data.num_features
        self.name = name
        self.dataset = dataset
        cached = False
        self.f_edge_dim = 8
        self.f_node_dim = 128
        self.g_phi_dim = 128
        self.beta = 0.1

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached))

        self.dropout = dropout
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0, 0]))

        if name not in ['ppa', 'citation2']:
            self.f_edge = torch.nn.Sequential(torch.nn.Linear(1, self.f_edge_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.f_edge_dim, 1).double())

            self.f_node = torch.nn.Sequential(torch.nn.Linear(1, self.f_node_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.f_node_dim, 1).double())

            self.g_phi = torch.nn.Sequential(torch.nn.Linear(1, self.g_phi_dim).double(),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.g_phi_dim, 1).double())
        else:
            self.f_edge = torch.nn.Sequential(torch.nn.Linear(1, self.f_edge_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.f_edge_dim, 1))

            self.f_node = torch.nn.Sequential(torch.nn.Linear(1, self.f_node_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.f_node_dim, 1))

            self.g_phi = torch.nn.Sequential(torch.nn.Linear(1, self.g_phi_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.g_phi_dim, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.constant_(self.alpha, 0)
        self.f_edge.apply(self.weight_reset)
        self.f_node.apply(self.weight_reset)
        self.g_phi.apply(self.weight_reset)

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    def forward(self, edge, data, A, predictor=None, emb=None, only_feature=False, only_structure=False, node_struct_feat=None):
        batch_size = edge.shape[-1]
        # 1. compute similarity scores of node pairs via conventionl GNNs (feature + adjacency matrix)
        adj_t = data.adj_t
        out_feat = None
        if not only_structure:
            if emb is None:
                x = data.x
            else:
                x = emb
            for conv in self.convs[:-1]:
                x = conv(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj_t)
            if predictor is not None:
                out_feat = predictor(x[edge[0]], x[edge[1]])
            else:
                out_feat = torch.sum(x[edge[0]] * x[edge[1]], dim=0)
        
        if only_feature:
            return None, None, out_feat
        # 2. compute similarity scores of node pairs via Neo-GNNs
        # 2-1. Structural feature generation
        if node_struct_feat is None:
            row_A, col_A = A.nonzero()
            tmp_A = torch.stack([torch.from_numpy(row_A), torch.from_numpy(col_A)]).type(torch.LongTensor).to(edge.device)
            row_A, col_A = tmp_A[0], tmp_A[1]
            edge_weight_A = torch.from_numpy(A.data).to(edge.device)
            edge_weight_A = self.f_edge(edge_weight_A.unsqueeze(-1))
            node_struct_feat = scatter_add(edge_weight_A, col_A, dim=0, dim_size=data.num_nodes)

        indexes_src = edge[0].cpu().numpy()
        row_src, col_src = A[indexes_src].nonzero()
        edge_index_src = torch.stack([torch.from_numpy(row_src), torch.from_numpy(col_src)]).type(torch.LongTensor).to(edge.device)
        edge_weight_src = torch.from_numpy(A[indexes_src].data).to(edge.device)
        edge_weight_src = edge_weight_src * self.f_node(node_struct_feat[col_src]).squeeze()

        indexes_dst = edge[1].cpu().numpy()
        row_dst, col_dst = A[indexes_dst].nonzero()
        edge_index_dst = torch.stack([torch.from_numpy(row_dst), torch.from_numpy(col_dst)]).type(torch.LongTensor).to(edge.device)
        edge_weight_dst = torch.from_numpy(A[indexes_dst].data).to(edge.device)
        edge_weight_dst = edge_weight_dst * self.f_node(node_struct_feat[col_dst]).squeeze()
        
        if self.name in ['ppa', 'citation2']:
            edge_index_dst = torch.stack([edge_index_dst[1], edge_index_dst[0]])
            edge_indexes, scores = spspmm(edge_index_src, edge_weight_src, edge_index_dst, edge_weight_dst, batch_size, data.num_nodes, batch_size, data_split=256)
            out_struct = torch.zeros(batch_size).to(edge.device)
            out_struct[edge_indexes[0][edge_indexes[0]==edge_indexes[1]]] = scores[edge_indexes[0]==edge_indexes[1]]
        else:
            mat_src = SparseTensor.from_edge_index(edge_index_src, edge_weight_src, [batch_size, data.num_nodes])
            mat_dst = SparseTensor.from_edge_index(edge_index_dst, edge_weight_dst, [batch_size, data.num_nodes])
            out_struct = (mat_src @ mat_dst.to_dense().t()).diag()
        
        out_struct = self.g_phi(out_struct.unsqueeze(-1))
        out_struct_raw = out_struct
        out_struct = torch.sigmoid(out_struct)

        if not only_structure:
            alpha = torch.softmax(self.alpha, dim=0)
            out = alpha[0] * out_struct + alpha[1] * out_feat + 1e-15
        else:
            out = None

        del edge_weight_src, edge_weight_dst, node_struct_feat
        torch.cuda.empty_cache()

        return out, out_struct, out_feat, out_struct_raw

    def forward_feature(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x
    

class NeoGNNLP(BaseModel):
    """NeoGNN Model."""
    def __init__(self, name: str, dataset, **kwargs):
        super(NeoGNNLP, self).__init__(name)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.n_epochs = 200
        self.alg = NeoGNN(name, dataset, hidden_channels=128, out_channels=64, num_layers=2)
        self.predictor = LinkPredictor(64, 32, 1, 3, 0)
        self.optimizer = torch.optim.Adam(params=list(self.alg.parameters()) + list(self.predictor.parameters()), lr=0.001)
        self.beta = 0.1

        # Need Sparse formats
        self.sparse_data = T.ToSparseTensor()(dataset.data)
        self.sparse_data.full_adj_t = self.sparse_data.adj_t
        self.sparse_data.adj_t = SparseTensor.from_edge_index(dataset.data.edge_index).t()
        self.sparse_data.adj_t = self.sparse_data.adj_t.to_symmetric()

        edge_weight = torch.ones(dataset.data.edge_index.size(1), dtype=float)
        self.A = sparse.csr_matrix((edge_weight, (dataset.data.edge_index[0], dataset.data.edge_index[1])), 
                        shape=(self.sparse_data.num_nodes, self.sparse_data.num_nodes))
        A2 = self.A * self.A
        self.A = self.A + self.beta * A2
        degree = torch.from_numpy(self.A.sum(axis=0)).squeeze()

    def train(self, train_g):
        """Training function."""
        self.alg.train()
        self.predictor.train()
        self.optimizer.zero_grad()

        pos_out, pos_out_struct, pos_out_feat, _ = self.alg(train_g.pos_edge_label_index.t(), self.sparse_data, self.A, self.predictor)
        
        neg_out, neg_out_struct, neg_out_feat, _ = self.alg(train_g.neg_edge_label_index.t(), self.sparse_data, self.A, self.predictor)

        pos_loss = -torch.log(pos_out_struct + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
        loss1 = pos_loss + neg_loss

        pos_loss = -torch.log(pos_out_feat + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out_feat + 1e-15).mean()
        loss2 = pos_loss + neg_loss

        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss3 = pos_loss + neg_loss

        loss = loss1 + loss2 + loss3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.alg.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        return loss
    
    def test(self, g):
        """Test function."""
        self.alg.eval()
        self.predictor.eval()

        h = self.alg.forward_feature(self.sparse_data.x, self.sparse_data.adj_t)

        pos_edges = g.pos_edge_label_index
        neg_edges = g.neg_edge_label_index

        edge_weight = torch.from_numpy(self.A.data)
        edge_weight = self.alg.f_edge(edge_weight.unsqueeze(-1))
        row, col = self.A.nonzero()

        edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=self.sparse_data.num_nodes)
        deg =  self.alg.f_node(deg).squeeze()

        deg = deg.detach().numpy()
        A_ = self.A.multiply(deg).tocsr()

        alpha = torch.softmax(self.alg.alpha, dim=0)

        # Positives
        edge = pos_edges#.t()
        gnn_scores = self.predictor(h[edge[0]], h[edge[1]]).squeeze()
        src, dst = pos_edges
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1))
        cur_scores = torch.sigmoid(self.alg.g_phi(cur_scores).squeeze())
        cur_scores = alpha[0] * cur_scores + alpha[1] * gnn_scores
        pos_pred = cur_scores

        # Negatives
        edge = neg_edges#.t()
        gnn_scores = self.predictor(h[edge[0]], h[edge[1]]).squeeze()
        src, dst = neg_edges
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1))
        cur_scores = torch.sigmoid(self.alg.g_phi(cur_scores).squeeze())
        cur_scores = alpha[0] * cur_scores + alpha[1] * gnn_scores
        neg_pred = cur_scores

        y_pred = torch.cat((pos_pred, neg_pred), dim=0)
        
        return y_pred.detach().numpy()
    
    def fit_predict(self, dataset, train_g, val_g, test_g, **kwargs):
        
        print('Training...')
        # Train model
        for epoch in range(1, self.n_epochs + 1):
            loss = self.train(train_g)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
            # Time constraint
            if time.time() > self.timeout:
                return -1
        print('Done!')

        # Validation and Test
        out_val = self.test(val_g)
        out_test = self.test(test_g)

        return out_val, out_test


def spspmm(indexA, valueA, indexB, valueB, m, k, n, data_split=1):
    """Matrix product of two sparse tensors. Both input sparse matrices need to
    be coalesced (use the :obj:`coalesced` attribute to force).

    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of first corresponding dense matrix.
        k (int): The second dimension of first corresponding dense matrix and
            first dimension of second corresponding dense matrix.
        n (int): The second dimension of second corresponding dense matrix.
        coalesced (bool, optional): If set to :obj:`True`, will coalesce both
            input sparse matrices. (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    with torch.no_grad():
        rowA, colA = indexA
        rowB, colB = indexB
        inc = int(k//data_split) + 1
        indsA, indsB = compare_all_elements(colA, rowB, k, data_split=data_split)
        prod_inds = torch.cat((rowA[indsA].unsqueeze(0), colB[indsB].unsqueeze(0)), dim=0)
    prod_vals = valueA[indsA]*valueB[indsB]
    return torch_sparse.coalesce(prod_inds, prod_vals, m, n)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

        self.alpha = Parameter(torch.Tensor(1))
        self.theta = Parameter(torch.Tensor(1))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        
        torch.nn.init.constant_(self.alpha, 0.5)
        torch.nn.init.constant_(self.theta, 2)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
def compare_all_elements(tensorA, tensorB, max_val, data_split=1):
    """
    Description.....
    
    Parameters:
        tensorA:         first array to be compared (1D torch.tensor of ints)
        tensorB:         second array to be compared (1D torch.tensor of ints)
        max_val:         the largest element in either tensorA or tensorB (real number)
        data_split:      the number of subsets to split the mask up into (int)
    Returns:
        compared_indsA:  indices of tensorA that match elements in tensorB (1D torch.tensor of ints, type torch.long)
        compared_indsB:  indices of tensorB that match elements in tensorA (1D torch.tensor of ints, type torch.long)
    """
    compared_indsA, compared_indsB, inc = torch.tensor([]).to(tensorA.device), torch.tensor([]).to(tensorA.device), int(max_val//data_split) + 1
    for iii in range(data_split):
        indsA, indsB = (iii*inc<=tensorA)*(tensorA<(iii+1)*inc), (iii*inc<=tensorB)*(tensorB<(iii+1)*inc)
        tileA, tileB = tensorA[indsA], tensorB[indsB]
        tileA, tileB = tileA.unsqueeze(0).repeat(tileB.size(0), 1), torch.transpose(tileB.unsqueeze(0), 0, 1).repeat(1, tileA.size(0))
        nz_inds = torch.nonzero(tileA == tileB, as_tuple=False)
        nz_indsA, nz_indsB = nz_inds[:, 1], nz_inds[:, 0]
        compared_indsA, compared_indsB = torch.cat((compared_indsA, indsA.nonzero()[nz_indsA]), 0), torch.cat((compared_indsB, indsB.nonzero()[nz_indsB]), 0)
    return compared_indsA.squeeze().long(), compared_indsB.squeeze().long()
