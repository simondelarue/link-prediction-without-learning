import time
import numpy as np

from model import get_model
from metric import compute_auc, compute_ap, compute_hits_k, compute_mrr


class Trainer:
    """Trainer class."""
    def __init__(self, train_g, val_g, test_g):
        self.train_g = train_g
        self.val_g = val_g
        self.test_g = test_g
        self.N_RUNS = 1

    def __call__(self, model: str, dataset, **kwargs):
        return self.train_eval(model, dataset, **kwargs)

    def train_eval(self, model, dataset, **kwargs):
        """Train algorithm on several runs and compute average performance on train and test spits.
        
        Parameters
        ----------
        model: str
            Model name.
        dataset
            Custom Dataset object
            
        Returns
        -------
            Tuple of train and test average accuracies on several runs, with corresponding running time.
        """
        scores = {}
        val_auc_tot = 0
        test_auc_tot = 0
        val_ap_tot = 0
        test_ap_tot = 0
        val_mrr_tot = 0
        test_mrr_tot = 0
        elapsed_time = 0
        
        for _ in range(self.N_RUNS):
            # Get model
            alg = get_model(model, dataset, **kwargs)
            
            # Training algorithm
            start = time.time()
            out_val, out_test = alg.fit_predict(dataset, self.train_g, self.val_g, self.test_g, **kwargs)
            end = time.time()
            elapsed_time += end - start

            # Time constraint triggered: returns a triplet of OOM values
            if isinstance(out_test, int) and out_test == -1:
                scores['elapsed time'] = 'OOM'
                return scores
            
            # Scores
            if model.lower() in ['gae_lp', 'vgae_lp', 'seal_lp', 'neognn_lp']:
                pos_y = np.ones(self.val_g.pos_edge_label_index.size(1))
                neg_y = np.zeros(self.val_g.neg_edge_label_index.size(1))
                val_labels = np.hstack((pos_y, neg_y))
                print(f'Validation: {len(pos_y)}, {len(neg_y)}')

                pos_y = np.ones(self.test_g.pos_edge_label_index.size(1))
                neg_y = np.zeros(self.test_g.neg_edge_label_index.size(1))
                test_labels = np.hstack((pos_y, neg_y))
                print(f'Test: {len(pos_y)}, {len(neg_y)}')
            else:
                val_labels = self.val_g.edge_label.numpy()
                test_labels = self.test_g.edge_label.numpy()

            val_auc_tot += compute_auc(val_labels, out_val)
            test_auc_tot += compute_auc(test_labels, out_test)

            val_ap_tot += compute_ap(val_labels, out_val)
            test_ap_tot += compute_ap(test_labels, out_test)

            val_mrr_tot += compute_mrr(val_labels, out_val)
            test_mrr_tot += compute_mrr(test_labels, out_test)

            for k_hits in [20, 50, 100, 200]:
                val_hit = compute_hits_k(val_labels, out_val, k_hits)
                test_hit = compute_hits_k(test_labels, out_test, k_hits)
                scores[f'val hit@{k_hits}'] = val_hit
                scores[f'test hit@{k_hits}'] = test_hit

        avg_elapsed_time = elapsed_time / self.N_RUNS
        avg_val_auc = val_auc_tot / self.N_RUNS
        avg_test_auc = test_auc_tot / self.N_RUNS
        avg_val_ap = val_ap_tot / self.N_RUNS
        avg_test_ap = test_ap_tot / self.N_RUNS
        avg_val_mrr = val_mrr_tot / self.N_RUNS
        avg_test_mrr = test_mrr_tot / self.N_RUNS
        
        scores['elapsed time'] = avg_elapsed_time
        scores['val auc'] = avg_val_auc
        scores['test auc'] = avg_test_auc
        scores['val ap'] = avg_val_ap
        scores['test ap'] = avg_test_ap
        scores['val mrr'] = avg_val_mrr
        scores['test mrr'] = avg_test_mrr

        return scores
