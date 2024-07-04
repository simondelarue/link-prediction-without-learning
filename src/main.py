import argparse
from collections import defaultdict
import numpy as np
import os
from utils import save_dict, check_exists
from dataset import get_dataset
from train import Trainer


def run(dataset, k, model, **kwargs):
    """Run experiment."""
    
    outs = defaultdict()
    TEST_RATIO = 0.1

    print(f'---Test ratio={TEST_RATIO}')
    outs = defaultdict(list)

    for fold in range(k):
        print(f'---Fold n°:{fold + 1}')
        transform = dataset.link_split(fold, TEST_RATIO)
        train_data, val_data, test_data = transform(dataset.data)
        trainer = Trainer(train_data, val_data, test_data)
        scores = trainer(model, dataset, **kwargs)
        
        # Save training and test scores for averages on n runs
        for key, val in scores.items():
            outs[key].append(val)

        # OOM triggered: remaining folds are not computed
        if scores.get('elapsed time') == 'OOM':
            print('OOM triggered.')
            return outs

    print('---')
    print(f'Avg results on Test ratio={TEST_RATIO}')
    print(f"Test AUC {' ':>6}: {round(np.mean(outs['test auc']), 2)}")
    print(f"Test AP  {' ':>6}: {round(np.mean(outs['test ap']), 2)}")
    print(f"Test MRR {' ':>6}: {round(np.mean(outs['test mrr']), 3)}")
    for k_hits in [20, 50, 100, 200]:
        print(f"Test Hits@{k_hits:<5}: {round(np.mean(outs[f'test hit@{k_hits}']), 2)}")

    return outs


if __name__=='__main__':
    FORCE_RUN = True # If True, run experiment even if results already exists
    USE_CACHE = True # If True, use cached dataset if existing 
    DATAPATH = os.path.join(os.path.dirname(os.getcwd()), 'data')
    RUNPATH = os.path.join(os.path.dirname(os.getcwd()), 'runs')

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--randomstate', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    # Output filename
    filename = []
    for attr, value in vars(args).items():
        if value is not None:
            if attr not in ['dataset', 'model']:
                filename.append(attr + str(value).lower())
            else:
                filename.append(str(value).lower())
    filename = '_'.join(filename)

    # If run already exists, pass
    check_exists(RUNPATH, filename, force_run=FORCE_RUN)
    
    # Create dataset for GNN based models
    dataset = get_dataset(args.dataset,
                          args.randomstate,
                          args.k,
                          args.model,
                          False)
    
    print(f'Algorithm: {args.model}')
    print(f'Dataset: {args.dataset} (#nodes={dataset.data.x.shape[0]}, #edges={len(dataset.data.edge_index[0])})')

    # Dictionary of arguments
    kwargs = {
        'dataset': dataset, 
        'random_state': args.randomstate,
        'k': args.k,
        'model': args.model,
    }
    
    # Run experiment
    outs = run(**kwargs)

    # Save results
    global_out = {}
    global_out['meta'] = args
    global_out['results'] = outs
        
    save_dict(RUNPATH, filename, global_out)
    