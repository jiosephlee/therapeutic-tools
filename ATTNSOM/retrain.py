"""Retrain ATTNSOM with proper train/val/test tracking.

Improvements over main.py:
- Logs train AND val metrics (F1, top-k) every epoch, not just val loss
- Early stopping with patience on val F1
- Final checkpoint trained on train+val with epoch count from best CV fold
- Saves training curves to JSON for inspection

Usage:
    cd openrlhf/tools/therapeutic_tools/ATTNSOM
    python retrain.py --max_epochs 200 --patience 30 --result_dir /vast/projects/myatskar/design-documents/hf_home/attnsom_results_v2
"""

import os
import sys
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import argparse

# Add ATTNSOM dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_multi_cyp, apply_no_leakage_to_dataloaders, load_exclusion_smiles
from dataset_utils import make_strata_labels, split_train_val_by_index_stratified_cyp
from train import train_one_epoch, evaluate, bce_pos_weight
from metrics import calc_threshold_metrics, topk_accuracy, per_atom_accuracy, molecule_exact_match
from model import GraphCliffMultiRegressor

CYP_LIST = ['1A2', '2A6', '2B6', '2C8', '2C9', '2C19', '2D6', '2E1', '3A4']
THRESHOLD = 0.5


def set_seed(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(all_true, all_probs, threshold=THRESHOLD):
    """Compute all metrics from evaluate() output."""
    m = calc_threshold_metrics(all_true, all_probs, threshold)
    return {
        'f1': m['f1_binary'],
        'precision': m['precision'],
        'recall': m['recall'],
        'mcc': m['mcc'],
        'top1': topk_accuracy(all_true, all_probs, 1),
        'top2': topk_accuracy(all_true, all_probs, 2),
        'top3': topk_accuracy(all_true, all_probs, 3),
        'per_atom_acc': per_atom_accuracy(all_true, all_probs, threshold),
        'mol_exact': molecule_exact_match(all_true, all_probs, threshold),
    }


def compute_val_loss(model, loader, device):
    """Compute validation loss."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            try:
                logits, _, _ = model.inference(batch)
            except Exception:
                logits, _, _ = model(batch)
            y_true = batch.y.to(logits.dtype)
            pw = bce_pos_weight(y_true)
            loss = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pw)
            total += loss.item()
    return total / max(1, len(loader))


def group_results_by_cyp(graphs_subset, all_true, all_probs):
    buckets = {c: {'true': [], 'probs': []} for c in CYP_LIST}
    for g, t, p in zip(graphs_subset, all_true, all_probs):
        buckets[g.cyp_name]['true'].append(t)
        buckets[g.cyp_name]['probs'].append(p)
    return buckets


def run_fold(args, graphs, train_idx, val_idx, test_idx, fold, device):
    """Run a single CV fold with full metric tracking."""
    atom_in_dim = graphs[0].x.size(-1)
    edge_dim = graphs[0].edge_attr.size(-1)

    model = GraphCliffMultiRegressor(
        atom_in_dim=atom_in_dim, edge_dim=edge_dim,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        groups=4, mid_K=3, dropout=args.dropout,
        cyp_names=CYP_LIST, num_attn_heads=args.num_attn_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]
    test_set = [graphs[i] for i in test_idx]

    # Remove leakage
    train_set = apply_no_leakage_to_dataloaders(train_set, test_set, CYP_LIST)
    train_set = apply_no_leakage_to_dataloaders(train_set, val_set, CYP_LIST)
    val_set = apply_no_leakage_to_dataloaders(val_set, test_set, CYP_LIST)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=0)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.max_epochs + 1):
        # Train
        tr_loss = train_one_epoch(args, model, train_loader, optimizer, device,
                                  args.gamma, lambda_main=args.lambda_main,
                                  lambda_attn=args.lambda_attn)

        # Train metrics
        train_true, train_probs = evaluate(model, train_loader, device)
        train_metrics = compute_metrics(train_true, train_probs)

        # Val metrics
        val_loss = compute_val_loss(model, val_loader, device)
        val_true, val_probs = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_probs)

        epoch_log = {
            'epoch': epoch,
            'train_loss': tr_loss['main'],
            'train_attn_loss': tr_loss['attn'],
            'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        history.append(epoch_log)

        # Early stopping on val F1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch <= 5 or patience_counter == 0:
            print(f"  [Fold {fold}] Epoch {epoch:3d} | "
                  f"train F1={train_metrics['f1']:.3f} top3={train_metrics['top3']:.3f} | "
                  f"val F1={val_metrics['f1']:.3f} top3={val_metrics['top3']:.3f} loss={val_loss:.4f}"
                  f"{' *best*' if patience_counter == 0 else ''}")

        if patience_counter >= args.patience:
            print(f"  [Fold {fold}] Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Load best model and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_true, test_probs = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_true, test_probs)

    # Per-CYP test metrics
    buckets = group_results_by_cyp(test_set, test_true, test_probs)
    per_cyp = {}
    for c in CYP_LIST:
        T, P = buckets[c]['true'], buckets[c]['probs']
        if T:
            per_cyp[c] = compute_metrics(T, P)
            per_cyp[c]['n_mols'] = len(T)

    return {
        'test_metrics': test_metrics,
        'per_cyp': per_cyp,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'history': history,
        'n_train': len(train_set),
        'n_val': len(val_set),
        'n_test': len(test_set),
    }


def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(args.result_dir, exist_ok=True)

    exclude_smiles = load_exclusion_smiles(args.exclusion_list) if args.exclude_tdc else set()
    graphs = load_multi_cyp(
        args.dataset_dir, CYP_LIST, exclude_smiles=exclude_smiles
    )
    if not graphs:
        raise SystemExit("No data found.")

    # Cross-validation
    y_strat = make_strata_labels(graphs)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    all_fold_results = []
    per_cyp_agg = {c: [] for c in CYP_LIST}

    for fold, (trainval_idx, test_idx) in enumerate(
        skf.split(np.arange(len(graphs)), y_strat), start=1
    ):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{args.n_splits}")
        print(f"{'='*60}")

        tr_idx, val_idx = split_train_val_by_index_stratified_cyp(
            graphs, trainval_idx, val_ratio=args.inner_val_ratio, seed=args.seed + fold
        )

        fold_result = run_fold(args, graphs, tr_idx, val_idx, test_idx, fold, device)
        all_fold_results.append(fold_result)

        # Print fold test results
        tm = fold_result['test_metrics']
        print(f"  Test: F1={tm['f1']:.3f} top1={tm['top1']:.3f} top3={tm['top3']:.3f} "
              f"MCC={tm['mcc']:.3f} (best epoch={fold_result['best_epoch']})")

        for c, m in fold_result['per_cyp'].items():
            per_cyp_agg[c].append(m)

        # Save fold history
        with open(os.path.join(args.result_dir, f'fold{fold}_history.json'), 'w') as f:
            json.dump(fold_result['history'], f, indent=2)

    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean ± std across folds)")
    print(f"{'='*60}")

    keys = ['f1', 'precision', 'recall', 'mcc', 'top1', 'top2', 'top3', 'per_atom_acc', 'mol_exact']
    overall_agg = {}
    for k in keys:
        vals = [fr['test_metrics'][k] for fr in all_fold_results]
        overall_agg[f'{k}_mean'] = float(np.mean(vals))
        overall_agg[f'{k}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {k:>14}: {overall_agg[f'{k}_mean']:.4f} ± {overall_agg[f'{k}_std']:.4f}")

    best_epochs = [fr['best_epoch'] for fr in all_fold_results]
    print(f"\n  Best epochs: {best_epochs} (mean={np.mean(best_epochs):.0f})")

    print("\nPer-CYP:")
    cyp_agg = {}
    for c in CYP_LIST:
        rows = per_cyp_agg[c]
        if not rows:
            continue
        agg = {}
        for k in keys:
            vals = [r[k] for r in rows]
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cyp_agg[c] = agg
        print(f"  {c:>5}: F1={agg['f1_mean']:.3f}±{agg['f1_std']:.3f}  "
              f"top3={agg['top3_mean']:.3f}±{agg['top3_std']:.3f}  "
              f"n={rows[0].get('n_mols','?')}")

    # Save summary
    summary = {
        'overall': overall_agg,
        'per_cyp': cyp_agg,
        'best_epochs': best_epochs,
        'args': vars(args),
    }
    with open(os.path.join(args.result_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Train final deployable checkpoint on all data
    if args.save_checkpoint:
        # Use median best epoch from CV
        final_epochs = int(np.median(best_epochs))
        print(f"\n=== Training final checkpoint on ALL data for {final_epochs} epochs ===")

        atom_in_dim = graphs[0].x.size(-1)
        edge_dim = graphs[0].edge_attr.size(-1)

        final_model = GraphCliffMultiRegressor(
            atom_in_dim=atom_in_dim, edge_dim=edge_dim,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
            groups=4, mid_K=3, dropout=args.dropout,
            cyp_names=CYP_LIST, num_attn_heads=args.num_attn_heads,
        ).to(device)
        final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.wd)
        full_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True, num_workers=0)

        for epoch in range(1, final_epochs + 1):
            tr_loss = train_one_epoch(
                args, final_model, full_loader, final_optimizer, device,
                args.gamma, lambda_main=args.lambda_main, lambda_attn=args.lambda_attn
            )
            if epoch % 10 == 0 or epoch == final_epochs:
                # Train metrics on full data
                train_true, train_probs = evaluate(final_model, full_loader, device)
                tm = compute_metrics(train_true, train_probs)
                print(f"  Epoch {epoch:3d}/{final_epochs} | "
                      f"loss={tr_loss['main']:.4f} F1={tm['f1']:.3f} top3={tm['top3']:.3f}")

        ckpt_path = os.path.join(args.result_dir, "attnsom_checkpoint.pt")
        torch.save({
            "model_state_dict": final_model.state_dict(),
            "args": vars(args),
            "cyp_list": CYP_LIST,
            "atom_in_dim": atom_in_dim,
            "edge_dim": edge_dim,
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print(f"\nAll results saved to {args.result_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrain ATTNSOM with proper metric tracking")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_dir', default=os.path.join(os.path.dirname(__file__), 'cyp_dataset'))
    parser.add_argument('--result_dir', default='/vast/projects/myatskar/design-documents/hf_home/attnsom_results_v2')

    parser.add_argument('--n_splits', default=10, type=int)
    parser.add_argument('--inner_val_ratio', default=0.1, type=float, help='Val ratio (default: 0.1, was 0.05)')

    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--patience', default=30, type=int, help='Early stopping patience on val F1')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--loss', default='focal', choices=['bce', 'focal'])
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--gamma', default=1, type=float)

    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--dropout', default=0, type=float)

    parser.add_argument('--lambda_main', default=1.0, type=float)
    parser.add_argument('--lambda_attn', default=0.5, type=float)
    parser.add_argument('--num_attn_heads', default=4, type=int)

    parser.add_argument('--exclude_tdc', default=False, action='store_true')
    parser.add_argument('--exclusion_list', default=None, type=str)
    parser.add_argument('--save_checkpoint', default=False, action='store_true')

    args = parser.parse_args()
    main(args)
