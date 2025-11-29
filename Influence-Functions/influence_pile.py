"""
Influence Functions for Pile Samples (Pretrained models, no LoRA)
"""

import os
import sys
import pickle
import argparse
import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


class TextDataset(Dataset):
    """Simple text dataset"""

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }


def load_local_jsonl(file_path):
    """Load JSONL file"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                texts.append(item['text'])
    return texts


def collect_hidden_states(model, tokenizer, member_texts, nonmember_texts, max_length=512):
    """Collect hidden states (last layer, last token) for member and nonmember samples"""

    member_dataset = TextDataset(member_texts, tokenizer, max_length)
    nonmember_dataset = TextDataset(nonmember_texts, tokenizer, max_length)

    collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")
    member_loader = DataLoader(member_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)
    nonmember_loader = DataLoader(nonmember_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

    model.eval()

    # Collect member hidden states
    print("Collecting member hidden states...")
    member_hidden_states = []
    for step, batch in enumerate(tqdm(member_loader)):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)

        # Get last layer, last token hidden state
        # Convert BFloat16 to float32 before numpy conversion
        hidden_state = outputs.hidden_states[-1][:, -1, :].float().cpu().numpy().flatten()
        member_hidden_states.append(hidden_state)

    # Collect nonmember hidden states
    print("Collecting nonmember hidden states...")
    nonmember_hidden_states = []
    for step, batch in enumerate(tqdm(nonmember_loader)):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)

        # Get last layer, last token hidden state
        # Convert BFloat16 to float32 before numpy conversion
        hidden_state = outputs.hidden_states[-1][:, -1, :].float().cpu().numpy().flatten()
        nonmember_hidden_states.append(hidden_state)

    return np.array(member_hidden_states), np.array(nonmember_hidden_states)


def collect_gradient(model, tokenizer, member_texts, nonmember_texts, max_length=512):
    """Collect gradients for member and nonmember samples"""

    member_dataset = TextDataset(member_texts, tokenizer, max_length)
    nonmember_dataset = TextDataset(nonmember_texts, tokenizer, max_length)

    collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")
    member_loader = DataLoader(member_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)
    nonmember_loader = DataLoader(nonmember_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

    model.eval()

    # Collect member gradients
    print("Collecting member gradients...")
    member_grad_dict = {}
    for step, batch in enumerate(tqdm(member_loader)):
        model.zero_grad()
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Only track embedding and lm_head for efficiency
                if 'embed' in name or 'lm_head' in name:
                    grad_dict[name] = param.grad.cpu().clone()
                # Clear gradient immediately
                param.grad = None

        member_grad_dict[step] = grad_dict

        # Clean up memory
        del batch, outputs, loss, grad_dict
        if step % 100 == 0:
            torch.cuda.empty_cache()

    # Collect nonmember gradients
    print("Collecting nonmember gradients...")
    nonmember_grad_dict = {}
    for step, batch in enumerate(tqdm(nonmember_loader)):
        model.zero_grad()
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Only track embedding and lm_head for efficiency
                if 'embed' in name or 'lm_head' in name:
                    grad_dict[name] = param.grad.cpu().clone()
                # Clear gradient immediately
                param.grad = None

        nonmember_grad_dict[step] = grad_dict

        # Clean up memory
        del batch, outputs, loss, grad_dict
        if step % 100 == 0:
            torch.cuda.empty_cache()

    # Final cleanup
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return member_grad_dict, nonmember_grad_dict


def influence_function(member_grad_dict, nonmember_grad_dict, hvp_cal='gradient_match',
                       lambda_const_param=10, n_iteration=10, alpha_const=1.):
    """Calculate influence function scores"""

    hvp_dict = defaultdict(dict)
    IF_dict = defaultdict(dict)
    n_train = len(member_grad_dict.keys())

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    def calculate_lambda_const(grad_dict, weight_name):
        S = torch.zeros(len(grad_dict.keys()))
        for id in grad_dict:
            tmp_grad = grad_dict[id][weight_name]
            S[id] = torch.mean(tmp_grad**2)
        return torch.mean(S) / lambda_const_param

    if hvp_cal == 'gradient_match':
        print("Using gradient_match method (fast)...")
        hvp_dict = nonmember_grad_dict.copy()
        print(f"✓ HVP dictionary created ({len(hvp_dict)} items)")

    elif hvp_cal == 'DataInf':
        print("Computing HVP using DataInf (GPU-accelerated)...")

        # Pre-move all gradients to GPU once (more efficient)
        print("Moving all gradients to GPU...")
        member_grad_dict_gpu = {}
        for tr_id in tqdm(member_grad_dict.keys(), desc="Moving member grads to GPU"):
            member_grad_dict_gpu[tr_id] = {}
            for weight_name in member_grad_dict[tr_id]:
                member_grad_dict_gpu[tr_id][weight_name] = member_grad_dict[tr_id][weight_name].to(device)

        nonmember_grad_dict_gpu = {}
        for val_id in tqdm(nonmember_grad_dict.keys(), desc="Moving nonmember grads to GPU"):
            nonmember_grad_dict_gpu[val_id] = {}
            for weight_name in nonmember_grad_dict[val_id]:
                nonmember_grad_dict_gpu[val_id][weight_name] = nonmember_grad_dict[val_id][weight_name].to(device)

        print("✓ All gradients moved to GPU, starting computation...")

        for val_id in tqdm(nonmember_grad_dict_gpu.keys(), desc="Processing nonmembers"):
            for weight_name in nonmember_grad_dict_gpu[val_id]:
                lambda_const = calculate_lambda_const(member_grad_dict, weight_name)

                # Gradients already on GPU
                test_grad = nonmember_grad_dict_gpu[val_id][weight_name]
                hvp = torch.zeros_like(test_grad, device=device)

                # Vectorized computation for all train samples
                # Stack all train gradients for this weight into a single tensor
                train_grads_list = []
                for tr_id in member_grad_dict_gpu:
                    if weight_name in member_grad_dict_gpu[tr_id]:
                        train_grads_list.append(member_grad_dict_gpu[tr_id][weight_name])

                if train_grads_list:
                    # Stack into [num_train, grad_shape]
                    train_grads = torch.stack(train_grads_list)  # [N, ...]

                    # Reshape for batch computation
                    original_shape = train_grads.shape[1:]
                    train_grads_flat = train_grads.flatten(1)  # [N, D]
                    test_grad_flat = test_grad.flatten()  # [D]

                    # Vectorized computation
                    # C_tmp = (v · g_i) / (λ + ||g_i||²)
                    dot_products = torch.matmul(train_grads_flat, test_grad_flat)  # [N]
                    grad_norms = torch.sum(train_grads_flat ** 2, dim=1)  # [N]
                    C_tmp = dot_products / (lambda_const + grad_norms)  # [N]

                    # hvp = Σ (v - C_i * g_i) / (N * λ)
                    # Expand C_tmp for broadcasting
                    C_tmp_expanded = C_tmp.view(-1, 1)  # [N, 1]
                    corrections = train_grads_flat * C_tmp_expanded  # [N, D]
                    hvp_flat = (test_grad_flat.unsqueeze(0) - corrections).sum(0) / (n_train * lambda_const)
                    hvp = hvp_flat.reshape(original_shape)

                hvp_dict[val_id][weight_name] = hvp.cpu()

        # Clean up GPU memory
        del member_grad_dict_gpu, nonmember_grad_dict_gpu
        torch.cuda.empty_cache()
        print("✓ GPU computation complete, results moved to CPU")

    elif hvp_cal == 'repsim':
        print("Using repsim method (hidden state cosine similarity)...")
        print("Note: This method requires hidden states, not gradients.")
        print("Please use --hvp_method hidden_state instead, or provide hidden states directly.")
        raise ValueError("repsim method should be called separately with hidden states")

    else:
        raise ValueError(f"Unknown hvp_cal method: {hvp_cal}")

    # Calculate influence scores
    n_members = len(member_grad_dict.keys())
    n_nonmembers = len(nonmember_grad_dict.keys())
    print(f"Computing influence scores ({n_members} members × {n_nonmembers} nonmembers = {n_members * n_nonmembers} pairs)...")
    print("This may take a while for large datasets...")

    for tr_id in tqdm(member_grad_dict.keys(), desc="Computing influence"):
        for val_id in nonmember_grad_dict.keys():
            if_tmp_value = 0
            for weight_name in nonmember_grad_dict[val_id]:
                if weight_name in member_grad_dict[tr_id]:
                    if_tmp_value += torch.sum(hvp_dict[val_id][weight_name] * \
                                             member_grad_dict[tr_id][weight_name])
            IF_dict[tr_id][val_id] = -if_tmp_value.item()

    print(f"✅ Influence computation complete!")

    return pd.DataFrame(IF_dict, dtype=float)


def compute_repsim_similarity(member_hidden_states, nonmember_hidden_states):
    """
    Compute cosine similarity between hidden states (repsim method)

    Args:
        member_hidden_states: numpy array [N_members, hidden_dim]
        nonmember_hidden_states: numpy array [N_nonmembers, hidden_dim]

    Returns:
        DataFrame with similarity scores [members x nonmembers]
    """
    print("Computing cosine similarity matrix (repsim) with GPU acceleration...")

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to torch tensors and move to GPU
    member_tensor = torch.from_numpy(member_hidden_states).to(device)
    nonmember_tensor = torch.from_numpy(nonmember_hidden_states).to(device)

    # Normalize for cosine similarity
    member_norms = torch.norm(member_tensor, dim=1, keepdim=True)
    nonmember_norms = torch.norm(nonmember_tensor, dim=1, keepdim=True)

    member_normalized = member_tensor / (member_norms + 1e-8)
    nonmember_normalized = nonmember_tensor / (nonmember_norms + 1e-8)

    # Compute cosine similarity: [N_members, N_nonmembers]
    similarity_matrix = torch.matmul(member_normalized, nonmember_normalized.T)

    # Move back to CPU and convert to numpy
    similarity_matrix = similarity_matrix.cpu().numpy()

    print(f"✅ Computed similarity matrix: {similarity_matrix.shape}")
    return pd.DataFrame(similarity_matrix, dtype=float)


def get_metrics(scores, labels):
    """Compute AUROC, FPR@95, TPR@5 (same as run_hydra.py)"""
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05


def compute_mia_metrics(influence_df):
    """
    Convert influence scores to MIA scores and compute metrics

    Args:
        influence_df: DataFrame with rows=member_idx, cols=nonmember_idx, values=influence_scores

    Returns:
        Dictionary of MIA metrics
    """
    # For each sample, compute aggregate influence score
    # Higher influence = more similar = more likely to be member

    # Member scores: average influence with nonmembers
    member_scores = influence_df.mean(axis=1).values  # Average across columns (nonmembers)

    # Nonmember scores: average influence from members
    nonmember_scores = influence_df.mean(axis=0).values  # Average across rows (members)

    # Combine scores and labels
    all_scores = np.concatenate([member_scores, nonmember_scores])
    all_labels = np.concatenate([
        np.ones(len(member_scores)),   # member = 1
        np.zeros(len(nonmember_scores))  # nonmember = 0
    ])

    # Compute metrics
    auroc, fpr95, tpr05 = get_metrics(all_scores, all_labels)

    results = {
        'auroc': auroc,
        'fpr95': fpr95,
        'tpr05': tpr05,
        'member_score_mean': member_scores.mean(),
        'member_score_std': member_scores.std(),
        'nonmember_score_mean': nonmember_scores.mean(),
        'nonmember_score_std': nonmember_scores.std(),
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Influence Functions for Pile Samples")
    parser.add_argument('--model', type=str, default='pythia-2.8b', help='model name')
    parser.add_argument('--domain', type=str, required=True, help='domain name (arxiv, github, etc.)')
    parser.add_argument('--data_dir', type=str, default='../pile_samples', help='pile_samples directory')
    parser.add_argument('--max_length', type=int, default=512, help='max sequence length')
    parser.add_argument('--max_samples', type=int, default=-1, help='max samples to use (-1 for all)')
    parser.add_argument('--lambda_c', type=float, default=10, help='lambda const')
    parser.add_argument('--iter', type=int, default=3, help='#iteration for LiSSA')
    parser.add_argument('--alpha', type=float, default=1., help='alpha const for LiSSA')
    parser.add_argument('--hvp_method', type=str, default='gradient_match',
                       help='HVP method: gradient_match, DataInf, LiSSA, repsim')
    parser.add_argument('--grad_cache', action='store_true', default=False,
                       help='use cached gradients')
    parser.add_argument('--only_collect_grad', action='store_true', default=False,
                       help='only collect gradients and exit')
    args = parser.parse_args()

    # Model path mapping
    model_mapping = {
        'pythia-70m': 'EleutherAI/pythia-70m',
        'pythia-160m': 'EleutherAI/pythia-160m',
        'pythia-410m': 'EleutherAI/pythia-410m',
        'pythia-1.4b': 'EleutherAI/pythia-1.4b',
        'pythia-2.8b': 'EleutherAI/pythia-2.8b',
        'pythia-6.7b': 'EleutherAI/pythia-6.7b',
        'pythia-12b': 'EleutherAI/pythia-12b',
    }

    if args.model not in model_mapping:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(model_mapping.keys())}")

    model_path = model_mapping[args.model]
    print(f"Model: {model_path}")

    # Setup paths
    domain_dir = os.path.join(args.data_dir, args.domain)
    train_file = os.path.join(domain_dir, 'train_text.jsonl')
    test_file = os.path.join(domain_dir, 'test_text.jsonl')

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Create directories (absolute path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grad_dir = os.path.join(script_dir, 'grad')
    cache_dir = os.path.join(script_dir, 'cache')

    os.makedirs(grad_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Check if using repsim method (hidden states)
    if args.hvp_method == 'repsim':
        # Use hidden state cache files
        hidden_cache_member = os.path.join(grad_dir, f'{args.model}_{args.domain}_member_hidden.npy')
        hidden_cache_nonmember = os.path.join(grad_dir, f'{args.model}_{args.domain}_nonmember_hidden.npy')

        if args.grad_cache and os.path.exists(hidden_cache_member) and os.path.exists(hidden_cache_nonmember):
            print("Loading cached hidden states...")
            member_hidden_states = np.load(hidden_cache_member)
            nonmember_hidden_states = np.load(hidden_cache_nonmember)
            print(f"✅ Loaded {len(member_hidden_states)} member + {len(nonmember_hidden_states)} nonmember hidden states")
        else:
            # Load model
            print(f"Loading model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
            model.eval()
            print("Model loaded")

            # Load data
            print(f"Loading data from {domain_dir}")
            member_texts = load_local_jsonl(train_file)
            nonmember_texts = load_local_jsonl(test_file)
            print(f"Loaded {len(member_texts)} member and {len(nonmember_texts)} nonmember samples")

            # Limit samples
            if args.max_samples > 0:
                member_texts = member_texts[:args.max_samples]
                nonmember_texts = nonmember_texts[:args.max_samples]
                print(f"Limited to {args.max_samples} samples")

            # Collect hidden states
            member_hidden_states, nonmember_hidden_states = collect_hidden_states(
                model, tokenizer, member_texts, nonmember_texts, args.max_length
            )

            # Save hidden states
            np.save(hidden_cache_member, member_hidden_states)
            np.save(hidden_cache_nonmember, nonmember_hidden_states)
            print(f"Hidden states saved")

            del model, tokenizer
            torch.cuda.empty_cache()

        # Compute cosine similarity
        print(f"Computing similarity with method: {args.hvp_method}")
        influence_df = compute_repsim_similarity(member_hidden_states, nonmember_hidden_states)

    else:
        # Gradient-based methods
        grad_cache_member = os.path.join(grad_dir, f'{args.model}_{args.domain}_member.pkl')
        grad_cache_nonmember = os.path.join(grad_dir, f'{args.model}_{args.domain}_nonmember.pkl')

        # Load or collect gradients
        if args.grad_cache and os.path.exists(grad_cache_member) and os.path.exists(grad_cache_nonmember):
            print("Loading cached gradients...")

            # Check file sizes
            member_size_gb = os.path.getsize(grad_cache_member) / (1024**3)
            nonmember_size_gb = os.path.getsize(grad_cache_nonmember) / (1024**3)
            print(f"  Member gradients: {member_size_gb:.1f} GB")
            print(f"  Nonmember gradients: {nonmember_size_gb:.1f} GB")
            print("  This may take several minutes for large files...")

            print("  Loading member gradients...")
            with open(grad_cache_member, 'rb') as f:
                member_grad_dict = pickle.load(f)
            print(f"  ✓ Loaded {len(member_grad_dict)} member gradients")

            print("  Loading nonmember gradients...")
            with open(grad_cache_nonmember, 'rb') as f:
                nonmember_grad_dict = pickle.load(f)
            print(f"  ✓ Loaded {len(nonmember_grad_dict)} nonmember gradients")

            print(f"✅ Total: {len(member_grad_dict)} member + {len(nonmember_grad_dict)} nonmember gradients loaded")
        else:
            # Load model
            print(f"Loading model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
            model.eval()
            print("Model loaded")

            # Load data
            print(f"Loading data from {domain_dir}")
            member_texts = load_local_jsonl(train_file)
            nonmember_texts = load_local_jsonl(test_file)
            print(f"Loaded {len(member_texts)} member and {len(nonmember_texts)} nonmember samples")

            # Limit samples
            if args.max_samples > 0:
                member_texts = member_texts[:args.max_samples]
                nonmember_texts = nonmember_texts[:args.max_samples]
                print(f"Limited to {args.max_samples} samples")

            # Collect gradients
            member_grad_dict, nonmember_grad_dict = collect_gradient(
                model, tokenizer, member_texts, nonmember_texts, args.max_length
            )

            # Save gradients
            with open(grad_cache_member, 'wb') as f:
                pickle.dump(member_grad_dict, f)
            with open(grad_cache_nonmember, 'wb') as f:
                pickle.dump(nonmember_grad_dict, f)
            print(f"Gradients saved")

            if args.only_collect_grad:
                print("Gradient collection complete. Exiting.")
                exit()

            del model, tokenizer
            torch.cuda.empty_cache()

        # Compute influence
        print(f"Computing influence with method: {args.hvp_method}")
        influence_df = influence_function(
            member_grad_dict, nonmember_grad_dict,
            hvp_cal=args.hvp_method,
            lambda_const_param=args.lambda_c,
            n_iteration=args.iter,
            alpha_const=args.alpha
        )

    # Save influence scores
    result_file = os.path.join(cache_dir, f'{args.model}_{args.domain}_{args.hvp_method}.csv')
    influence_df.to_csv(result_file, index=True)
    print(f"Results saved to {result_file}")

    # Compute MIA metrics
    print("\nComputing MIA metrics...")
    mia_metrics = compute_mia_metrics(influence_df)

    # Print results
    print("\n" + "=" * 60)
    print("MIA METRICS (Membership Inference Attack)")
    print("=" * 60)
    print(f"AUROC:       {mia_metrics['auroc']:.1%}")
    print(f"FPR@95:      {mia_metrics['fpr95']:.1%}")
    print(f"TPR@5:       {mia_metrics['tpr05']:.1%}")
    print("-" * 60)
    print(f"Member score (mean ± std):    {mia_metrics['member_score_mean']:.6f} ± {mia_metrics['member_score_std']:.6f}")
    print(f"Nonmember score (mean ± std): {mia_metrics['nonmember_score_mean']:.6f} ± {mia_metrics['nonmember_score_std']:.6f}")
    print("=" * 60)

    # Save metrics
    metrics_file = os.path.join(cache_dir, f'{args.model}_{args.domain}_{args.hvp_method}_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("MIA METRICS (Membership Inference Attack)\n")
        f.write("=" * 60 + "\n")
        f.write(f"AUROC:       {mia_metrics['auroc']:.6f}\n")
        f.write(f"FPR@95:      {mia_metrics['fpr95']:.6f}\n")
        f.write(f"TPR@5:       {mia_metrics['tpr05']:.6f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Member score (mean ± std):    {mia_metrics['member_score_mean']:.6f} ± {mia_metrics['member_score_std']:.6f}\n")
        f.write(f"Nonmember score (mean ± std): {mia_metrics['nonmember_score_mean']:.6f} ± {mia_metrics['nonmember_score_std']:.6f}\n")
        f.write("=" * 60 + "\n")
    print(f"\nMetrics saved to {metrics_file}")
