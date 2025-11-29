"""
Influence Functions utilities for pretrained models (no LoRA)
Adapted for Pile samples with member/nonmember data
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


class TextDataset(Dataset):
    """Simple text dataset for influence functions"""

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize
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
    """Load data from JSONL file"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                texts.append(item['text'])
    return texts


def collect_gradient(model, tokenizer, member_texts, nonmember_texts,
                     max_length=512, batch_size=1,
                     use_all_params=False, target_modules=None):
    """
    Collect gradients for member and nonmember texts

    Args:
        model: Pretrained model
        tokenizer: Tokenizer
        member_texts: List of member text strings
        nonmember_texts: List of nonmember text strings
        max_length: Maximum sequence length
        batch_size: Batch size (default 1 for influence functions)
        use_all_params: If True, use all parameters. If False, use only target_modules
        target_modules: List of module names to track (e.g., ['embed', 'layer'])

    Returns:
        member_grad_dict: Dictionary of gradients for member samples
        nonmember_grad_dict: Dictionary of gradients for nonmember samples
    """

    # Create datasets
    member_dataset = TextDataset(member_texts, tokenizer, max_length)
    nonmember_dataset = TextDataset(nonmember_texts, tokenizer, max_length)

    # Create dataloaders
    collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")
    member_loader = DataLoader(member_dataset, shuffle=False, batch_size=batch_size)
    nonmember_loader = DataLoader(nonmember_dataset, shuffle=False, batch_size=batch_size)

    model.eval()

    # Collect member gradients
    print("Collecting member gradients...")
    member_grad_dict = {}
    for step, batch in enumerate(tqdm(member_loader)):
        model.zero_grad()

        # Move to device
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Collect gradients
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Filter parameters if needed
                if use_all_params:
                    grad_dict[name] = param.grad.cpu().clone()
                elif target_modules is not None:
                    if any(module in name for module in target_modules):
                        grad_dict[name] = param.grad.cpu().clone()
                else:
                    # Default: only embedding and last layer
                    if 'embed' in name or 'lm_head' in name or 'final' in name:
                        grad_dict[name] = param.grad.cpu().clone()

        member_grad_dict[step] = grad_dict
        del grad_dict

    # Collect nonmember gradients
    print("Collecting nonmember gradients...")
    nonmember_grad_dict = {}
    for step, batch in enumerate(tqdm(nonmember_loader)):
        model.zero_grad()

        # Move to device
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Collect gradients
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Filter parameters if needed
                if use_all_params:
                    grad_dict[name] = param.grad.cpu().clone()
                elif target_modules is not None:
                    if any(module in name for module in target_modules):
                        grad_dict[name] = param.grad.cpu().clone()
                else:
                    # Default: only embedding and last layer
                    if 'embed' in name or 'lm_head' in name or 'final' in name:
                        grad_dict[name] = param.grad.cpu().clone()

        nonmember_grad_dict[step] = grad_dict
        del grad_dict

    return member_grad_dict, nonmember_grad_dict


def influence_function(member_grad_dict, nonmember_grad_dict,
                       hvp_cal='gradient_match',
                       lambda_const_param=10,
                       n_iteration=10,
                       alpha_const=1.):
    """
    Calculate influence function scores

    Args:
        member_grad_dict: Gradients for member samples
        nonmember_grad_dict: Gradients for nonmember samples
        hvp_cal: Method for HVP calculation ['gradient_match', 'DataInf', 'LiSSA', 'Original']
        lambda_const_param: Lambda constant parameter
        n_iteration: Number of iterations for LiSSA
        alpha_const: Alpha constant for LiSSA

    Returns:
        DataFrame of influence scores (rows=member, cols=nonmember)
    """

    hvp_dict = defaultdict(dict)
    IF_dict = defaultdict(dict)
    n_train = len(member_grad_dict.keys())

    def calculate_lambda_const(grad_dict, weight_name):
        S = torch.zeros(len(grad_dict.keys()))
        for id in grad_dict:
            tmp_grad = grad_dict[id][weight_name]
            S[id] = torch.mean(tmp_grad**2)
        return torch.mean(S) / lambda_const_param

    if hvp_cal == 'gradient_match':
        # Simple gradient matching (fastest)
        hvp_dict = nonmember_grad_dict.copy()

    elif hvp_cal == 'DataInf':
        print("Computing HVP using DataInf...")
        for val_id in tqdm(nonmember_grad_dict.keys()):
            for weight_name in nonmember_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(member_grad_dict, weight_name)

                hvp = torch.zeros(nonmember_grad_dict[val_id][weight_name].shape)
                for tr_id in member_grad_dict:
                    tmp_grad = member_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(nonmember_grad_dict[val_id][weight_name] * tmp_grad) / \
                            (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (nonmember_grad_dict[val_id][weight_name] - C_tmp * tmp_grad) / \
                           (n_train * lambda_const)

                hvp_dict[val_id][weight_name] = hvp

    elif hvp_cal == 'LiSSA':
        print("Computing HVP using LiSSA...")
        for val_id in tqdm(nonmember_grad_dict.keys()):
            for weight_name in nonmember_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(member_grad_dict, weight_name)

                running_hvp = nonmember_grad_dict[val_id][weight_name]
                for _ in range(n_iteration):
                    hvp_tmp = torch.zeros(nonmember_grad_dict[val_id][weight_name].shape)
                    for tr_id in member_grad_dict:
                        tmp_grad = member_grad_dict[tr_id][weight_name]
                        hvp_tmp += (torch.sum(tmp_grad * running_hvp) * tmp_grad - \
                                   lambda_const * running_hvp) / n_train / 1e3

                    running_hvp = nonmember_grad_dict[val_id][weight_name] + running_hvp - \
                                 alpha_const * hvp_tmp

                hvp_dict[val_id][weight_name] = running_hvp

    else:
        raise ValueError(f"Unknown hvp_cal method: {hvp_cal}")

    # Calculate influence scores
    print("Computing influence scores...")
    for tr_id in tqdm(member_grad_dict.keys()):
        for val_id in nonmember_grad_dict.keys():
            if_tmp_value = 0
            for weight_name in nonmember_grad_dict[val_id]:
                if weight_name in member_grad_dict[tr_id]:
                    if_tmp_value += torch.sum(hvp_dict[val_id][weight_name] * \
                                             member_grad_dict[tr_id][weight_name])

            IF_dict[tr_id][val_id] = -if_tmp_value.item()

    return pd.DataFrame(IF_dict, dtype=float)


def calculate_influence_metrics(influence_df):
    """
    Calculate metrics from influence function results

    Args:
        influence_df: DataFrame of influence scores (rows=member, cols=nonmember)

    Returns:
        Dictionary of metrics
    """
    # For each nonmember, find most influential member
    most_influential = []
    influence_scores = []

    for col in influence_df.columns:
        scores = influence_df[col].values
        max_idx = np.argmax(scores)
        most_influential.append(max_idx)
        influence_scores.append(scores[max_idx])

    metrics = {
        'mean_max_influence': np.mean(influence_scores),
        'std_max_influence': np.std(influence_scores),
        'mean_influence': influence_df.values.mean(),
        'std_influence': influence_df.values.std()
    }

    return metrics
