import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib
import json

import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


def convert_huggingface_data_to_list_dic(dataset):
    """Convert HuggingFace dataset to list of dictionaries"""
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data


def load_local_jsonl(file_path):
    """Load data from a JSONL file

    Args:
        file_path: Path to JSONL file

    Returns:
        List of text strings
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item['text'])
    return data


def load_local_pile_samples(pile_samples_dir, domain):
    """Load member and nonmember data from local pile_samples directory

    Args:
        pile_samples_dir: Path to pile_samples directory
        domain: Domain name (e.g., 'arxiv', 'github')

    Returns:
        List of dictionaries with 'member' and 'nonmember' keys
    """
    domain_dir = os.path.join(pile_samples_dir, domain)

    if not os.path.exists(domain_dir):
        raise ValueError(f"Domain directory not found: {domain_dir}")

    train_file = os.path.join(domain_dir, 'train_text.jsonl')
    test_file = os.path.join(domain_dir, 'test_text.jsonl')

    if not os.path.exists(train_file):
        raise ValueError(f"Train file not found: {train_file}")
    if not os.path.exists(test_file):
        raise ValueError(f"Test file not found: {test_file}")

    print(f"Loading member data from: {train_file}")
    member_texts = load_local_jsonl(train_file)
    print(f"✅ Loaded {len(member_texts)} member samples")

    print(f"Loading nonmember data from: {test_file}")
    nonmember_texts = load_local_jsonl(test_file)
    print(f"✅ Loaded {len(nonmember_texts)} nonmember samples")

    # Create paired data
    data = []
    for member_text, nonmember_text in zip(member_texts, nonmember_texts):
        data.append({
            'member': member_text,
            'nonmember': nonmember_text
        })

    return data


def load_model(model_path, half=False, int8=False):
    """Load model and tokenizer with optional quantization"""
    int8_kwargs = {}
    half_kwargs = {}

    if int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)

    if 'mamba' in model_path:
        try:
            from transformers import MambaForCausalLM
        except ImportError:
            raise ImportError("MambaForCausalLM not available")
        model = MambaForCausalLM.from_pretrained(
            model_path, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def compute_single_text_scores(text, model, tokenizer, mink_ratios):
    """Compute MIA scores for a single text

    Returns:
        Dictionary with scores for this text
    """
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    loss, logits = outputs[:2]
    ll = -loss.item()  # log-likelihood

    scores = {}
    # Loss and zlib
    scores['loss'] = ll
    scores['zlib'] = ll / len(zlib.compress(bytes(text, 'utf-8')))

    # Min-K and Min-K++
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    # Min-K
    for ratio in mink_ratios:
        k_length = int(len(token_log_probs) * ratio)
        topk = np.sort(token_log_probs.cpu())[:k_length]
        scores[f'mink_{ratio}'] = np.mean(topk).item()

    # Min-K++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    for ratio in mink_ratios:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]
        scores[f'mink++_{ratio}'] = np.mean(topk).item()

    return scores


def compute_mia_scores_mimir(model, tokenizer, data, mink_ratios):
    """Compute MIA scores for mimir dataset

    Args:
        model: Language model
        tokenizer: Tokenizer
        data: List of mimir data samples (each has 'member' and 'nonmember')
        mink_ratios: List of ratios for Min-K calculation

    Returns:
        Dictionary of scores for each metric, list of labels
    """
    all_scores = defaultdict(list)
    labels = []

    for i, d in enumerate(tqdm(data, total=len(data), desc='Computing MIA scores')):
        # Process member (label=1)
        member_scores = compute_single_text_scores(d['member'], model, tokenizer, mink_ratios)
        for key, value in member_scores.items():
            all_scores[key].append(value)
        labels.append(1)

        # Process nonmember (label=0)
        nonmember_scores = compute_single_text_scores(d['nonmember'], model, tokenizer, mink_ratios)
        for key, value in nonmember_scores.items():
            all_scores[key].append(value)
        labels.append(0)

    return all_scores, labels


def get_metrics(scores, labels):
    """Compute AUROC, FPR@95, TPR@5"""
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05


@hydra.main(version_base=None, config_path="../config", config_name="mia_analysis")
def main(cfg: DictConfig):
    """Main function for MIA analysis"""

    print("=" * 60)
    print("MIA Analysis Configuration")
    print("=" * 60)
    print(f"Model: {cfg.model_path}")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Config: {cfg.dataset_config}")
    print(f"Split: {cfg.dataset_split}")
    print(f"Half precision: {cfg.half}")
    print(f"Int8 quantization: {cfg.int8}")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model: {cfg.model_path}...")
    model, tokenizer = load_model(cfg.model_path, half=cfg.half, int8=cfg.int8)
    print("✅ Model loaded successfully")
    print()

    # Load dataset
    if cfg.get('use_local_files', False):
        print(f"Loading local dataset from: {cfg.pile_samples_dir}")
        print(f"Domain: {cfg.dataset_config}")
        data = load_local_pile_samples(cfg.pile_samples_dir, cfg.dataset_config)
        print(f"✅ Loaded {len(data)} sample pairs")
    else:
        print(f"Loading dataset: {cfg.dataset_name} ({cfg.dataset_config})...")
        dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
        data = convert_huggingface_data_to_list_dic(dataset)
        print(f"✅ Loaded {len(data)} samples")
    print()

    # Compute MIA scores
    print("Computing MIA scores...")
    scores, labels = compute_mia_scores_mimir(model, tokenizer, data, cfg.mink_ratios)
    print(f"✅ Scores computed ({len(labels)} total samples)")
    print()

    # Compute metrics
    print("Computing metrics...")
    results = defaultdict(list)

    for method, method_scores in scores.items():
        auroc, fpr95, tpr05 = get_metrics(method_scores, labels)

        results['method'].append(method)
        results['auroc'].append(f"{auroc:.1%}")
        results['fpr95'].append(f"{fpr95:.1%}")
        results['tpr05'].append(f"{tpr05:.1%}")

    # Create results dataframe
    df = pd.DataFrame(results)
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df)
    print("=" * 60)

    # Save results
    if cfg.save_csv:
        # Use absolute path relative to script directory
        output_base = os.path.join(script_dir, cfg.output_dir)
        save_root = os.path.join(output_base, cfg.dataset_config)
        os.makedirs(save_root, exist_ok=True)

        model_id = cfg.model_path.split('/')[-1]
        output_file = os.path.join(save_root, f"{model_id}.csv")

        if os.path.isfile(output_file):
            df.to_csv(output_file, index=False, mode='a', header=False)
        else:
            df.to_csv(output_file, index=False)

        print()
        print(f"✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
