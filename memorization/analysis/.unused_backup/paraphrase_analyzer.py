import sys
import os
import logging
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_dir)

from evaluate_util import run_generation
from memorization.utils import compute_per_token_accuracy, combine_dual_model_results

logger = logging.getLogger(__name__)


class DualModelAnalyzer:

    def __init__(self, batch_size: int, output_dir: str, model_config: Any):
      
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.model_config = model_config

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"🚀 DualModelAnalyzer initialized with output_dir={output_dir}")

    def run_dual_model_analysis(self, full_model, retain_model, tokenizer, dataset, tag: str = "analysis") -> List[Dict[str, Any]]:

        logger.info(f"Running dual model analysis with tag '{tag}'...")

        # Set models to eval mode
        full_model.eval()
        retain_model.eval()

        # Custom collate function for TextDatasetQA
        def collate_fn(samples):
            # TextDatasetQA returns (input_ids, labels, attention_mask) directly
            input_ids = torch.stack([sample[0] for sample in samples])
            labels = torch.stack([sample[1] for sample in samples])
            attention_mask = torch.stack([sample[2] for sample in samples])

            return input_ids, labels, attention_mask

        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        all_results = []

        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Analyzing ({tag})")):
                # Unpack batch from collate_fn
                input_ids, labels, attention_mask = batch

                # Move to device
                input_ids = input_ids.to(full_model.device)
                labels = labels.to(full_model.device)
                attention_mask = attention_mask.to(full_model.device)

                batch_dict = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

                # Calculate counterfactual scores
                acc_in_scores, acc_out_scores = self._calculate_counterfactual_scores(
                    batch_dict, full_model, retain_model, tokenizer
                )

                # Calculate memorization and simplicity
                memorization_scores = [acc_in - acc_out for acc_in, acc_out in zip(acc_in_scores, acc_out_scores)]
                simplicity_scores = [acc_in + acc_out for acc_in, acc_out in zip(acc_in_scores, acc_out_scores)]

                # Generate text for inspection
                input_texts, output_texts, ground_truths = run_generation(
                    self.model_config, batch_dict, full_model, tokenizer
                )

                # Store results
                for i in range(len(output_texts)):
                    result = {
                        'question': input_texts[i],
                        'generated_answer': output_texts[i],
                        'ground_truth': ground_truths[i],
                        'acc_in_score': acc_in_scores[i],
                        'acc_out_score': acc_out_scores[i],
                        'memorization_score': memorization_scores[i],
                        'simplicity_score': simplicity_scores[i],
                        'batch_idx': batch_idx,
                        'example_idx': i,
                        'tag': tag
                    }
                    all_results.append(result)

        logger.info(f"✅ Dual model analysis completed: {len(all_results)} examples processed")

        # Calculate and log summary statistics
        if all_results:
            avg_acc_in = np.mean([r['acc_in_score'] for r in all_results])
            avg_acc_out = np.mean([r['acc_out_score'] for r in all_results])
            avg_memorization = np.mean([r['memorization_score'] for r in all_results])
            avg_simplicity = np.mean([r['simplicity_score'] for r in all_results])

            logger.info(f"📊 Summary for '{tag}':")
            logger.info(f"  • Avg Acc_IN (full model): {avg_acc_in:.4f}")
            logger.info(f"  • Avg Acc_OUT (retain model): {avg_acc_out:.4f}")
            logger.info(f"  • Avg Memorization: {avg_memorization:.4f}")
            logger.info(f"  • Avg Simplicity: {avg_simplicity:.4f}")

        return all_results

    def combine_and_save_results(self, original_results: List[Dict[str, Any]],
                                 paraphrase_results: List[Dict[str, Any]],
                                 tag: str = "combined") -> Dict[str, Any]:
        """Combine original and paraphrase results and save to disk

        Args:
            original_results: Results from analyzing original questions
            paraphrase_results: Results from analyzing paraphrased questions
            tag: Tag for identifying this combined analysis

        Returns:
            Combined results dictionary
        """
        return combine_dual_model_results(original_results, paraphrase_results, self.output_dir, tag)

    def _calculate_counterfactual_scores(self, batch_dict, full_model, retain_model, tokenizer):
        """Calculate counterfactual memorization scores using per-token accuracy"""
        with torch.no_grad():
            # Calculate per-token accuracy for full model (IN condition)
            full_model.eval()
            # Move batch to full_model's device
            batch_full = {k: v.to(full_model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch_dict.items()}
            acc_in_scores = compute_per_token_accuracy(batch_full, full_model, tokenizer)

            # Calculate per-token accuracy for retain model (OUT condition)
            retain_model.eval()
            # Move batch to retain_model's device
            batch_retain = {k: v.to(retain_model.device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch_dict.items()}
            acc_out_scores = compute_per_token_accuracy(batch_retain, retain_model, tokenizer)

        return acc_in_scores, acc_out_scores
