import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class ParaphraseVisualizer:
    """Handles visualization for paraphrase analysis results"""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set matplotlib style
        plt.style.use('default')
        logger.info(f"📊 ParaphraseVisualizer initialized with output_dir={output_dir}")

    def create_memorization_simplicity_plot(self, results: List[Dict[str, Any]], tag: str):
        """Create hexbin plot with marginal histograms for memorization vs simplicity

        Args:
            results: List of analysis results with memorization and simplicity scores
            tag: Tag for naming the output file
        """
        if not results:
            logger.warning(f"No results provided for memorization-simplicity plot ({tag})")
            return

        try:
            memorization_scores = np.array([r['memorization_score'] for r in results])
            simplicity_scores = np.array([r['simplicity_score'] for r in results])

            # Calculate correlations
            pearson_corr, _ = pearsonr(simplicity_scores, memorization_scores)
            spearman_corr, _ = spearmanr(simplicity_scores, memorization_scores)

            # Create figure with gridspec
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0.05, hspace=0.05)

            # Main hexbin plot
            ax_main = fig.add_subplot(gs[1:, :3])
            hb = ax_main.hexbin(
                simplicity_scores,
                memorization_scores,
                gridsize=30,
                cmap="PuBu",
                mincnt=1
            )
            ax_main.set_xlabel("Simplicity Score", fontsize=12)
            ax_main.set_ylabel("Memorization Score", fontsize=12)
            ax_main.grid(alpha=0.3)
            ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            # Right histogram (y-axis)
            ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)
            ax_right.hist(memorization_scores, bins=30, orientation="horizontal",
                         color="skyblue", edgecolor="k", alpha=0.7)
            ax_right.set_xlabel("Count")
            ax_right.grid(alpha=0.3)
            ax_right.tick_params(labelleft=False)

            # Top histogram (x-axis)
            ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
            ax_top.hist(simplicity_scores, bins=30, color="skyblue", edgecolor="k", alpha=0.7)
            ax_top.set_ylabel("Count")
            ax_top.grid(alpha=0.3)
            ax_top.tick_params(labelbottom=False)

            # Colorbar
            plt.colorbar(hb, ax=ax_main, label="Count", orientation="vertical")

            # Add figure title using suptitle (outside of subplots)
            fig.suptitle(
                f"Memorization vs Simplicity ({tag})\nPearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}",
                fontsize=14, fontweight="bold", y=0.98
            )

            plot_path = os.path.join(self.output_dir, f'memorization_simplicity_{tag}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Saved memorization-simplicity hexbin plot to {plot_path}")

        except Exception as e:
            logger.error(f"Failed to create memorization-simplicity plot for {tag}: {e}")

    def create_original_vs_paraphrase_simplicity_plot(self,
                                                      original_results: List[Dict[str, Any]],
                                                      paraphrase_results: List[Dict[str, Any]]):
        """Create hexbin plot comparing original and paraphrase simplicity scores

        Args:
            original_results: Results from original questions
            paraphrase_results: Results from paraphrased questions (may have multiple paraphrases per original)
        """
        if not original_results or not paraphrase_results:
            logger.warning("No results provided for original vs paraphrase simplicity plot")
            return

        try:
            # Handle case where there are multiple paraphrases per original question
            # Create pairs: each paraphrase paired with its corresponding original
            num_original = len(original_results)
            num_paraphrase = len(paraphrase_results)
            num_per_original = num_paraphrase // num_original

            logger.info(f"Creating plot with {num_paraphrase} points ({num_per_original} paraphrases per original)")

            original_simplicity = []
            paraphrase_simplicity = []

            for i, orig_result in enumerate(original_results):
                orig_simp = orig_result['simplicity_score']

                # Get all paraphrases for this original question
                start_idx = i * num_per_original
                end_idx = start_idx + num_per_original

                if end_idx > len(paraphrase_results):
                    break

                # Add a point for each paraphrase
                for j in range(start_idx, end_idx):
                    para_simp = paraphrase_results[j]['simplicity_score']
                    original_simplicity.append(orig_simp)
                    paraphrase_simplicity.append(para_simp)

            original_simplicity = np.array(original_simplicity)
            paraphrase_simplicity = np.array(paraphrase_simplicity)

            # Calculate correlations
            pearson_corr, _ = pearsonr(original_simplicity, paraphrase_simplicity)
            spearman_corr, _ = spearmanr(original_simplicity, paraphrase_simplicity)

            # Create figure with gridspec
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0.05, hspace=0.05)

            # Main hexbin plot
            ax_main = fig.add_subplot(gs[1:, :3])
            hb = ax_main.hexbin(
                original_simplicity,
                paraphrase_simplicity,
                gridsize=30,
                cmap="YlGn",
                mincnt=1
            )
            ax_main.set_xlabel("Original Simplicity Score", fontsize=12)
            ax_main.set_ylabel("Paraphrase Simplicity Score", fontsize=12)
            ax_main.grid(alpha=0.3)

            # Add diagonal reference line (y=x)
            min_val = min(original_simplicity.min(), paraphrase_simplicity.min())
            max_val = max(original_simplicity.max(), paraphrase_simplicity.max())
            ax_main.plot([min_val, max_val], [min_val, max_val],
                        'r--', alpha=0.7, linewidth=2, label='y=x')
            ax_main.legend()

            # Right histogram (paraphrase simplicity)
            ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)
            ax_right.hist(paraphrase_simplicity, bins=30, orientation="horizontal",
                         color="lightgreen", edgecolor="k", alpha=0.7)
            ax_right.set_xlabel("Count")
            ax_right.grid(alpha=0.3)
            ax_right.tick_params(labelleft=False)

            # Top histogram (original simplicity)
            ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
            ax_top.hist(original_simplicity, bins=30, color="lightgreen", edgecolor="k", alpha=0.7)
            ax_top.set_ylabel("Count")
            ax_top.grid(alpha=0.3)
            ax_top.tick_params(labelbottom=False)

            # Colorbar
            plt.colorbar(hb, ax=ax_main, label="Count", orientation="vertical")

            # Add figure title
            fig.suptitle(
                f"Original vs Paraphrase Simplicity\nPearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}",
                fontsize=14, fontweight="bold", y=0.98
            )

            plot_path = os.path.join(self.output_dir, 'simplicity_original_vs_paraphrase.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Saved original vs paraphrase simplicity plot to {plot_path}")

        except Exception as e:
            logger.error(f"Failed to create original vs paraphrase simplicity plot: {e}")


    def create_all_visualizations(self, original_results: List[Dict[str, Any]],
                                 paraphrase_results: List[Dict[str, Any]]):
        """Create all standard visualizations for paraphrase analysis

        Args:
            original_results: Results from original questions
            paraphrase_results: Results from paraphrased questions
        """
        logger.info("📊 Creating all visualizations...")

        # Individual plots
        self.create_memorization_simplicity_plot(original_results, "original")
        self.create_memorization_simplicity_plot(paraphrase_results, "paraphrase")

        # Comparison plot
        self.create_original_vs_paraphrase_simplicity_plot(original_results, paraphrase_results)

        logger.info("✅ All visualizations created successfully")
