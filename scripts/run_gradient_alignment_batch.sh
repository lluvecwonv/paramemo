#!/bin/bash

# Gradient Alignment Batch Processing Script
# Automatically finds train/test pairs and computes gradient alignment
#
# Usage:
#   ./scripts/run_gradient_alignment_batch.sh [model_family] [results_dir] [output_dir]
#
# Examples:
#   ./scripts/run_gradient_alignment_batch.sh pythia-1.4b
#   ./scripts/run_gradient_alignment_batch.sh pythia-2.8b ./results ./grad_results

set -e  # Exit on error

# Parameters
model_family=${1:-pythia-160m}
results_dir=${2:-./results}
output_dir=${3:-./gradient_alignment_results}

# MIA domains (automatically filters by these)
domains=(
    "dm_mathematics"
    "github"
    "hackernews"
    "pile_cc"
    "pubmed_central"
    "wikipedia_en"
)

# Environment setup
export PYTHONPATH=/root/paramem:$PYTHONPATH

# Change to project root
cd /root/paramem

echo "=========================================="
echo "GRADIENT ALIGNMENT BATCH PROCESSING"
echo "=========================================="
echo "Model family: $model_family"
echo "Results directory: $results_dir"
echo "Output directory: $output_dir"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Filtering domains:"
for domain in "${domains[@]}"; do
    echo "  - $domain"
done
echo "=========================================="
echo ""
master_port=29503

# Run gradient alignment with torchrun (single GPU)
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 --master_port=$master_port \
    memorization/gradient_alignment_main.py \
    model_family=$model_family \
    results_dir=$results_dir \
    output_dir=$output_dir

echo ""
echo "=========================================="
echo "✅ Gradient alignment completed!"
echo "Results saved to: $output_dir"
echo "=========================================="
