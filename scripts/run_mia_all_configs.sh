#!/bin/bash

# MIA Analysis Runner for All Local Domains
# Usage: ./run_mia_all_configs.sh [model_family]
# Example: ./run_mia_all_configs.sh pythia-2.8b
# Example: ./run_mia_all_configs.sh pythia-410m

# Get parameters from command line
model_family=${1:-pythia-160m}

# Auto-detect model path
case $model_family in
    pythia-2.8b)
        model_path="EleutherAI/pythia-2.8b"
        ;;
    pythia-1.4b)
        model_path="EleutherAI/pythia-1.4b"
        ;;
    pythia-410m)
        model_path="EleutherAI/pythia-410m"
        ;;
    pythia-160m)
        model_path="EleutherAI/pythia-160m"
        ;;
    pythia-70m)
        model_path="EleutherAI/pythia-70m"
        ;;
    *)
        echo "❌ Error: Unknown model family: $model_family"
        echo "Available models: pythia-2.8b, pythia-1.4b, pythia-410m, pythia-160m, pythia-70m"
        exit 1
        ;;
esac

# All local domains to iterate over
local_domains=(
    "arxiv"
    "dm_mathematics"
    "github"
    "hackernews"
    "pile_cc"
    "pubmed_central"
    "wikipedia_en"
)

echo "=========================================="
echo "MIA Analysis - All Local Domains"
echo "=========================================="
echo "Model: $model_family ($model_path)"
echo "Number of domains: ${#local_domains[@]}"
echo "=========================================="
echo ""

# Change to parent directory
cd "$(dirname "$0")/.."

# Counter for tracking progress
total=${#local_domains[@]}
current=0
failed=0
succeeded=0

# Loop through all domains
for domain in "${local_domains[@]}"; do
    current=$((current + 1))

    echo ""
    echo "[$current/$total] Processing domain: $domain"
    echo "----------------------------------------"

    # Check if domain directory exists
    if [ ! -d "./pile_samples/$domain" ]; then
        echo "⚠️  Domain directory not found: ./pile_samples/$domain"
        echo "Skipping..."
        failed=$((failed + 1))
        echo "----------------------------------------"
        continue
    fi

    # Run MIA analysis with local files
    CUDA_VISIBLE_DEVICES=1 python mia/run_hydra.py \
        model_family=$model_family \
        model_path="$model_path" \
        dataset_config=$domain \
        use_local_files=true \
        pile_samples_dir=./pile_samples \
        half=false \
        int8=false

    if [ $? -eq 0 ]; then
        echo "✅ Domain $domain completed successfully"
        succeeded=$((succeeded + 1))
    else
        echo "❌ Domain $domain failed"
        failed=$((failed + 1))
    fi

    echo "----------------------------------------"
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total domains: $total"
echo "Succeeded: $succeeded"
echo "Failed: $failed"
echo "=========================================="

if [ $failed -eq 0 ]; then
    echo ""
    echo "✅ All domains completed successfully!"
    echo "Results saved to: results/"
else
    echo ""
    echo "⚠️  Some domains failed. Check logs above."
    exit 1
fi
