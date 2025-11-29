#!/bin/bash

# Influence Function Analysis for All Domains and Methods
# Usage: ./scripts/run_influence_all.sh [model]
# Example: ./scripts/run_influence_all.sh pythia-2.8b
# Example: ./scripts/run_influence_all.sh pythia-410m

model=${1:-pythia-160m}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

domains=(
    "arxiv"
    "dm_mathematics"
    "github"
    "hackernews"
    "pile_cc"
    "pubmed_central"
    "wikipedia_en"
)

hvp_methods=(
    "gradient_match"
    "DataInf"
    "LiSSA"
    "repsim"
)

echo "=========================================="
echo "Influence Function Analysis - All Domains & Methods"
echo "=========================================="
echo "Model: $model"
echo "Domains: ${#domains[@]}"
echo "HVP Methods: ${#hvp_methods[@]}"
echo "Total runs: $((${#domains[@]} * ${#hvp_methods[@]}))"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

cd "$(dirname "$0")/../Influence-Functions"

total=$((${#domains[@]} * ${#hvp_methods[@]}))
current=0
succeeded=0
failed=0

for domain in "${domains[@]}"; do
    # Check if domain exists
    if [ ! -d "../pile_samples/$domain" ]; then
        echo "⚠️  Domain directory not found: ../pile_samples/$domain"
        echo "Skipping all methods for this domain..."
        failed=$((failed + ${#hvp_methods[@]}))
        continue
    fi

    for hvp_method in "${hvp_methods[@]}"; do
        current=$((current + 1))

        echo ""
        echo "[$current/$total] Processing: $domain + $hvp_method"
        echo "----------------------------------------"

        # Check if result already exists
        result_file="cache/${model}_${domain}_${hvp_method}.csv"
        if [ -f "$result_file" ]; then
            echo "⚠️  Result already exists: $result_file"
            echo "Skipping..."
            succeeded=$((succeeded + 1))
            echo "----------------------------------------"
            continue
        fi

        # Run influence analysis (single GPU to avoid OOM)
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29547  influence_pile.py \
            --model $model \
            --domain $domain \
            --data_dir ../pile_samples \
            --max_length 512 \
            --max_samples -1 \
            --hvp_method $hvp_method \
            --lambda_c 10 \
            --iter 3 \
            --alpha 1.0 \
            --grad_cache

        if [ $? -eq 0 ]; then
            echo "✅ $domain + $hvp_method completed"
            succeeded=$((succeeded + 1))
        else
            echo "❌ $domain + $hvp_method failed"
            failed=$((failed + 1))
        fi

        echo "----------------------------------------"
    done
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total runs: $total (${#domains[@]} domains × ${#hvp_methods[@]} methods)"
echo "Succeeded: $succeeded"
echo "Failed: $failed"
echo "=========================================="

if [ $failed -eq 0 ]; then
    echo ""
    echo "✅ All combinations completed!"
    echo "Results: cache/"
    echo ""
    echo "Generated files:"
    echo "  - cache/${model}_<domain>_<method>.csv (influence scores)"
    echo "  - cache/${model}_<domain>_<method>_metrics.txt (MIA metrics)"
else
    echo ""
    echo "⚠️  Some runs failed. Check logs above."
    exit 1
fi
