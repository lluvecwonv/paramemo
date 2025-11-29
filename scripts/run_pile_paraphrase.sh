#!/bin/bash

# Pile Paraphrase Analysis Runner Script (Using pile_samples JSONL files)
# Usage: ./scripts/run_pile_paraphrase.sh [model_family] [num_samples] [batch_size]
# Example: ./scripts/run_pile_paraphrase.sh pythia-2.8b 100 8
# Example: ./scripts/run_pile_paraphrase.sh pythia-2.8b null 4
# Example: ./scripts/run_pile_paraphrase.sh pythia-2.8b   (uses all samples, batch_size=1)

# 설정
model_family=${1:-pythia-2.8b}
num_samples=${2:-null}
batch_size=${3:-2}

# 환경 설정b
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 디렉토리 이동
cd "$PROJECT_ROOT"

echo "=========================================="
echo "PILE PARAPHRASE ANALYSIS"
echo "=========================================="
echo "Data source: pile_samples/ (JSONL files)"
echo "Model family: $model_family"
if [ "$num_samples" = "null" ]; then
    echo "Num samples: ALL"
else
    echo "Num samples: $num_samples per file"
fi
echo "Batch size: $batch_size"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# pile_samples 폴더 확인
if [ ! -d "./pile_samples" ]; then
    echo "❌ pile_samples directory not found!"
    exit 1
fi

# 도메인 폴더 확인
domains=$(ls -d pile_samples/*/ 2>/dev/null | wc -l)
if [ "$domains" -eq 0 ]; then
    echo "❌ No domain folders found in pile_samples/"
    exit 1
fi

echo "Found $domains domains in pile_samples/"
echo ""
master_port=29500

# 실행
if [ "$num_samples" = "null" ]; then
    torchrun --nproc_per_node=2 --master_port=$master_port \
        memorization/paraphrase_main.py \
        --config-name=pile_paraphrase_analysis \
        model_family=$model_family \
        pile_samples_dir=./pile_samples \
        analysis.generation_mode=beam_single_prompt \
        analysis.num_paraphrases=3 \
        analysis.num_beams=3 \
        analysis.batch_size=$batch_size \
        analysis.paraphrase_temperature=0.25 \
        analysis.top_p=0.9 \
        analysis.top_k=40 \
        analysis.repetition_penalty=1.1 \
        analysis.use_soft_beam_sampling=true
else
    torchrun --nproc_per_node=2 --master_port=$master_port \
        memorization/paraphrase_main.py \
        --config-name=pile_paraphrase_analysis \
        model_family=$model_family \
        num_samples=$num_samples \
        pile_samples_dir=./pile_samples \
        analysis.generation_mode=beam_single_prompt \
        analysis.num_paraphrases=3 \
        analysis.num_beams=3 \
        analysis.batch_size=$batch_size \
        analysis.paraphrase_temperature=0.25 \
        analysis.top_p=0.9 \
        analysis.top_k=40 \
        analysis.repetition_penalty=1.1 \
        analysis.use_soft_beam_sampling=true
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Paraphrase analysis completed successfully!"
    echo ""
    echo "Results saved in: results/"
    echo "Output format: {model}_{mode}_{domain}_{train|test}.json"
else
    echo ""
    echo "❌ Paraphrase analysis failed!"
    exit 1
fi
