#!/bin/bash

# Pile Data Sampling Script (MIMIR style)
# Usage: ./scripts/sample_pile_data.sh [output_dir]

# 설정
output_dir=${1:-./pile_samples}

# 환경 설정
# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 디렉토리 이동
cd "$PROJECT_ROOT"

echo "=========================================="
echo "PILE DATA SAMPLING (MIMIR style)"
echo "=========================================="
echo "Output directory: $output_dir"
echo "Sampling scheme:"
echo "  - Per domain: 1,000 members + 1,000 non-members"
echo "  - Aggregate: 10,000 members + 10,000 non-members"
echo "  - Word range: 100-200 words per sample"
echo "  - Members from train split, non-members from validation split"
echo "=========================================="
echo ""

# 실행
python datas/sample_pile_data.py \
    sample_output_dir="$output_dir"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Pile data sampling completed successfully!"
    echo "📁 Results saved to: $output_dir"
else
    echo ""
    echo "❌ Pile data sampling failed!"
    exit 1
fi
