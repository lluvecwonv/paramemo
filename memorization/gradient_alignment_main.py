#!/usr/bin/env python3

import os
import sys
import json
import torch
import hydra
import logging
import warnings
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_curve, auc

# 현재 스크립트 디렉토리 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from utils import get_model_identifiers_from_yaml
from memorization.gradient.gradient_alignment_analyzer import compute_alignment_batch

# 경고 억제
warnings.filterwarnings("ignore", category=UserWarning)

# CUDA 메모리 최적화
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_paraphrase_results(json_path):
    """패러프레이즈 결과 JSON 파일 로드

    Supports two formats:
    1. Old format: {'results': [{'Question': ..., 'ParaphraseResults': [...]}]}
    2. New format: {'GeneratedParaphrases': [{'original_text': ..., 'paraphrases': [...]}]}
    """
    logger.info(f"Loading paraphrase results from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_texts = []
    paraphrase_texts_list = []
    original_memorization_scores = []

    # Check which format
    if 'GeneratedParaphrases' in data:
        # New format (pile_samples)
        logger.info("Detected new format (pile_samples)")
        for item in data['GeneratedParaphrases']:
            original_text = item.get('original_text', '')
            if not original_text:
                continue

            original_texts.append(original_text)
            original_memorization_scores.append(None)  # No memorization scores in new format

            # Get paraphrases
            paraphrases = item.get('paraphrases', [])
            # Filter out empty or identical paraphrases
            valid_paraphrases = [p for p in paraphrases if p and p.strip() != original_text.strip()]

            paraphrase_texts_list.append(valid_paraphrases if valid_paraphrases else [original_text])

    elif 'results' in data:
        # Old format (TOFU)
        logger.info("Detected old format (TOFU)")
        for item in data['results']:
            original_question = item.get('Question', '')
            if not original_question:
                continue

            original_texts.append(original_question)
            original_memorization_scores.append(item.get('OriginalMemorization'))

            # 패러프레이즈 수집
            paraphrases = []
            for para_result in item.get('ParaphraseResults', []):
                paraphrased_question = para_result.get('paraphrased_question', '')
                if paraphrased_question and paraphrased_question != original_question:
                    paraphrases.append(paraphrased_question)

            # 패러프레이즈가 없으면 원본 사용
            paraphrase_texts_list.append(paraphrases if paraphrases else [original_question])
    else:
        raise ValueError(f"Unknown format in {json_path}. Expected 'GeneratedParaphrases' or 'results' key.")

    logger.info(f"✅ Loaded {len(original_texts)} texts, {sum(len(p) for p in paraphrase_texts_list)} total paraphrases")
    return original_texts, paraphrase_texts_list, original_memorization_scores


def compute_auc_metrics(train_scores, test_scores):
    """Compute AUC ROC for membership inference

    Higher gradient alignment = more likely to be member (train)

    Args:
        train_scores: list of alignment scores for training samples
        test_scores: list of alignment scores for test samples

    Returns:
        dict with AUROC, FPR@95, TPR@5
    """
    # Combine scores and labels
    all_scores = np.concatenate([train_scores, test_scores])
    all_labels = np.concatenate([
        np.ones(len(train_scores)),   # train = member = 1
        np.zeros(len(test_scores))     # test = nonmember = 0
    ])

    # Compute ROC curve
    fpr_list, tpr_list, thresholds = roc_curve(all_labels, all_scores)
    auroc = auc(fpr_list, tpr_list)

    # FPR@95: False Positive Rate when TPR = 95%
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]] if np.any(tpr_list >= 0.95) else 1.0

    # TPR@5: True Positive Rate when FPR = 5%
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]] if np.any(fpr_list <= 0.05) else 0.0

    return {
        'auroc': auroc,
        'fpr95': fpr95,
        'tpr05': tpr05
    }


def print_analysis_report(scores):
    """분석 결과 출력"""
    mean_score = np.mean(scores) if scores else 0.0
    std_score = np.std(scores) if scores else 0.0

    logger.info("=" * 60)
    logger.info("📊 GRADIENT ALIGNMENT ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total processed samples: {len(scores)}")
    logger.info(f"Mean Cosine Alignment: {mean_score:.6f} ± {std_score:.6f}")

    # 암기 전이 평가
    if mean_score > 0.7:
        logger.info("🔥 High gradient alignment - Strong memorization transfer")
    elif mean_score > 0.3:
        logger.info("⚠️  Moderate gradient alignment - Partial memorization transfer")
    else:
        logger.info("✅ Low gradient alignment - Weak memorization transfer")

    # 분포 분석
    high_align = sum(1 for score in scores if score > 0.7)
    med_align = sum(1 for score in scores if 0.3 <= score <= 0.7)
    low_align = sum(1 for score in scores if score < 0.3)
    total = len(scores)

    if total > 0:
        logger.info(f"Distribution: High({high_align/total*100:.1f}%) | Med({med_align/total*100:.1f}%) | Low({low_align/total*100:.1f}%)")
    logger.info("=" * 60)


def save_results(output_file, cfg, scores, original_texts, paraphrase_texts_list, paraphrase_json_path):
    """결과를 JSON 파일로 저장"""
    logger.info("💾 Saving results...")

    mean_score = np.mean(scores) if scores else 0.0
    std_score = np.std(scores) if scores else 0.0

    # 분포 분석
    high_align = sum(1 for score in scores if score > 0.7)
    med_align = sum(1 for score in scores if 0.3 <= score <= 0.7)
    low_align = sum(1 for score in scores if score < 0.3)
    total = len(scores) if scores else 1

    results_dict = {
        'config': {
            'model_family': cfg.model_family,
            'model_path': cfg.model_path,
            'total_questions': len(original_texts),
            'total_samples': len(scores),
            'analysis_type': 'gradient_alignment',
            'paraphrase_source': paraphrase_json_path
        },
        'overall_statistics': {
            'num_samples': len(scores),
            'mean_cosine_alignment': float(mean_score),
            'std_cosine_alignment': float(std_score),
        },
        'individual_results': [
            {
                'sample_index': i,
                'original_text': orig[:100] + "..." if len(orig) > 100 else orig,
                'num_paraphrases': len(paras),
                'cosine_alignment': float(score)
            }
            for i, (orig, paras, score) in enumerate(zip(original_texts, paraphrase_texts_list, scores))
        ],
        'distribution_analysis': {
            'high_alignment_count': high_align,
            'medium_alignment_count': med_align,
            'low_alignment_count': low_align,
            'high_alignment_percent': high_align/total*100,
            'medium_alignment_percent': med_align/total*100,
            'low_alignment_percent': low_align/total*100
        },
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Results saved to: {output_file}")


def process_single_pair(train_file, test_file, model, tokenizer, device, cfg, output_dir):
    """단일 train/test 쌍에 대해 gradient alignment 계산"""

    logger.info(f"\n{'='*70}")
    logger.info(f"📊 Processing pair:")
    logger.info(f"   Train: {train_file}")
    logger.info(f"   Test:  {test_file}")
    logger.info(f"{'='*70}\n")

    # Train 데이터 로드
    logger.info("Loading TRAIN data...")
    train_texts, train_paraphrases, _ = load_paraphrase_results(train_file)

    # Test 데이터 로드
    logger.info("Loading TEST data...")
    test_texts, test_paraphrases, _ = load_paraphrase_results(test_file)

    # Gradient Alignment 계산 (train과 test 독립적으로 계산)
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Computing gradient alignments SEPARATELY")
    logger.info(f"{'='*70}")

    # 1. Train 데이터 gradient alignment 계산
    logger.info(f"\n📊 [1/2] Computing TRAIN gradient alignments ({len(train_texts)} samples)...")
    train_scores = compute_alignment_batch(
        model=model,
        tokenizer=tokenizer,
        original_texts=train_texts,
        paraphrase_lists=train_paraphrases,
        max_len=cfg.max_length,
        device=str(device)
    )
    logger.info(f"✅ Train scores computed: {len(train_scores)} samples")
    logger.info(f"   Mean: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")

    # 2. Test 데이터 gradient alignment 계산
    logger.info(f"\n📊 [2/2] Computing TEST gradient alignments ({len(test_texts)} samples)...")
    test_scores = compute_alignment_batch(
        model=model,
        tokenizer=tokenizer,
        original_texts=test_texts,
        paraphrase_lists=test_paraphrases,
        max_len=cfg.max_length,
        device=str(device)
    )
    logger.info(f"✅ Test scores computed: {len(test_scores)} samples")
    logger.info(f"   Mean: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

    logger.info(f"\n✅ Successfully processed {len(train_scores) + len(test_scores)} samples total")
    logger.info(f"   - Train scores: {len(train_scores)}")
    logger.info(f"   - Test scores:  {len(test_scores)}")

    # Compute AUC metrics for MIA
    logger.info("📈 Computing AUC ROC metrics...")
    auc_metrics = compute_auc_metrics(train_scores, test_scores)

    # 출력 파일명 생성
    base_name = os.path.basename(train_file).replace('_train.json', '')
    output_file = os.path.join(output_dir, f"{base_name}_gradient_alignment.json")

    # 결과 저장 (AUC 메트릭 포함)
    results_dict = {
        'mia_metrics': {
            'auroc': float(auc_metrics['auroc']),
            'fpr95': float(auc_metrics['fpr95']),
            'tpr05': float(auc_metrics['tpr05']),
        },
        'train_results': [
            {
                'sample_index': i,
                'original_text': orig,
                'num_paraphrases': len(paras),
                'cosine_alignment': float(score)
            }
            for i, (orig, paras, score) in enumerate(zip(train_texts, train_paraphrases, train_scores))
        ],
        'test_results': [
            {
                'sample_index': i,
                'original_text': orig,
                'num_paraphrases': len(paras),
                'cosine_alignment': float(score)
            }
            for i, (orig, paras, score) in enumerate(zip(test_texts, test_paraphrases, test_scores))
        ],
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to: {output_file}")

    # 결과 출력
    train_mean = np.mean(train_scores) if train_scores else 0.0
    train_std = np.std(train_scores) if train_scores else 0.0
    test_mean = np.mean(test_scores) if test_scores else 0.0
    test_std = np.std(test_scores) if test_scores else 0.0

    logger.info("\n" + "="*70)
    logger.info("📊 GRADIENT ALIGNMENT RESULTS")
    logger.info("="*70)
    logger.info(f"TRAIN - Mean: {train_mean:.6f} ± {train_std:.6f}")
    logger.info(f"TEST  - Mean: {test_mean:.6f} ± {test_std:.6f}")
    logger.info("-"*70)
    logger.info("🎯 MIA METRICS (Membership Inference Attack)")
    logger.info("-"*70)
    logger.info(f"AUROC:       {auc_metrics['auroc']:.1%}")
    logger.info(f"FPR@95:      {auc_metrics['fpr95']:.1%}")
    logger.info(f"TPR@5:       {auc_metrics['tpr05']:.1%}")
    logger.info("="*70 + "\n")

    return output_file


@hydra.main(version_base=None, config_path="../config", config_name="gradient_alignment")
def main(cfg: DictConfig):
    """그라디언트 정렬도 분석 메인 함수"""

    from utils import find_train_test_pairs

    # Local rank 설정 (분산 학습 지원)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    logger.info("🔥 Starting Gradient Alignment Analysis")
    logger.info(f"🔧 Model family: {cfg.model_family}")
    logger.info(f"🔧 Model path: {cfg.model_path}")

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
        logger.info(f"🔧 Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        if world_size > 1:
            logger.info(f"🔧 Distributed mode: Rank {local_rank}/{world_size}")
    else:
        device = torch.device('cpu')
        logger.info("🔧 Using CPU")

    try:
        # 모델과 토크나이저 로드
        logger.info("📥 Loading model and tokenizer...")
        model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
        model_id = model_cfg["hf_key"]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model from: {cfg.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        model.eval()
        logger.info("✅ Model loaded successfully")

        # Train/Test 쌍 찾기 또는 단일 파일 처리
        if hasattr(cfg, 'results_dir') and cfg.results_dir:
            # 배치 모드: results_dir에서 모든 train/test 쌍 찾기
            logger.info(f"\n🔍 Batch mode: Finding train/test pairs in {cfg.results_dir}")
            pairs = find_train_test_pairs(cfg.results_dir)

            if not pairs:
                logger.error("❌ No train/test pairs found!")
                return

            logger.info(f"✅ Found {len(pairs)} train/test pair(s)")

            # 출력 디렉토리 설정
            output_dir = cfg.get('output_dir', './gradient_alignment_results')
            os.makedirs(output_dir, exist_ok=True)

            # 각 쌍에 대해 처리
            success_count = 0
            for base_name, files in sorted(pairs.items()):
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing: {base_name}")
                logger.info(f"{'='*70}")

                try:
                    process_single_pair(
                        train_file=files['train'],
                        test_file=files['test'],
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        cfg=cfg,
                        output_dir=output_dir
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to process {base_name}: {e}")
                    import traceback
                    traceback.print_exc()

            logger.info(f"\n{'='*70}")
            logger.info(f"🎯 Batch processing completed!")
            logger.info(f"✅ Successfully processed: {success_count}/{len(pairs)}")
            logger.info(f"📁 Results saved to: {output_dir}")
            logger.info(f"{'='*70}\n")

        else:
            # 단일 파일 모드 (기존 방식)
            paraphrase_json_path = cfg.paraphrase_results_file
            if not os.path.exists(paraphrase_json_path):
                raise FileNotFoundError(f"Paraphrase results file not found: {paraphrase_json_path}")

            original_texts, paraphrase_texts_list, original_memorization_scores = \
                load_paraphrase_results(paraphrase_json_path)

            # Gradient Alignment 계산
            logger.info("🚀 Computing gradient alignments...")

            scores = compute_alignment_batch(
                model=model,
                tokenizer=tokenizer,
                original_texts=original_texts,
                paraphrase_lists=paraphrase_texts_list,
                max_len=cfg.max_length,
                device=str(device)
            )

            logger.info(f"✅ Successfully processed {len(scores)} samples")

            # 결과 출력
            print_analysis_report(scores)

            # 결과 저장
            save_results(
                output_file=cfg.output_file,
                cfg=cfg,
                scores=scores,
                original_texts=original_texts,
                paraphrase_texts_list=paraphrase_texts_list,
                paraphrase_json_path=paraphrase_json_path
            )

            logger.info("🎯 Gradient Alignment Analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
