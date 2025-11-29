import sys
import os
import json
import datasets
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv()

# MIMIR 데이터셋의 7개 도메인 (MIMIR 논문과 동일)
MIMIR_DOMAINS = [
    "pile_cc",           # General web
    "wikipedia_(en)",    # Knowledge
    "pubmed_central",    # Academic papers
    "arxiv",             # Academic papers
    "hackernews",        # Dialogues
    "dm_mathematics",    # Specialized domains
    "github"             # Specialized domains
]


def write_jsonl(data, file_path):
    """JSONL 형식으로 데이터 저장 (MIMIR 방식)"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"💾 Saved {len(data)} samples to: {file_path}")


def load_mimir_domain(domain_name, split=None, num_samples=None):
    """
    MIMIR 데이터셋에서 특정 도메인 로드

    Args:
        domain_name: MIMIR domain name (e.g., 'arxiv', 'wikipedia_(en)')
        split: Split name (ngram_7_0.2, ngram_13_0.2, ngram_13_0.8)
        num_samples: Number of samples to load (None = all)

    Returns:
        Dataset with 'member' and 'nonmember' fields
    """
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN not found in environment. Please set it in .env file")

    print(f"\nLoading MIMIR dataset: domain='{domain_name}', split='{split}'")

    ds = datasets.load_dataset(
        'iamgroot42/mimir',
        domain_name,
        split=split,
        token=token,
        trust_remote_code=True
    )

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    print(f"✅ Loaded {len(ds)} samples from MIMIR")
    return ds


def create_mimir_dataset(domain_name, output_dir, split='none', num_samples=None):
    """
    MIMIR 데이터셋에서 train/test JSONL 파일 생성

    Args:
        domain_name: MIMIR domain name
        output_dir: Output directory
        split: MIMIR split name
        num_samples: Number of samples to use (None = all)

    Output:
        domain_name/train_text.jsonl: Member samples (models trained on)
        domain_name/test_text.jsonl: Nonmember samples (models never seen)
    """
    print(f"\n{'='*60}")
    print(f"Creating dataset: {domain_name}")
    print(f"  Split: {split}")
    if num_samples:
        print(f"  Samples: {num_samples}")
    print(f"{'='*60}")

    # MIMIR 데이터셋 로드
    ds = load_mimir_domain(domain_name, split=split, num_samples=num_samples)

    # Member와 Nonmember 추출
    members = []
    nonmembers = []

    for sample in tqdm(ds, desc=f"Processing {domain_name}"):
        members.append({'text': sample['member']})
        nonmembers.append({'text': sample['nonmember']})

    print(f"✅ Extracted {len(members)} members, {len(nonmembers)} nonmembers")

    # 도메인 이름을 파일명으로 변환
    # wikipedia_(en) -> wikipedia_en
    safe_domain = domain_name.replace('(', '').replace(')', '')
    domain_dir = os.path.join(output_dir, safe_domain)

    # JSONL 형식으로 저장
    write_jsonl(members, os.path.join(domain_dir, 'train_text.jsonl'))
    write_jsonl(nonmembers, os.path.join(domain_dir, 'test_text.jsonl'))

    return {
        'domain': domain_name,
        'train_samples': len(members),
        'test_samples': len(nonmembers)
    }


@hydra.main(version_base=None, config_path="../config", config_name="pile_sampling")
def main(cfg: DictConfig):
    """MIMIR 데이터 샘플링 메인 함수"""

    output_dir = cfg.sample_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # MIMIR split 선택 (기본값: ngram_7_0.2)
    mimir_split = getattr(cfg, 'mimir_split', 'none')

    # 샘플 수 제한 (None = 전체 사용)
    num_samples = getattr(cfg, 'num_samples_per_domain', None)

    print(f"\n🚀 Starting MIMIR Data Sampling")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Using iamgroot42/mimir dataset")
    print(f"🔢 Split: {mimir_split}")
    if num_samples:
        print(f"📝 Samples per domain: {num_samples}")
    else:
        print(f"📝 Samples per domain: ALL (약 1000개)")
    print(f"✅ Member (train_text.jsonl) - 모델이 학습한 데이터")
    print(f"✅ Nonmember (test_text.jsonl) - 모델이 보지 못한 데이터")

    results = []

    # 각 도메인별로 샘플링
    for domain in MIMIR_DOMAINS:
        try:
            result = create_mimir_dataset(
                domain_name=domain,
                output_dir=output_dir,
                split=mimir_split,
                num_samples=num_samples
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to create dataset for {domain}: {e}")
            results.append({
                'domain': domain,
                'train_samples': 0,
                'test_samples': 0,
                'error': str(e)
            })

    # 결과 요약 저장
    summary_file = os.path.join(output_dir, 'sampling_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'config': {
                'dataset': 'iamgroot42/mimir',
                'split': mimir_split,
                'num_samples_per_domain': num_samples if num_samples else 'all',
                'description': 'MIMIR dataset with member/nonmember split, no custom ngram filtering needed'
            },
            'results': results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("✅ Sampling Complete!")
    print(f"📊 Summary saved to: {summary_file}")
    print(f"{'='*60}")

    # 결과 출력
    print("\nSampling Results:")
    for r in results:
        if 'error' in r:
            print(f"  ❌ {r['domain']}: ERROR - {r['error']}")
        else:
            print(f"  ✅ {r['domain']}: train={r['train_samples']}, test={r['test_samples']}")


if __name__ == "__main__":
    main()
