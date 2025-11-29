import sys
import os
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import datasets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from memorization.analysis.paraphrase_generator import ParaphraseGenerator
from memorization.utils import get_model_config, check_paraphrase_completed

# Pile 서브셋들 (MIMIR 데이터셋의 도메인들)
PILE_SUBSETS = [
    "arxiv",
    "dm_mathematics",
    "github",
    "hackernews",
    "pile_cc",
    "pubmed_central",
    "wikipedia_(en)"
]

def load_jsonl_texts(jsonl_path, num_samples=None):
    """Load texts from JSONL file

    Args:
        jsonl_path: Path to JSONL file
        num_samples: Maximum number of samples to load (None = all)

    Returns:
        list: List of texts
    """
    texts = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])
                if num_samples and len(texts) >= num_samples:
                    break
        return texts
    except Exception as e:
        print(f"❌ Failed to load {jsonl_path}: {e}")
        return []


def load_pile_samples_data(pile_samples_dir, domain_name, split_type='train', num_samples=None):
    """Load data from pile_samples directory

    Args:
        pile_samples_dir: Path to pile_samples directory
        domain_name: Domain name (folder name)
        split_type: 'train' or 'test'
        num_samples: Maximum number of samples to load (None = all)

    Returns:
        list: List of texts
    """
    jsonl_file = f"{split_type}_text.jsonl"
    jsonl_path = os.path.join(pile_samples_dir, domain_name, jsonl_file)

    if not os.path.exists(jsonl_path):
        print(f"❌ File not found: {jsonl_path}")
        return []

    print(f"Loading {jsonl_path}...")
    texts = load_jsonl_texts(jsonl_path, num_samples)
    print(f"✅ Loaded {len(texts)} texts from {domain_name}/{split_type}_text.jsonl")
    return texts


@hydra.main(version_base=None, config_path="../config", config_name="pile_paraphrase_analysis")
def main(cfg: DictConfig):
    """Pile 서브셋별 패러프레이즈 분석 (pile_samples JSONL 파일 사용)"""

    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device('cpu')

    print(f"Process {local_rank} using device: {device}")

    generation_mode = getattr(cfg.analysis, 'generation_mode', 'greedy_Nprompts')
    num_samples = cfg.get('num_samples', None)  # None = use all
    pile_samples_dir = cfg.get('pile_samples_dir', './pile_samples')

    # Load model configuration and get HuggingFace model path
    model_config = get_model_config(cfg.model_family)
    model_path = model_config['hf_key']

    print(f"\n{'='*60}")
    print(f"Paraphrase Analysis Configuration")
    print(f"{'='*60}")
    print(f"Model: {cfg.model_family} ({model_path})")
    print(f"Generation mode: {generation_mode}")
    print(f"Pile samples dir: {pile_samples_dir}")
    print(f"Num samples per file: {num_samples if num_samples else 'ALL'}")
    print(f"Num paraphrases per text: {cfg.analysis.num_paraphrases}")
    print(f"{'='*60}\n")

    # Load model
    print(f"[Rank {local_rank}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model in FP16 to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
    model = model.to(device)
    print(f"[Rank {local_rank}] ✅ Model loaded on {device}\n")

    generator = ParaphraseGenerator(cfg)

    # pile_samples 디렉토리에서 각 도메인 폴더 찾기
    if not os.path.exists(pile_samples_dir):
        print(f"❌ Pile samples directory not found: {pile_samples_dir}")
        return

    domain_folders = [d for d in os.listdir(pile_samples_dir)
                     if os.path.isdir(os.path.join(pile_samples_dir, d))]

    print(f"Found {len(domain_folders)} domains: {domain_folders}\n")

    # Distribute domains across GPUs
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    my_domains = [d for i, d in enumerate(domain_folders) if i % world_size == local_rank]

    print(f"[Rank {local_rank}] Processing {len(my_domains)} domains: {my_domains}\n")

    # 각 도메인별로 train과 test 모두 처리
    skipped_count = 0
    processed_count = 0

    for domain_name in my_domains:
        for split_type in ['train', 'test']:
            print(f"\n{'='*60}")
            print(f"Processing: {domain_name} / {split_type}")
            print(f"{'='*60}")

            # Check if already completed
            is_completed, output_file = check_paraphrase_completed(
                cfg.analysis.output_dir,
                cfg.model_family,
                generation_mode,
                domain_name,
                split_type
            )

            if is_completed:
                print(f"⏭️  SKIPPED: {domain_name}/{split_type} - Already exists at {output_file}")
                skipped_count += 1
                continue

            # pile_samples에서 데이터 로드
            texts = load_pile_samples_data(
                pile_samples_dir,
                domain_name,
                split_type=split_type,
                num_samples=num_samples
            )

            if not texts:
                print(f"❌ No data loaded for {domain_name}/{split_type}, skipping...")
                continue

            processed_count += 1

            print(f"✅ Loaded {len(texts)} texts from {domain_name}/{split_type}")

            # GPU 메모리 정리 (다른 프로세스가 메모리 많이 사용 중)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                print(f"🧹 GPU memory cleared")

            # 패러프레이즈 생성 (배치 처리 - GPU 병렬 생성)
            all_paraphrases = []
            failed_count = 0
            batch_size = getattr(cfg.analysis, 'batch_size', 1)  # 배치 크기 (메모리 제약 고려)

            # 배치 단위로 병렬 처리
            for batch_start in tqdm(range(0, len(texts), batch_size),
                                   desc=f"Paraphrasing {domain_name}/{split_type}",
                                   unit="batch"):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]

                try:
                    # 배치 전체를 한 번에 GPU에서 병렬 처리
                    batch_paraphrases = generator.generate_paraphrases_batch(
                        batch_texts, model, tokenizer
                    )

                    # 결과 처리
                    for i, paraphrases in enumerate(batch_paraphrases):
                        # Extract text from paraphrase dicts
                        paraphrase_texts = [p['text'] for p in paraphrases][:cfg.analysis.num_paraphrases]

                        if not paraphrase_texts or len(paraphrase_texts) < cfg.analysis.num_paraphrases:
                            failed_count += 1
                            # Fallback: use original text to fill missing
                            while len(paraphrase_texts) < cfg.analysis.num_paraphrases:
                                paraphrase_texts.append(batch_texts[i])

                        all_paraphrases.append(paraphrase_texts)

                except Exception as e:
                    print(f"⚠️  Batch generation failed: {e}, using fallback")
                    # Fallback: sequential processing for this batch
                    for text in batch_texts:
                        try:
                            paraphrases = generator._generate_paraphrases(
                                text, model, tokenizer, input_type='text', question=""
                            )
                            paraphrase_texts = [p['text'] for p in paraphrases][:cfg.analysis.num_paraphrases]

                            if not paraphrase_texts:
                                failed_count += 1
                                paraphrase_texts = [text] * cfg.analysis.num_paraphrases

                            all_paraphrases.append(paraphrase_texts)
                        except Exception as e2:
                            failed_count += 1
                            all_paraphrases.append([text] * cfg.analysis.num_paraphrases)

                # 배치 처리 후 GPU 메모리 정리 (메모리 단편화 방지)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 결과 저장
            os.makedirs(cfg.analysis.output_dir, exist_ok=True)

            output_file = os.path.join(
                cfg.analysis.output_dir,
                f"{cfg.model_family}_{generation_mode}_{domain_name}_{split_type}.json"
            )

            # Format data with original_text paired with its paraphrases
            generated_paraphrases = []
            for i, (original_text, paraphrases) in enumerate(zip(texts, all_paraphrases)):
                generated_paraphrases.append({
                    "original_text": original_text,
                    "paraphrases": paraphrases
                })

            output_data = {
                "domain_name": domain_name,
                "split_type": split_type,
                "data_source": f"pile_samples/{domain_name}/{split_type}_text.jsonl",
                "NumParaphrasesGenerated": cfg.analysis.num_paraphrases,
                "GeneratedParaphrases": generated_paraphrases,
                "generation_mode": generation_mode,
                "failed_count": failed_count,
                "config": {
                    "model_family": cfg.model_family,
                    "model_path": model_path,
                    "temperature": getattr(cfg.analysis, 'paraphrase_temperature', None),
                    "top_p": getattr(cfg.analysis, 'top_p', None),
                    "num_samples": len(texts),
                    "domain_name": domain_name,
                    "split_type": split_type
                }
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"✅ Saved to {output_file}")
            print(f"📊 Generated {len(all_paraphrases)} paraphrases, {failed_count} failed")

    print(f"\n{'='*60}")
    print(f"[Rank {local_rank}] ✅ All domains processed!")
    print(f"{'='*60}")
    print(f"[Rank {local_rank}] 📊 Summary:")
    print(f"   - Processed: {processed_count}")
    print(f"   - Skipped (already exists): {skipped_count}")
    print(f"   - Total: {processed_count + skipped_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
