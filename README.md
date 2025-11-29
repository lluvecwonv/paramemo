# Paramem

Paramem은 KSC 2025에 채택된
「의역 문장 생성을 통한 거대 언어모델의 메모라이제이션 측정」
(Measuring Memorization in Large Language Models via Near-Duplicate Sample Generation)
논문의 공식 코드 저장소입니다.

본 레포지토리는 대규모 언어모델이 학습 데이터 일부를 얼마나 암기하고 있는지를 측정하기 위해,
의역(near-duplicate) 문장을 생성하여 모델의 반응을 비교하고,
원문–패러프레이즈 쌍의 그래디언트 정렬도를 분석함으로써 의미적 메모라이제이션을 정량화하는 Paramem 프레임워크를 제공합니다.

Paramem is the official codebase for the KSC 2025 paper
“Measuring Memorization in Large Language Models via Near-Duplicate Sample Generation”.

This repository provides the Paramem framework for quantifying memorization in large language models by
(1) generating near-duplicate paraphrases of training samples,
(2) comparing model outputs on originals vs. paraphrases, and
(3) analyzing gradient alignment between each pair to assess semantic memorization.

## 주요 기능

- **패러프레이즈 생성 파이프라인**: Pile 및 TOFU와 같은 데이터셋에서 원문을 불러와 다양한 디코딩 전략(beam, nucleus 등)으로 의역 문장을 생성한다.
- **Gradient Alignment 분석**: 원문과 패러프레이즈에 대한 출력층 그래디언트를 비교하여 코사인 유사도를 계산하고, 암기 전이 정도를 수치화한다.
- **Membership Inference Attack (MIA)**: train/test 세트를 나누어 정렬도 분포 차이를 기반으로 AUROC, FPR@95, TPR@5 등을 계산한다.

## 디렉터리 구조 요약

```
memorization/
├── analysis/                # 패러프레이즈 생성 및 분석
├── gradient/                # 그래디언트 정렬도 계산 로직
├── gradient_alignment_main.py
├── paraphrase_main.py
└── utils.py
mia/                         # Membership Inference 관련 스크립트
Influence-Functions/         # Influence function 계산 도구
config/                      # Hydra 기반 설정 파일 (.yaml)
results/, gradient_alignment_results*/  # 결과 저장 경로 (gitignore 처리)
```

## 빠른 시작

### 1. 환경 설정

```bash
git clone https://github.com/lluvecwonv/paramemo.git
cd paramem
python -m venv .venv && source .venv/bin/activate
pip install -r Influence-Functions/requirements.txt
pip install -r mia/requirements.txt
```

필요한 모델 가중치는 Hugging Face에서 자동으로 다운로드되며, `config/model_config.yaml`에서 모델 식별자를 수정할 수 있다.

### 2. 패러프레이즈 생성

`memorization/config/pile_paraphrase_analysis.yaml`을 적절히 수정한 뒤:

```bash
cd memorization
python paraphrase_main.py
```

출력 파일은 `analysis.output_dir` 하위에 `{model}_{mode}_{domain}_{split}.json` 형태로 저장된다.

### 3. Gradient Alignment 분석

`config/gradient_alignment.yaml`에서 `model_family`, `results_dir`, `output_dir` 등을 지정한 뒤:

```bash
cd memorization
python gradient_alignment_main.py
```

각 train/test 쌍에 대해 정렬도 및 MIA 지표가 `gradient_alignment_results/`에 저장된다.

## 실험 팁

- GPU 메모리 한계가 있다면 `memorization/gradient/gradient_alignment_analyzer.py`의 head-only 경로를 사용하면 효율적이다.
- Hydra를 사용하는 스크립트는 `LOCAL_RANK` 및 `WORLD_SIZE` 환경 변수를 인식하므로 `torchrun`으로 멀티 GPU 돌릴 수 있다.
- `results/`, `outputs/`, `gradient_alignment_results*/` 등 대용량 산출물은 `.gitignore`에 반영되어 있으므로 필요 시 수동 백업을 권장한다.

