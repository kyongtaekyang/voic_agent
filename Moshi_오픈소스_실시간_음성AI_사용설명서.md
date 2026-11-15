# Moshi 오픈소스 실시간 음성 AI 사용 설명서

> **Version:** 1.0  
> **작성 기준일:** 2026-11-12  
> **대상 환경:** Linux GPU 서버 · Windows 11 개발 환경  
> **범위:** 모델 소개, 설치, 사용, Docker, Rust, Python API, 파인튜닝, 라이선스, 오류 해결, 디지털휴먼 연계, PoC

본 설명서는 프랑스 AI 연구기관 Kyutai에서 개발한 **Moshi**를 직접 다운로드하고 설치해 실시간 음성 대화를 실행하는 방법과 서버 운영·상용화 검토 사항을 정리합니다. 공식 저장소 및 공개된 모델 자료를 기준으로 작성하였으며, 명령어는 실제 환경에서 실행 검증한 결과가 아닌 가이드입니다. 특히 프로젝트 버전 변경에 따라 CLI 옵션과 설정 파일 경로는 달라질 수 있으므로 공식 문서를 함께 확인하십시오.

## 주요 공식 링크

- Moshi GitHub: https://github.com/kyutai-labs/moshi
- Hugging Face 모델: https://huggingface.co/kyutai/moshiko-pytorch-bf16
- 실시간 음성 데모: https://moshi.chat/
- 공식 파인튜닝 프로젝트: https://github.com/kyutai-labs/moshi-finetune
- PyTorch CUDA 설치 안내: https://pytorch.org/get-started/locally/

---

## 1. Moshi 모델 소개

Moshi는 프랑스 AI 연구기관 **Kyutai**가 개발한 실시간 음성 대화 모델입니다. 일반적인 `STT → LLM → TTS` 파이프라인과 달리, 음성 입력과 출력을 스트리밍 방식으로 처리하며 사용자가 말하는 동안 AI도 반응할 수 있는 **Full-Duplex(전이중)** 대화 구조를 지원합니다.

| 항목 | 내용 |
|---|---|
| 개발기관 | Kyutai |
| 모델 아키텍처 | Speech-Text Foundation Model |
| 언어모델 규모 | 약 7B 파라미터 |
| 음성 코덱 | Mimi |
| 오디오 샘플레이트 | 24kHz |
| 오디오 프레임 처리 | 초당 12.5프레임 |
| 대화 방식 | Full-Duplex |
| 주요 언어 | 영어 중심 |
| 실행 백엔드 | PyTorch, Rust/Candle, MLX |
| 모델 가중치 라이선스 | CC BY 4.0 |
| Python 코드 라이선스 | MIT |
| Rust 코드 라이선스 | Apache 2.0 |

개발사 발표 기준 이론상 지연시간은 약 **160ms**이며, 특정 GPU 환경에서 약 **200ms** 수준의 실사용 지연시간이 소개되었습니다. 이는 발표된 조건의 수치이며 모든 환경에서 보장되는 성능은 아닙니다.

### 기존 구조와 비교

**일반적인 순차 음성 AI 파이프라인**

```text
사용자 음성
   ↓
STT (음성 → 텍스트)
   ↓
LLM (텍스트 응답 생성)
   ↓
TTS (텍스트 → 음성)
   ↓
사용자에게 음성 출력
```

**Moshi의 음성 직접 처리 방식(개념도)**

```text
사용자 음성 ──┐
              ↕ 실시간 양방향 스트리밍
       Moshi + Mimi
              ↕
AI 응답 음성 ──┘
```

모든 기존 음성 모델이 순차 파이프라인만 사용하는 것은 아닙니다. 위 비교는 일반적인 구조를 단순화한 것입니다.

---

## 2. 전체 시스템 구성

```text
┌──────────────────────────────────────┐
│ Windows 11 / 웹 브라우저             │
│ 마이크 입력 · 스피커 출력             │
└──────────────────┬───────────────────┘
                   ↕
         WebSocket / SSH 터널
                   ↕
┌──────────────────┴───────────────────┐
│ Moshi Inference Server               │
│ PyTorch 또는 Rust/Candle + CUDA GPU │
└──────────────────┬───────────────────┘
                   ↕
┌──────────────────┴───────────────────┐
│ Mimi + Moshi 7B                      │
│ 음성 토큰화 · 대화 모델 · 음성 생성    │
└──────────────────────────────────────┘
```

개발 초기에는 **Python/PyTorch** 버전으로 기능을 검증하는 것이 간편합니다. 동시 사용자 처리, 배포 및 운영 성능을 최적화하는 단계에서는 **Rust/Candle** 백엔드도 검토할 수 있습니다.

---

## 3. 권장 설치 환경

| 구성 | 최소·권장 환경 |
|---|---|
| 운영체제 | Ubuntu 22.04/24.04 등 Linux 권장 |
| Python | 3.10 이상, 3.12 권장 |
| GPU | NVIDIA CUDA 지원 GPU |
| GPU 메모리 | 비양자화 PyTorch 추론 약 24GB 이상 권장 |
| 시스템 RAM | 32GB 이상 권장 |
| 디스크 여유 | 30~50GB 이상 권장 |
| 네트워크 | 최초 모델 다운로드를 위한 인터넷 접속 |
| 클라이언트 | Chrome 또는 Edge + 마이크·스피커 |

- GPU 메모리 기준은 공식 PyTorch 실행 요구 사항에 근거합니다.
- 시스템 RAM·디스크는 운영 편의를 위한 권장값이며 설치 버전과 캐시에 따라 사용량이 달라집니다.
- PyTorch 버전의 양자화 지원 여부는 Rust 백엔드와 다르므로 구분해야 합니다.
- Windows에서는 네이티브 실행보다 Linux 서버에 SSH로 접속하는 구성이 편리합니다.

---

## 4. Linux GPU 서버 설치 절차

### Step 1. NVIDIA GPU 및 CUDA 확인

```bash
nvidia-smi
```

GPU 모델, GPU 메모리와 드라이버 버전을 확인합니다. 실행되지 않으면 NVIDIA 드라이버 또는 장치 접근부터 점검하십시오.

### Step 2. Python 및 기본 도구 설치

```bash
sudo apt update
sudo apt install -y git curl wget python3 python3-venv python3-pip

python3 --version
```

Python 3.10 이상인지 확인합니다. 신규 환경에서는 3.12를 권장합니다.

### Step 3. 작업 폴더 및 가상환경 생성

```bash
mkdir -p ~/ai-models/moshi
cd ~/ai-models/moshi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### Step 4. PyTorch 및 Moshi 설치

```bash
# 서버의 CUDA 환경에 맞는 PyTorch 설치 명령을 공식 사이트에서 우선 확인
pip install torch torchaudio

# Moshi 설치
pip install -U moshi

# CUDA 인식 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('PyTorch:', torch.__version__)"
```

`CUDA: True`를 확인하십시오. `False`라면 CUDA 지원 PyTorch 빌드 설치 여부를 점검하고 https://pytorch.org/get-started/locally/ 에서 해당 환경에 맞는 설치 명령을 선택합니다.

### Step 5. Moshi 서버 실행

```bash
python -m moshi.server
```

특정 모델 저장소를 명시하려면:

```bash
python -m moshi.server \
  --hf-repo kyutai/moshiko-pytorch-bf16
```

최초 실행 시 Hugging Face에서 모델 가중치를 다운로드할 수 있습니다. 모델 파일은 약 15.8GB 수준으로 소개되지만 실제 내려받는 파일과 버전에 따라 다를 수 있습니다.

> **버전 주의:** CLI 인자 `--hf-repo` 등은 설치된 `moshi` 버전의 `python -m moshi.server --help` 또는 공식 README에서 확인하십시오.

### Step 6. 로컬 브라우저 접속

```text
http://localhost:8998
```

일반적인 테스트 순서:

1. 웹 페이지에 접속합니다.
2. 브라우저 마이크 권한을 허용합니다.
3. 사용 가능한 입력·출력 오디오 장치를 확인합니다.
4. 연결 또는 대화 시작을 선택합니다.
5. 영어 음성으로 대화를 시험합니다.
6. 끼어들기, 맞장구 및 턴 전환 반응을 확인합니다.

기본 포트와 UI는 릴리스에 따라 달라질 수 있습니다.

---

## 5. Windows 11에서 원격 Linux GPU 서버 접속

원격 서버의 일반 HTTP 주소는 브라우저 보안 정책으로 마이크 접근이 제한될 수 있습니다. **SSH 로컬 포트 포워딩**을 권장합니다.

### ① Linux 서버에서 실행

```bash
source ~/ai-models/moshi/.venv/bin/activate
python -m moshi.server
```

### ② Windows PowerShell에서 SSH 터널 생성

```powershell
ssh -N -L 8998:localhost:8998 USER@SERVER_IP
```

예시:

```powershell
ssh -N -L 8998:localhost:8998 ubuntu@192.168.0.100
```

명령을 실행한 PowerShell 창은 터널 유지를 위해 열어 둡니다.

### ③ Windows 브라우저에서 접속

```text
http://localhost:8998
```

GPU 연산은 Linux 서버에서, 음성 입출력은 Windows 노트북에서 수행합니다. 외부 사용자에게 서비스를 공개한다면 HTTPS·인증·접속 제어를 별도로 구성해야 합니다.

---

## 6. Docker 기반 설치 및 실행

Docker Compose 기반 설치를 위해 Docker Engine, Compose 플러그인, NVIDIA Container Toolkit을 사전에 준비합니다.

### Step 1. 도구 확인

```bash
docker --version
docker compose version
nvidia-smi
```

GPU 컨테이너 접근 테스트 예시:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi
```

> CUDA 이미지 태그의 존재 여부 및 호환성은 실제 사용 시 확인해야 합니다.

### Step 2. 공식 저장소 복제

```bash
git clone https://github.com/kyutai-labs/moshi.git
cd moshi
```

### Step 3. Docker Compose 실행

```bash
docker compose up
```

백그라운드 실행 및 로그 확인:

```bash
docker compose up -d
docker compose logs -f
```

종료:

```bash
docker compose down
```

`docker-compose.yml`이 해당 릴리스에 존재하는지, GPU 런타임·포트·TLS·모델 설정이 어떻게 정의되어 있는지 먼저 확인하십시오.

---

## 7. Rust/Candle 기반 고성능 추론 서버

공식 프로젝트는 Rust/Candle 기반 추론 구현을 제공합니다. Python 버전으로 기능을 검증한 다음 GPU 효율, 지연시간, 동시 세션 처리에 대한 비교 테스트를 권장합니다.

### Step 1. Rust·CUDA 빌드 도구 확인

```bash
rustc --version
cargo --version
nvcc --version
```

`nvcc`는 CUDA Toolkit이 설치된 환경에서 확인할 수 있습니다.

### Step 2. 공식 저장소에서 실행

```bash
cd ~/ai-models
git clone https://github.com/kyutai-labs/moshi.git
cd moshi/rust

cargo run --features cuda \
  --bin moshi-backend -r -- \
  --config moshi-backend/config.json standalone
```

빌드 시 필요한 패키지와 네이티브 라이브러리는 OS 및 저장소 버전에 따라 달라질 수 있습니다. 로컬 개발용 인증서 경고가 발생할 경우 테스트용 인증서와 운영용 TLS 구성을 구분해야 합니다.

### Step 3. INT8 양자화 구성 예시

```bash
cargo run --features cuda \
  --bin moshi-backend -r -- \
  --config moshi-backend/config-q8.json standalone
```

Rust 백엔드의 양자화 모델은 메모리 절감 가능성이 있습니다. 실제 모델 파일·설정 경로·성능은 해당 버전의 README 및 실측으로 확인해야 합니다.

---

## 8. Python API로 Mimi 코덱 시험

웹 UI 외에도 Moshi의 음성 코덱 Mimi를 Python 코드에서 직접 사용할 수 있습니다. 아래 코드는 **실시간 대화 서버가 아닌 오디오 토큰화·복원 실험 예시**입니다.

```python
import torch
from moshi.models import loaders

repo = "kyutai/moshiko-pytorch-bf16"

# Hugging Face 체크포인트 정보
checkpoint = loaders.CheckpointInfo.from_hf_repo(repo)

# Mimi 모델 로드
mimi = checkpoint.get_mimi(device="cuda")
mimi.set_num_codebooks(8)
mimi.eval()

# 테스트용 10초 모노 오디오
wav = torch.randn(1, 1, 24000 * 10, device="cuda")

with torch.no_grad():
    codes = mimi.encode(wav)
    decoded = mimi.decode(codes)

print("Audio:", wav.shape)
print("Codes:", codes.shape)
print("Decoded:", decoded.shape)
```

> 예제의 클래스명과 함수 인자는 설치된 패키지 버전에 따라 달라질 수 있으므로 공식 코드를 확인하십시오. 마이크 스트리밍과 대화 모델 응답을 통합하려면 별도의 세션·스트리밍 처리가 필요합니다.

---

## 9. 파인튜닝 및 자체 데이터 학습

공식 LoRA 파인튜닝 프로젝트: https://github.com/kyutai-labs/moshi-finetune

영어 학습 대화, 튜터 응답 스타일, 특정 상호작용 상황을 위한 도메인 적응 실험이 가능합니다. 다만 데이터 처리·성능 검증과 파생 모델의 라이선스 관리는 별도 과제입니다.

### Step 1. 파인튜닝 프로젝트 설치

```bash
git clone https://github.com/kyutai-labs/moshi-finetune.git
cd moshi-finetune

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

### Step 2. 학습 데이터 준비

공식 파인튜닝 프로젝트는 스테레오 대화 오디오, 전처리 및 전사 파이프라인을 활용할 수 있도록 설계되어 있습니다.

| 데이터 | 용도 |
|---|---|
| 학습자 발화 음성 | 발화 패턴 파악 |
| AI 교사 발화 음성 | 응답 스타일 학습 |
| 발화 전사문 | 음성·텍스트 정렬 |
| 발화 순서 및 타임스탬프 | 턴 전환 학습 |
| 상황별 대화 | 도메인 적응 |

이 표는 데이터 구성의 개념적 예시이며, 즉시 학습에 사용할 수 있는 확정된 파일 스키마는 아닙니다. 공식 예제에 따라 파일 규격을 준비하십시오.

### Step 3. LoRA 설정 예시

```yaml
lora:
  enable: true
  rank: 128
  scaling: 2.

duration_sec: 100
batch_size: 16
max_steps: 2000
```

이 코드는 대표 항목만 발췌한 예시입니다. 실제 학습에는 모델 경로, 데이터 경로, 옵티마이저 등 완전한 설정이 필요합니다.

### Step 4. 학습 실행 예시

```bash
torchrun --nproc-per-node 1 \
  -m train example/moshi_7B.yaml
```

공식 예시에서는 H100 GPU 한 장으로 최대 약 **39.6GB**의 GPU 메모리가 사용된 설정이 소개됩니다. 일반 추론용 24GB GPU가 동일 학습 설정에 충분하다는 의미는 아닙니다.

### Step 5. LoRA 체크포인트 로드 예시

```bash
CHECKPOINT_DIR="$HOME/moshi-ft/checkpoints/checkpoint_000500"

python -m moshi.server \
  --lora-weight="$CHECKPOINT_DIR/consolidated/lora.safetensors" \
  --config-path="$CHECKPOINT_DIR/consolidated/config.json"
```

실제 체크포인트 폴더와 CLI 지원 옵션을 확인해야 합니다. 원본 모델과 파인튜닝 모델을 동일한 발화 테스트셋에서 비교해 개선 효과를 검증하십시오.

---

## 10. 상업적 이용과 오픈소스 라이선스

| 구성요소 | 라이선스 | 상업적 이용 |
|---|---|---|
| 공개 모델 가중치 | CC BY 4.0 | 조건 준수 시 가능 |
| Python 코드 | MIT | 가능 |
| Rust 백엔드 | Apache 2.0 | 가능 |
| 웹 클라이언트 | MIT (해당 구성 기준) | 가능 |
| 수정·재배포 | 구성별 라이선스 준수 | 가능 |

**주요 준수 사항**

1. 모델 가중치의 원저작자와 출처를 표시합니다.
2. CC BY 4.0 라이선스 고지 및 변경 사실 표기를 준수합니다.
3. 포함된 소스코드의 저작권·라이선스 고지를 유지합니다.
4. 데이터셋 및 제3자 콘텐츠의 권리를 별도로 확인합니다.
5. 운영 서비스의 개인정보 보호 및 보안 통제를 마련합니다.

**아동 음성 데이터 유의 사항:** 오픈소스 모델이라는 사실은 아동 학습 음성을 제한 없이 수집·학습할 수 있다는 의미가 아닙니다. 수집 및 이용 목적, 법정대리인 동의, 보유 기간, 학습용 재사용 적법성, 위탁·제3자 제공 등을 법률·보안 담당자와 검토해야 합니다.

> 제품·모델 버전별로 라이선스 조건이 변경될 수 있으므로 상용 배포 직전에 각 저장소의 LICENSE 및 모델 카드를 재확인하십시오.

---

## 11. 설치·운영 중 주요 오류 해결

| 증상 | 점검 및 해결 방안 |
|---|---|
| `CUDA: False` | CPU 전용 PyTorch 설치 여부, 드라이버 및 CUDA 호환성 확인 |
| `CUDA out of memory` | 다른 GPU 프로세스 종료, 메모리 사용량 점검, 양자화 Rust 모델 검토 |
| `No module named moshi` | 가상환경 활성화 및 `pip install -U moshi` 확인 |
| `nvcc: command not found` | CUDA Toolkit 설치 및 PATH 확인 |
| `Connection refused` | 서버 실행 상태, 포트 및 SSH 터널 확인 |
| 브라우저에서 마이크 불가 | 권한 확인, localhost SSH 터널 또는 HTTPS 사용 |
| 음성 끊김·지연 | 네트워크 RTT, GPU 사용률, 에코 캔슬링 및 스트리밍 상태 확인 |
| 모델 다운로드 실패 | Hugging Face 접근, 네트워크 및 디스크 공간 확인 |
| 한국어 대화 품질 저하 | 영어 중심 모델이라는 제약을 고려하고 별도 평가 수행 |

### 점검 명령어 모음

```bash
# GPU 상태
nvidia-smi

# Python/패키지
python --version
python -m pip show moshi
python -m pip show torch

# CUDA 인식
python -c "import torch; print(torch.cuda.is_available())"

# 서버 포트 확인
ss -lntp | grep 8998

# 서버 도움말 확인
python -m moshi.server --help
```

---

## 12. 영어교육·디지털휴먼 적용 아키텍처

Moshi를 영어교육용 디지털휴먼에 적용할 때는 실시간 대화와 교육 평가·통제 역할을 분리하는 것이 효과적입니다.

```text
┌─────────────────────────────────────┐
│ 학습자 / 디지털휴먼                   │
│ 마이크 · 스피커 · 캐릭터 UI           │
└──────────────────┬──────────────────┘
                   ↕
┌──────────────────┴──────────────────┐
│ Moshi 실시간 대화 엔진               │
│ Full-Duplex · 음성 이해/생성         │
└──────────────────┬──────────────────┘
                   ↕ 이벤트/데이터 연동
┌──────────────────┴──────────────────┐
│ 교육 평가·제어 계층                 │
│ STT · 정답 판정 · 발음/문법 평가     │
└──────────────────┬──────────────────┘
                   ↕
┌──────────────────┴──────────────────┐
│ 학습 정책 및 LLM 계층               │
│ 커리큘럼 · 학습 규칙 · 피드백        │
└──────────────────┬──────────────────┘
                   ↕
┌──────────────────┴──────────────────┐
│ LMS / 학습 데이터                   │
│ 진도 · 점수 · 학습 이력 · 분석       │
└─────────────────────────────────────┘
```

**설계 관점의 권장 역할 분담**

- Moshi: 자연스러운 실시간 음성 상호작용 및 턴 전환
- STT/평가 엔진: 텍스트 인식, 정답·발음·문법 평가
- 정책·LLM 계층: 수업 목표, 피드백 생성, 학습 규칙 적용
- LMS: 진도·점수·학습 이력 관리

이 구조는 제안 아키텍처입니다. **Moshi에 교육 평가나 LMS 연계가 기본 내장되어 있다는 의미가 아닙니다.** 대화 흐름과 별도 평가 계층의 동기화, 세션 관리, 오류 처리 등이 추가 개발 과제입니다.

---

## 13. 단계별 PoC 추진 방안

### 1단계. 기본 설치 및 동작 확인

- Linux GPU 서버에 PyTorch 버전을 설치합니다.
- Windows 브라우저에서 영어 음성 대화를 실시합니다.
- 기본 응답, GPU 메모리 및 음질을 확인합니다.

### 2단계. 성능 및 동시 접속 평가

- 음성 응답 지연시간과 턴 전환을 측정합니다.
- 사용자가 AI 발화를 중단시키는 상황을 테스트합니다.
- 동시 세션 수와 GPU 사용률을 확인합니다.

### 3단계. 기존 AI 서비스 연계

- 웹 기반 디지털휴먼 프런트엔드와 연동합니다.
- 교육용 이벤트 및 평가 엔진과 연계합니다.
- 세션 로그·관측성과 실패 시 처리 방안을 준비합니다.

### 4단계. LoRA 파인튜닝

- 적법하게 확보한 영어 학습 대화 데이터를 준비합니다.
- LoRA 기반 도메인 적응을 실험합니다.
- 원본 대비 응답 품질과 학습효과를 비교합니다.

### 5단계. 상용화 가능성 검증

- Rust 백엔드와 Python 백엔드를 비교합니다.
- GPU 용량 및 동시 사용자 규모를 산정합니다.
- 보안·개인정보·장애 대응·운영 비용을 검토합니다.

---

## 14. 설치 완료 검증 체크리스트

- [ ] NVIDIA GPU 및 CUDA 정상 인식
- [ ] Python 가상환경 생성 및 Moshi 설치
- [ ] Moshi 모델 다운로드 및 서버 실행
- [ ] Windows 브라우저 원격 접속
- [ ] 마이크·스피커 음성 입출력 정상
- [ ] 영어 실시간 대화 및 끼어들기 테스트
- [ ] GPU 메모리와 응답 지연시간 측정
- [ ] 라이선스 및 음성 데이터 이용 조건 검토

---

## 최종 권고

Moshi 도입의 첫 목표는 기존 음성 AI를 즉시 대체하는 것보다, **Full-Duplex 대화 기술이 실사용 교육 서비스에서 얼마나 상호작용 품질을 개선하는지 확인**하는 것입니다.

**권장 PoC 환경:** `Linux + NVIDIA GPU + Python 3.12 + Moshi(PyTorch) + Windows 브라우저(SSH 터널)`

기능 검증을 마친 뒤 Rust/Candle 운영 백엔드와 자체 파인튜닝을 단계적으로 검토하십시오.

---

## 참고 자료

1. Kyutai Moshi 공식 저장소: https://github.com/kyutai-labs/moshi
2. Moshi 모델 가중치: https://huggingface.co/kyutai/moshiko-pytorch-bf16
3. Moshi 데모: https://moshi.chat/
4. Moshi 파인튜닝 프로젝트: https://github.com/kyutai-labs/moshi-finetune
5. PyTorch 설치 안내: https://pytorch.org/get-started/locally/

**정보 신뢰도:** 💚 높은 편 — 공식 공개 자료를 중심으로 작성. 다만 릴리스별 CLI 옵션·설정 파일·GPU 성능은 설치 환경에서 검증해야 합니다.
