# Nemotron 3.5 Transcribe 검토 보고서: 이클라우드에이아이 적용 방안

Oct 31, 2026 · @이클라우드에이아이

## 개요

Nemotron 3.5 Transcribe는 녹음된 음성을 처리하는 오프라인 전사 모델이므로, 이클라우드에이아이에서는 실시간 대화보다 녹음 기반 업무에 먼저 적용하는 것이 적합합니다. 한국어는 상위 지원 언어(Leading tier)에 포함되고 화자 구분도 지원하지만, 아동 음성에 대한 성능은 공개된 자료가 없어서 자체 데이터로 검증해야 합니다.

- **우선 적용 후보:** 아동 발화(VOC) 데이터 라벨링 초벌 전사, 수업 녹화 영상 전사와 화자 구분, 사내 회의록 작성입니다.
- **부적합 영역:** ELEA 실시간 대화와 음성 파이프라인 PoC의 실시간 STT 단계입니다. 이 용도에는 같은 계열의 스트리밍 모델(Nemotron 3.5 ASR Streaming 0.6B)을 별도로 검토해야 합니다.
- **권장 다음 단계:** 보유 중인 Blackwell GPU 서버에 설치한 뒤, 현재 사용 중인 Whisper 파인튜닝 모델과 같은 아동 음성 평가 세트로 인식률을 비교합니다.

## Nemotron 3.5 Transcribe 소개

NVIDIA가 2026년 10월 28일에 공개한 오픈웨이트 음성 모델이며, 모델 하나로 전사, 화자 구분, 번역, 음성 질의응답을 처리합니다. 기존 STT처럼 음성을 그대로 받아쓰는 방식이 아니라, 텍스트 지시문(프롬프트)으로 작업과 출력 형식을 선택하는 음성 LLM입니다.

| 항목 | 내용 |
| --- | --- |
| 개발사 | NVIDIA (Nemotron 3.5 계열) |
| 공개일 | 2026-10-28 |
| 모델 ID | [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) |
| 규모 | 총 34B 파라미터, 토큰당 약 4.3B 활성(MoE 구조) |
| 처리 방식 | 오프라인(녹음 파일), 요청당 최대 1시간, 컨텍스트 65,536 토큰 |
| 지원 언어 | 전사 40개, 영어 양방향 번역 31개, 화자 구분 13개(한국어 포함) |
| 정밀도 | BF16, FP8, NVFP4 |
| 라이선스 | OpenMDW-1.1(상업적 이용 허용) |

같은 시기에 공개된 학습 데이터 파이프라인인 Granary v2는 48개 언어, 약 463,062시간 분량의 공개 매니페스트를 제공합니다. 실시간 용도로는 2026년 6월에 공개된 0.6B 규모의 Nemotron 3.5 ASR Streaming이 별도로 있습니다.

## 주요 특징과 기술 설명

이 모델은 음성 인코더와 화자 인코더가 추출한 특징을 언어 모델에 전달하고, 언어 모델이 지시문에 맞춰 텍스트를 생성하는 구조입니다. 처리 순서는 다음과 같습니다.

1. 16kHz 모노 음성(wav, flac, mp3, ogg)을 입력받아 log-mel 특징으로 변환합니다.
2. 600M 규모의 Transformer ASR 인코더가 30초 단위로 음성을 처리하고, 99M 규모의 Nemotron 3 Diarization 화자 인코더가 화자 정보를 추출합니다.
3. 두 인코더의 출력을 Nemotron 3.5 Lightning 30B-A3B 언어 모델(Mamba-2, Attention, MoE 혼합 구조)이 받아 지시문에 맞는 텍스트를 생성합니다.
4. 생성이 끝나면 CTC 헤드가 단어별 타임스탬프를 음성에 정렬합니다.

지시문만 바꾸면 재학습 없이 다음 기능을 선택할 수 있습니다.

- **축어 전사(Verbatim):** 추임새와 반복을 포함해 들리는 그대로 기록합니다.
- **문어 전사(Written form):** 추임새와 실수로 반복한 말을 제거합니다.
- **컨텍스트 용어 지정:** 고유명사와 전문 용어 목록을 지시문에 넣어 인식률을 높입니다.
- **화자 구분:** 최대 8명까지 `<spk:0>`\~`<spk:7>` 태그와 단어 타임스탬프를 붙입니다. 태그는 목소리를 구분할 뿐, 실제 인물을 식별하지는 않습니다.
- **리치 전사:** `<ev:laughter>`와 같은 소리 이벤트를 표시합니다.
- **번역과 언어 식별:** 영어와 31개 언어 사이의 음성 번역, 발화 언어 식별을 지원합니다.
- **음성 질의응답:** 녹음 내용에 대해 질문할 수 있지만, 영어로만 학습하고 평가되었습니다.

### 성능(NVIDIA 발표 기준, 낮을수록 우수)

| 평가 항목 | Nemotron 3.5 Transcribe | 비교 모델 |
| --- | --- | --- |
| 영어 Open ASR 9개 세트 평균 WER | 4.71% | Qwen3-ASR-1.7B 4.94%, Whisper large-v3 6.66% |
| 장시간 영어 평균 WER | 10.98% | Parakeet-TDT-0.6B-v3 13.37% |
| 다국어 7개 언어 평균 WER | 4.62% | Qwen3-ASR-1.7B 6.49% |
| 다화자 평균 cpWER | 15.80% | DiCoW v3.3 + DiariZen 16.83% |
| 한국어 CER | 2.78% | 미공개 |

한국어 CER은 2차 자료([OrcaRouter](https://www.orcarouter.ai/blog/nemotron-3-5-transcribe-30b-explained))에서 확인한 수치이므로, 도입 전에 모델 카드와 [기술 보고서](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf)로 다시 확인해야 합니다. 코드형 발화(VoiceCodeBench)와 언어 혼용 발화(CS-FLEURS)에서는 Whisper large-v3가 더 좋은 결과를 보였습니다.

## 설치 방법

NVIDIA가 제시하는 기준 경로는 Linux(또는 WSL2) 환경에서 vLLM 0.32.0과 NeMo Speech 플러그인으로 OpenAI 호환 API 서버를 띄우는 방식입니다. NVIDIA는 NVIDIA GPU가 장착된 Linux와 Windows만 공식적으로 지원합니다.

### 요구 사양

| 정밀도 | 공식 예시 GPU | 비고 |
| --- | --- | --- |
| BF16 | H100 80GB 1장 | 기준 구성이며 정확도가 가장 높습니다. |
| FP8 | L40S 48GB, H100 | 평균 오류율이 약 1.0% 증가합니다. |
| NVFP4 | RTX 5090 | 평균 오류율이 약 3.6% 증가하며, 비영어와 화자 구분에서 더 크게 떨어집니다. |

모델 카드는 A100, H200, B200, RTX PRO 6000, DGX Spark, Jetson Thor 등도 지원 하드웨어로 명시합니다. BF16 가중치는 약 68GB로 추정되므로, 96GB의 RTX PRO 6000 Blackwell에서는 BF16 구동이 가능하고 48GB의 RTX 6000 Ada에서는 FP8 이하로 구동해야 할 것으로 보입니다(파라미터 수 기준 추정치).

### 설치 절차

1. NVIDIA 드라이버와 CUDA가 설치된 Linux 서버를 준비합니다.
2. vLLM 0.32.0과 NeMo Speech 플러그인을 설치합니다. 정확한 패키지 설치 명령은 Hugging Face 모델 카드와 [NVIDIA-NeMo Speech 저장소](https://github.com/NVIDIA-NeMo)에서 확인해야 합니다.
3. 가중치를 내려받습니다.
4. 환경 변수를 설정하고 vLLM 서버를 실행합니다.

```bash
# 3. 가중치 다운로드
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --local-dir ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B

# 4. 환경 변수와 서버 실행(모델 카드 기준)
export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0
vllm serve ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.94 \
  --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 192 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":6,"length":120000000}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
```

빠르게 전사만 확인하려면 NeMo 없이 구동되는 SGLang-Omni 0.2.0을 사용할 수 있습니다. 다만 이 경로에서는 타임스탬프, 화자 구분, 번역을 사용할 수 없습니다.

```bash
sgl-omni serve --model-path nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --model-name nemotron-transcribe --mem-fraction-static 0.85 --port 8000
```

첫 요청은 커널을 컴파일하느라 느리게 처리됩니다. 공개 빌드(vLLM, NeMo)는 NVIDIA NIM 컨테이너보다 약 2배 느린 것으로 안내됩니다.

## 사용법

서버를 띄운 뒤에는 OpenAI Python SDK로 음성과 지시문을 함께 보내면 됩니다. 전사할 때는 시스템 프롬프트를 비워 두고, temperature는 0으로, 추론(thinking) 기능은 끄는 것이 권장 설정입니다. 음성 질의응답에서만 추론을 켜고 temperature를 0.6으로 설정합니다.

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
audio = base64.b64encode(open("audio.wav", "rb").read()).decode()

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Transcribe the following audio."},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}},
    ]}],
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False},
                "skip_special_tokens": False},
)
print(response.choices[0].message.content)
```

### 작업별 지시문 예시

| 작업 | 지시문 예시 | 용도 |
| --- | --- | --- |
| 기본 전사 | Transcribe the following audio. | 일반 전사 |
| 용어 지정 | Transcribe with punctuation and capitalization. Context terms: ELEA, 엘라버스, 노른자국어. | 제품명, 교재 어휘 인식 |
| 번역 | Please translate the given speech to English. | 한국어 발화를 영어로 변환 |
| 회의 요약용 질문 | What decisions were made in this meeting? | 영어 녹음에 한해 안정적입니다. |

축어, 문어, 화자 구분 등 세부 형식을 선택하는 정확한 지시문 문구는 모델 카드에 공개된 템플릿을 그대로 사용하는 것이 안전합니다. 화자 구분과 단어 타임스탬프는 위의 vLLM 서버처럼 타임스탬프 미들웨어를 켜야 반환됩니다.

사용 시 주의할 점은 다음과 같습니다.

- vLLM은 기본값으로 음성 디코딩을 600초로 제한하므로, 1시간 파일을 처리하려면 환경 변수를 설정해야 합니다.
- 발화가 촘촘한 1시간 녹음은 출력 토큰이 약 20,000개까지 필요하므로 max\_tokens를 충분히 늘려야 합니다.
- 용어 목록을 잘못 넣으면 엉뚱한 단어로 끌려갈 수 있고, 드물게 같은 출력이 반복되는 현상도 보고되었습니다.

## 활용법

이 모델은 전사, 화자 구분, 타임스탬프, 번역을 한 번에 처리하므로, 여러 모델을 연결해 쓰던 녹음 처리 업무를 단순화하는 데 강점이 있습니다.

| 활용 분야 | 사용 기능 | 기대 효과 |
| --- | --- | --- |
| 회의록, 인터뷰 기록 | 화자 구분, 문어 전사 | 누가 무엇을 말했는지 정리된 텍스트를 바로 얻을 수 있습니다. |
| 강의, 수업 영상 자막 | 타임스탬프, 번역 | 단어 단위 시간 정보로 자막과 다국어 자막을 만들 수 있습니다. |
| 음성 학습 데이터 구축 | 축어 전사, 타임스탬프 | 사람이 검수하기 전의 초벌 라벨을 자동으로 생성합니다. |
| 전문 분야 녹음 처리 | 컨텍스트 용어 지정 | 실적 발표 통화에서 키워드 재현율이 58%에서 96%로 올랐다고 보고되었습니다. |
| 상담, 콜센터 품질 관리 | 축어 전사, 화자 구분 | 통화 녹음을 일괄 처리해 검수와 분석에 활용합니다. |

필요하면 NeMo 기반 파인튜닝이나 ModelScope의 ms-swift로 도메인에 맞게 추가 학습할 수 있고, NVIDIA는 코딩 에이전트가 적응 학습 절차를 수행하도록 돕는 파인튜닝 스킬도 함께 공개했습니다. 다만 의료 음성으로 적응 학습한 실험에서는 일반 영어 성능이 약간 떨어졌으므로, 파인튜닝 후에는 기존 용도의 성능도 다시 점검해야 합니다.

## 이클라우드에이아이 적용 방안

당사는 현재 Whisper와 MS Azure STT를 병행하고 있으며 Whisper 파인튜닝으로 일원화할 계획이므로, Nemotron 3.5 Transcribe는 이 계획을 대체하기보다 녹음 기반 업무에서 보완하는 방향으로 검토하는 것이 현실적입니다. 아래 적합도는 공개 자료를 바탕으로 판단한 것이며, 아동 음성 성능은 PoC로 확인해야 합니다.

| 적용 대상 | 적합도 | 활용 방식과 판단 근거 |
| --- | --- | --- |
| 아동 발화(VOC) 데이터 라벨링 | 높음 | 수집된 녹음을 축어 전사하고 단어 타임스탬프를 붙여 초벌 라벨로 사용합니다. 검수 인력의 작업량을 줄이고, Whisper 파인튜닝용 학습 데이터 구축 속도를 높일 수 있습니다. |
| 수업 녹화 영상 분석(KIST 과제 연계) | 높음 | 튜터와 학생의 발화를 화자별로 나누고 시간 정보를 붙여, 학습자 상태 인식 모델의 입력 데이터로 활용합니다. 한국어와 영어 모두 화자 구분 지원 언어입니다. |
| 사내 회의록, 산학 협의 기록 | 높음 | 회의 녹음을 화자 구분 전사하고, 사내 RAG 어시스턴트의 문서로 적재합니다. 온프레미스에서 처리하므로 내부 정보가 외부로 나가지 않습니다. |
| 노른자국어, 한국어 강사 콘텐츠 | 보통 | 강의 영상의 한국어 자막과 영어 번역 자막을 만드는 데 쓸 수 있습니다. 한국어의 번역 지원 여부는 언어 지원표에서 다시 확인해야 합니다. |
| ELEA 실시간 대화 | 낮음 | 오프라인 전용 모델이라 실시간 응답에 쓸 수 없습니다. 실시간 STT는 Nemotron 3.5 ASR Streaming 0.6B(한국어 포함, CPU 구동 가능)를 별도로 비교 평가하는 것이 적합합니다. |
| 한국어 음성 파이프라인 PoC(STT→LLM→TTS) | 낮음 | 경량 모델 기준의 실시간 대화 구조이므로 규모와 처리 방식이 맞지 않습니다. 녹음 파일 입력 모드의 비교 옵션으로만 추가할 수 있습니다. |

### 인프라 적용

- **외부 시스템(Blackwell 서버):** RTX PRO 6000 Blackwell 96GB라면 BF16 구동이 가능할 것으로 보이므로, PoC는 이 서버에서 진행하는 것을 권장합니다.
- **내부 시스템(RTX 6000 Ada 48GB):** FP8 가중치로 구동해야 하며, 한국어와 화자 구분에서 정확도가 얼마나 떨어지는지 함께 측정해야 합니다.
- **외부 API:** DeepInfra가 분당 US$0.002, 데이터 미보관 조건으로 제공하지만, 아동 음성을 국외 서버로 보내는 것은 개인정보 국외 이전 문제가 있어 성인 음성이나 비개인정보 데이터로 한정하는 것이 안전합니다.

### R&D 로드맵과의 연계

WBS의 Task2(STT/OCR 오픈소스 전환)에서 Whisper 외의 비교 후보로 등록해 평가하면 됩니다. 서울시립대와 진행하는 아동 영어 발화 인식 과제에도 기준 모델(베이스라인) 중 하나로 제시할 수 있습니다.

## 고려 사항과 제언

도입 여부는 아동 음성 인식률과 하드웨어 운영 비용을 자체 데이터로 확인한 뒤에 결정하는 것이 적절합니다.

- **아동 음성 성능 미검증:** 공개 벤치마크는 성인 음성 기준이며, 5\~13세 아동의 발음과 비문법적 발화에 대한 결과는 없습니다.
- **모델 규모:** 활성 파라미터는 약 4.3B이지만 전체 34B 가중치를 메모리에 올려야 하므로, Whisper(1.55B)보다 GPU 자원을 훨씬 많이 사용합니다.
- **라이선스:** OpenMDW-1.1은 상업적 이용을 허용하지만, 파인튜닝 모델을 파생물로 배포하거나 기술특례상장 자료에 자체 모델(ELM)로 설명할 때의 조건은 법무 검토가 필요합니다.
- **개인정보:** 아동 음성은 온프레미스에서만 처리하고, 화자 태그가 실제 인물을 식별하지 않는다는 점을 내부관리규정에 반영합니다.
- **문어 전사의 한계:** 추임새와 반복을 제거하면 원본 발화가 바뀌므로, 발화 분석이나 학습 데이터에는 반드시 축어 전사를 사용해야 합니다.
- **미확인 사항:** 한국어가 번역 31개 언어에 포함되는지, FP8과 NVFP4 가중치가 모두 공개되었는지는 모델 카드에서 확인해야 합니다.

### 다음 단계

- [ ] Blackwell 서버에 vLLM 기반으로 설치하고 구동을 확인합니다.
- [ ] 아동 영어·한국어 발화 평가 세트(예: 500건)로 Whisper 파인튜닝 모델, Azure STT와 WER/CER을 비교합니다.
- [ ] 수업 녹화 영상 샘플로 화자 구분 정확도를 확인합니다.
- [ ] 실시간 용도로 Nemotron 3.5 ASR Streaming 0.6B를 별도로 평가합니다.
- [ ] OpenMDW-1.1 라이선스 조건을 법무 검토합니다.

## 참고 자료

- [NVIDIA Nemotron 3.5 Transcribe 기술 보고서](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf)
- [Hugging Face 모델 카드](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
- [Cekura: 사양, 언어, 설치 정리](https://www.cekura.ai/blogs/nemotron-3-5-transcribe)
- [Nicolas de Ville: 기술 요약과 실행 예시](https://notes.nicolasdeville.com/ai/nemotron-transcribe/)
- [OrcaRouter: 화자 구분 언어와 언어별 성능](https://www.orcarouter.ai/blog/nemotron-3-5-transcribe-30b-explained)
- [XenoSpectrum: 하드웨어와 제한 사항](https://xenospectrum.com/en/nvidia-nemotron-3-5-transcribe-comparison/)
- [DeepInfra 호스팅 API](https://deepinfra.com/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
- [The New Stack: 출시 기사](https://thenewstack.io/?p=22841900)
- [Unite.AI: Granary v2와 배포 경로](https://www.unite.ai/?p=496393)
- [Together AI: Nemotron 3.5 ASR Streaming 0.6B](https://www.together.ai/models/nvidia-nemotron-35-asr)
