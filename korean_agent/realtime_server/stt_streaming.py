"""Nemotron 3.5 ASR Streaming 0.6B 를 transformers 로 실행하는 speech-to-speech STT 처리기.

speech-to-speech 1.1.0 의 `--stt nemotron-streaming` 은 NeMo 툴킷이 필요하다. NeMo 대신 1단계에서 검증한
transformers 경로(AutoModelForRNNT)를 쓰도록 NemoASRSTTHandler 의 모델 로드와 인식 부분만 바꾼다.
VAD 가 자른 발화 구간(progressive: 말하는 중 부분 인식, final: 발화 끝 확정 인식)을 한 번에 인식한다.
1단계 측정에서 CPU 8스레드 오프라인 인식의 RTF 는 0.066 이었다.

language
- "ko-KR": 한국어로 고정한다. 이 설정에서는 영어 발화가 빈 문자열이나 "이" 같은 한두 음절로 인식된다
  (docs/reports/lang_mismatch_probe.json). 그래서 확정 결과의 한글 음절 수가 발화 길이에 비해 적으면
  (초당 probe_min_cps 미만, 발화 probe_min_sec 이상) 같은 구간을 language=auto 로 한 번 더 인식해서
  언어 태그(없으면 한글 비율)로 영어 여부를 판단한다. 한국어 인식 결과 자체는 바꾸지 않는다.
- "auto" : 모델이 발화마다 언어 태그(<ko-KR>, <en-US> ...)를 붙이고, 그 태그로 감지한다.
영어 위주 발화가 mismatch_turns 번 이어지면 language_mismatch 이벤트를 남긴다 (프롬프트 2.1절 6번).
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np

from speech_to_speech.STT.nemo_asr_handler import NemoASRSTTHandler, _split_lang_tag

from realtime_server.korean_lm_processor import log_event

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_MODEL = os.environ.get(
    "KVA_NEMOTRON_MODEL", "/home/yang/pgm/Korean_interface/models/stt/nemotron-3.5-asr-streaming-0.6b")
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_LANG_TAG = re.compile(r"<([a-z]{2})-[A-Z]{2}>")


def hangul_ratio(text: str) -> float | None:
    h, latin = len(_HANGUL.findall(text)), len(_LATIN.findall(text))
    return None if h + latin == 0 else h / (h + latin)


class NemotronTransformersSTTHandler(NemoASRSTTHandler):
    def setup(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", language: str = "ko-KR",
              gen_kwargs: dict | None = None, **_ignored) -> None:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        if model_name.startswith("nvidia/") and os.path.isdir(DEFAULT_MODEL):
            model_name = DEFAULT_MODEL  # HF 캐시에는 가중치가 없으므로 로컬 사본을 쓴다 (docs/version_check.md)
        self.device = "cpu" if device in ("auto", "cpu") else device
        threads = int(os.environ.get("KVA_STT_THREADS", "8"))
        if self.device == "cpu":
            torch.set_num_threads(threads)
        self.model_name = model_name
        self.prompt_language = language if language != "ko" else "ko-KR"
        self._detects_utterance_language = self.prompt_language == "auto"
        self._detect_language_from_text = False
        self._is_farsi = False
        self.start_language = "ko"
        self.language = "ko"
        self.last_language = "ko"
        self.gen_kwargs = dict(gen_kwargs or {})
        self.mismatch_turns = int(os.environ.get("KVA_MISMATCH_TURNS", "2"))
        self.probe_min_samples = int(float(os.environ.get("KVA_LANG_PROBE_MIN_SEC", "0.6")) * SAMPLE_RATE)
        self.probe_min_cps = float(os.environ.get("KVA_LANG_PROBE_MIN_CPS", "1.5"))  # 초당 한글 음절 하한
        self._non_korean_streak = 0
        logger.info("Loading Nemotron (transformers) %s on %s, language=%s, threads=%s",
                    model_name, self.device, self.prompt_language, threads)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForRNNT.from_pretrained(model_name, device_map=self.device).eval()
        self._torch = torch
        self.warmup()

    def _transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        if audio.size < SAMPLE_RATE // 10:
            return ""
        language = language or self.prompt_language
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, language=language)
        inputs.to(self.model.device, dtype=self.model.dtype)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=512)
        text = self.processor.decode(out.sequences, skip_special_tokens=language != "auto")
        if isinstance(text, list):
            text = text[0]
        # skip_special_tokens=False 일 때 남는 특수 토큰 가운데 언어 태그만 남긴다
        text = re.sub(r"<(?![a-z]{2}-[A-Z]{2}>)[^>]*>", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def process(self, vad_audio):
        for out in super().process(vad_audio):
            if vad_audio.mode != "progressive":
                self._check_language(out, np.asarray(vad_audio.audio, dtype=np.float32))
            yield out

    def _is_sparse(self, text: str, audio: np.ndarray) -> bool:
        """발화 길이에 비해 한글 음절이 너무 적으면(영어를 ko-KR 로 인식하면 빈 문자열이나 "이" 같은 한두 음절이 나온다) True."""
        if _LATIN.search(text) and (hangul_ratio(text) or 0) < 0.5:
            return False  # 라틴 문자가 이미 많으면 한글 비율로 판단한다
        seconds = audio.size / SAMPLE_RATE
        return len(_HANGUL.findall(text)) < seconds * self.probe_min_cps

    def _probe_language(self, audio: np.ndarray) -> tuple[bool, str, str | None]:
        """ko-KR 결과가 빈 구간을 auto 로 다시 인식한다. 반환: (한국어가 아님, 인식 텍스트, 언어 태그)"""
        text = self._transcribe(audio, language="auto")
        m = _LANG_TAG.search(text)
        clean = _LANG_TAG.sub("", text).strip()
        if m:
            return m.group(1) != "ko", clean, m.group(0)
        r = hangul_ratio(clean)
        return (r is not None and r < 0.5), clean, None

    def _check_language(self, out, audio: np.ndarray | None = None) -> None:
        text = getattr(out, "text", "") or ""
        language_code = getattr(out, "language_code", None)
        probed = False
        if self._detects_utterance_language:
            non_korean = language_code not in ("ko", None)
        elif audio is not None and audio.size >= self.probe_min_samples and self._is_sparse(text, audio):
            non_korean, text, language_code = self._probe_language(audio)
            probed = True
        else:
            r = hangul_ratio(text)
            non_korean = r is not None and r < 0.5
        self._non_korean_streak = self._non_korean_streak + 1 if non_korean else 0
        if non_korean:
            logger.info("non-Korean utterance (streak=%d, probed=%s, tag=%s)", self._non_korean_streak, probed,
                        language_code)
        if self._non_korean_streak >= self.mismatch_turns:
            # 사용자(아동) 발화 원문은 log_transcripts 를 켰을 때만 남긴다.
            keep_text = os.environ.get("KVA_LOG_TRANSCRIPTS", "0") == "1"
            log_event("language_mismatch", text=text if keep_text else None, chars=len(text),
                      streak=self._non_korean_streak, probed=probed,
                      language_code=language_code, turn_id=getattr(out, "turn_id", None))

    def on_session_end(self) -> None:
        self._non_korean_streak = 0
        super().on_session_end()


def split_lang_tag(text: str):  # 테스트용 재노출
    return _split_lang_tag(text)
