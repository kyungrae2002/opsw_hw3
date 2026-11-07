"""영어 리뷰 감성(긍정/부정) 분석기 - ONNX Runtime CPU 추론.

사용할 모델 설정은 환경변수 MODEL_NAME 으로 고릅니다 (기본값: DEFAULT_MODEL).
모델마다 출력 형식이 달라서(2클래스 / 3클래스 / 별점 5단계) MODEL_SPECS 표로 관리하고,
최종 결과는 모두 "긍정 확률 vs 부정 확률" 두 값으로 바꿔서 돌려줍니다.
모델 비교 실험 결과는 reports/model_update.md 참고.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# 모델 출력(클래스 순서)과 긍정/부정으로 합칠 때의 가중치.
# 중립(3클래스의 neutral, 별점 3점)은 긍정·부정에 반씩 나눕니다.
OUTPUT_KINDS: dict[str, tuple[list[str], list[float]]] = {
    "binary": (["NEGATIVE", "POSITIVE"], [0.0, 1.0]),
    "3class": (["negative", "neutral", "positive"], [0.0, 0.5, 1.0]),
    "stars5": (["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], [0.0, 0.0, 0.5, 1.0, 1.0]),
}


@dataclass(frozen=True)
class ModelSpec:
    file: str  # models/ 안의 ONNX 파일
    tokenizer: str  # models/ 안의 tokenizer.json
    output: str  # OUTPUT_KINDS 의 키
    source: str  # 원본 모델 (Hugging Face)
    max_tokens: int = 512  # 특수 토큰 포함 최대 길이
    truncation: str = "head"  # "head": 앞부분만, "head_tail": 앞+뒤 (긴 리뷰의 결론부 보존)
    clean: bool = False  # HTML 엔티티·반복 문장부호·공백 정리


# 기존(v1.0) 모델 + 개선 실험에서 시도한 설정 (scripts/download_model.py 와 파일명이 같아야 함)
MODEL_SPECS: dict[str, ModelSpec] = {
    # 기준선: 영화 리뷰 문장(SST-2)으로 학습한 DistilBERT (Apache-2.0)
    "distilbert-sst2": ModelSpec(
        "distilbert-sst2.onnx", "distilbert-sst2.tokenizer.json", "binary",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    ),
    # 후보: 상품 리뷰 별점(1~5)으로 학습한 다국어 BERT (MIT), INT8 양자화 ONNX
    "nlptown-stars": ModelSpec(
        "nlptown-stars-int8.onnx", "nlptown-stars.tokenizer.json", "stars5",
        "nlptown/bert-base-multilingual-uncased-sentiment (ONNX: Xenova)",
    ),
    # 후보: 트윗으로 학습한 RoBERTa 3클래스 (CC BY 4.0), INT8 양자화 ONNX
    "twitter-roberta": ModelSpec(
        "twitter-roberta-int8.onnx", "twitter-roberta.tokenizer.json", "3class",
        "cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX: Xenova)",
    ),
}
DEFAULT_MODEL = "distilbert-sst2"  # scripts/download_model.py 의 DEFAULT_MODEL 과 같아야 함

HEAD_TOKENS = 128  # head_tail 방식에서 앞부분에 남길 토큰 수 (나머지는 뒷부분)


class InvalidTextError(ValueError):
    """분석할 수 없는 입력(빈 문자열 등)."""


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"([!?.])\1{2,}", r"\1\1", text)  # "!!!!!" → "!!"
    return re.sub(r"\s+", " ", text).strip()


class SentimentModel:
    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or os.getenv("MODEL_NAME") or DEFAULT_MODEL
        if name not in MODEL_SPECS:
            raise ValueError(f"알 수 없는 모델: {name} (가능: {', '.join(MODEL_SPECS)})")
        self.spec = spec = MODEL_SPECS[name]
        model_path = MODELS_DIR / spec.file
        tok_path = MODELS_DIR / spec.tokenizer
        for path in (model_path, tok_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"모델 파일이 없습니다: {path}. `python scripts/download_model.py {name}`를 먼저 실행하세요."
                )
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        # 빈 문자열을 인코딩하면 [CLS][SEP] / <s></s> 처럼 앞뒤 특수 토큰만 남음
        empty = self.tokenizer.encode("").ids
        self.prefix, self.suffix = empty[:1], empty[1:]

        self.class_labels, weights = OUTPUT_KINDS[spec.output]
        self.pos_weights = np.array(weights, dtype=np.float64)
        self.model_name = name

    def encode(self, text: str) -> tuple[list[int], bool]:
        body = self.tokenizer.encode(text, add_special_tokens=False).ids
        room = self.spec.max_tokens - len(self.prefix) - len(self.suffix)
        truncated = len(body) > room
        if truncated:
            if self.spec.truncation == "head_tail":
                body = body[:HEAD_TOKENS] + body[-(room - HEAD_TOKENS):]
            else:
                body = body[:room]
        return self.prefix + body + self.suffix, truncated

    def analyze(self, text: str) -> dict:
        if self.spec.clean:
            text = clean_text(text)
        if not text or not text.strip():
            raise InvalidTextError("분석할 문장을 입력해 주세요.")
        ids, truncated = self.encode(text)
        arr = np.array([ids], dtype=np.int64)
        feeds = {"input_ids": arr, "attention_mask": np.ones_like(arr)}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(arr)
        logits = self.session.run(None, feeds)[0][0].astype(np.float64)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        positive = float(np.clip(probs @ self.pos_weights, 0.0, 1.0))
        label = "positive" if positive >= 0.5 else "negative"
        return {
            "label": label,
            "score": positive if label == "positive" else 1.0 - positive,
            "scores": {"positive": positive, "negative": 1.0 - positive},
            "detail": [{"label": l, "probability": float(p)} for l, p in zip(self.class_labels, probs)],
            "tokens": len(ids),
            "truncated": truncated,
        }
