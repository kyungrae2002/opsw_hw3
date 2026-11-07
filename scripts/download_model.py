"""사전학습 감성 분석 모델(ONNX)과 토크나이저를 내려받고 SHA256으로 무결성을 확인합니다.

모든 URL은 Hugging Face 저장소의 특정 커밋으로 고정되어 있어 항상 같은 파일을 받습니다.

사용법:
  python scripts/download_model.py                  # 서비스 기본 모델
  python scripts/download_model.py nlptown-stars    # 특정 모델
  python scripts/download_model.py --all            # 비교 실험용 전체
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
SST2 = f"{HF}/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/714eb0fa89d2f80546fda750413ed43d93601a13"
NLPTOWN = f"{HF}/Xenova/bert-base-multilingual-uncased-sentiment/resolve/fad9b372f770c208d2efaf00d02f2cb3b61ec450"
TWITTER = f"{HF}/Xenova/twitter-roberta-base-sentiment-latest/resolve/f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445"

# 모델 이름 → [(URL, 저장 파일명, SHA256), ...]  (파일명은 app/model.py 의 MODEL_SPECS 와 같아야 함)
MODELS: dict[str, list[tuple[str, str, str]]] = {
    "distilbert-sst2": [
        (f"{SST2}/onnx/model.onnx", "distilbert-sst2.onnx",
         "252cf7048af94a1599019fef35961b2bd3d6db13df0b0a4b032b92baeae31939"),
        (f"{SST2}/onnx/tokenizer.json", "distilbert-sst2.tokenizer.json",
         "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    ],
    "nlptown-stars": [
        (f"{NLPTOWN}/onnx/model_quantized.onnx", "nlptown-stars-int8.onnx",
         "aefeb9b1f7e84d525d0a91f9f06d0d12bb7270186194a421436b728d6be55643"),
        (f"{NLPTOWN}/tokenizer.json", "nlptown-stars.tokenizer.json",
         "11aaf894a4ccf3d95e8830e27c0f8152791fbbff2b988e29a265580b86edd216"),
    ],
    "twitter-roberta": [
        (f"{TWITTER}/onnx/model_quantized.onnx", "twitter-roberta-int8.onnx",
         "046f7e4cc46b399558fa9b2de966f6b0d42c69e4b01f582bfb4099b01a0bb5d7"),
        (f"{TWITTER}/tokenizer.json", "twitter-roberta.tokenizer.json",
         "1e6506713f00e34406a757acb80f9f3233c1c3950857d32bbc41bcd419d5d8b6"),
    ],
}
DEFAULT_MODEL = "nlptown-stars"  # app/model.py 의 DEFAULT_MODEL 과 같아야 함
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, filename: str, expected: str) -> bool:
    target = MODELS_DIR / filename
    if target.exists() and sha256(target) == expected:
        print(f"이미 준비됨: {target.name}")
        return True
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            print(f"다운로드 중: {url}")
            urllib.request.urlretrieve(url, tmp)
            break
        except Exception as exc:  # 일시적인 네트워크 오류는 재시도
            if attempt == 3:
                print(f"다운로드 실패: {exc}", file=sys.stderr)
                return False
            time.sleep(5 * (attempt + 1))
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        print(f"SHA256 불일치 ({filename}): {actual}", file=sys.stderr)
        return False
    tmp.replace(target)
    print(f"완료: {target.name} ({target.stat().st_size / 1e6:.1f} MB, SHA256 OK)")
    return True


def main(argv: list[str]) -> int:
    names = list(MODELS) if "--all" in argv else [a for a in argv if not a.startswith("-")]
    names = names or [os.getenv("MODEL_NAME") or DEFAULT_MODEL]
    # 전처리만 다른 설정(예: nlptown-stars-headtail)은 같은 파일을 쓰므로 기본 이름으로 바꿈
    names = [next((b for b in MODELS if n.startswith(b)), n) for n in names]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"알 수 없는 모델: {unknown} (가능: {list(MODELS)})", file=sys.stderr)
        return 1
    return 0 if all(fetch(*f) for n in names for f in MODELS[n]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
