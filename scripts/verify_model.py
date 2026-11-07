"""이미지 빌드 시 모델이 정상적으로 로드·추론되는지 검증합니다.

사용법:  python scripts/verify_model.py   (실패 시 종료코드 1 → docker build 중단)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import tokenizers  # noqa: E402

from app.model import SentimentModel  # noqa: E402

CHECKS = [
    ("I love it. Works perfectly and arrived early!", "positive"),
    ("Terrible quality. It broke on the first day and I want a refund.", "negative"),
]


def main() -> int:
    m = SentimentModel()
    for text, expected in CHECKS:
        r = m.analyze(text)
        s = r["scores"]
        assert r["label"] in ("positive", "negative"), f"라벨 오류: {r['label']}"
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in s.values()), f"확률 값 오류: {s}"
        assert abs(s["positive"] + s["negative"] - 1) < 1e-6, "확률 합 오류"
        assert abs(sum(d["probability"] for d in r["detail"]) - 1) < 1e-4, "원본 출력 합 오류"
        assert r["label"] == expected, f"명백한 예문을 잘못 분석: {text!r} → {r['label']}"
        print(f"  {expected:8s} → {r['label']} ({r['score']:.3f})  {text}")

    long = m.analyze("good " * 2000)  # 최대 길이를 넘는 입력도 잘라서 처리해야 함
    assert long["truncated"] and long["tokens"] <= m.spec.max_tokens, "긴 입력 처리 오류"

    print(f"모델 검증 OK: {m.model_name} (onnxruntime {ort.__version__}, "
          f"tokenizers {tokenizers.__version__}, numpy {np.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
