"""모델 비교용 공개 평가 데이터 100건을 내려받습니다.

- 출처: Hugging Face `fancyzhx/amazon_polarity` (Apache License 2.0)
  아마존 상품 리뷰(제목+본문)와 긍정/부정 라벨. test split(40만 건)에서만 뽑음.
- 고정 시드로 100건을 무작위로 뽑아 30건(tune: 개선 방법 선택용) / 70건(test: 최종 평가 전용)으로 분리.
- 결과는 data/eval/eval.jsonl 에 저장되며 Git에는 올리지 않습니다(.gitignore).
- datasets-server 는 요청 한도(429)가 잦으므로 요청 간격 1초 + 지수 백오프 + 받은 행 캐시를 씁니다.

사용법:  python scripts/prepare_eval_data.py
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval"
OUT = OUT_DIR / "eval.jsonl"
ROW_CACHE = OUT_DIR / "rows_cache.json"

DATASET = "fancyzhx/amazon_polarity"
CONFIG, SPLIT, N_ROWS = "amazon_polarity", "test", 400_000
ROWS_API = "https://datasets-server.huggingface.co/rows"
LABELS = ["negative", "positive"]
SEED = 2026
N_TOTAL, N_TUNE = 100, 30


def review_text(title: str, content: str) -> str:
    """서비스에 넣을 리뷰 전체 문장 = 제목 + 본문."""
    title = title.strip()
    if title and title[-1] not in ".!?":
        title += "."
    return f"{title} {content.strip()}".strip()


def fetch_row(offset: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": offset, "length": 1})
    for attempt in range(8):
        try:
            time.sleep(1.0)
            with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=30) as res:
                return json.load(res)["rows"][0]["row"]
        except Exception as exc:  # 429 등 일시적 오류 → 지수 백오프 후 재시도
            if attempt == 7:
                raise
            wait = min(120, 5 * 2 ** attempt)
            print(f"  재시도 offset={offset} ({exc}) - {wait}초 대기", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> int:
    rng = random.Random(SEED)
    offsets = rng.sample(range(N_ROWS), N_TOTAL)
    splits = ["tune"] * N_TUNE + ["test"] * (N_TOTAL - N_TUNE)  # 표본 자체가 무작위 순서

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(ROW_CACHE.read_text(encoding="utf-8")) if ROW_CACHE.exists() else {}
    rows = []
    for i, (offset, split) in enumerate(zip(offsets, splits), 1):
        key = str(offset)
        if key not in cache:
            cache[key] = fetch_row(offset)
            tmp = ROW_CACHE.with_name(ROW_CACHE.name + ".part")
            tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(ROW_CACHE)
        r = cache[key]
        rows.append({"id": f"ap-test-{offset:06d}", "offset": offset, "split": split,
                     "label": LABELS[int(r["label"])], "text": review_text(r["title"], r["content"])})
        print(f"[{i:3d}/{N_TOTAL}] {split:4s} {offset:6d} {rows[-1]['label']:8s} {rows[-1]['text'][:60]}", flush=True)

    # 검증: 100건, 중복 없음, 라벨·텍스트 정상
    assert len(rows) == N_TOTAL and len({r["offset"] for r in rows}) == N_TOTAL
    assert all(r["label"] in LABELS and len(r["text"]) > 10 for r in rows)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("tune", "test"):
        part = [r for r in rows if r["split"] == s]
        pos = sum(r["label"] == "positive" for r in part)
        print(f"{s}: {len(part)}건 (긍정 {pos} / 부정 {len(part) - pos})")
    print(f"완료: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
