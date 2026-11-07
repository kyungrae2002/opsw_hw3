"""모델 설정별 감성 분석 성능 평가 (정확도, 클래스별 재현율, macro-F1 + 오류 목록).

평가 데이터는 scripts/prepare_eval_data.py 로 준비합니다.
  - tune (30건): 오류 분석과 개선 방법 선택에만 사용
  - test (70건): 최종 비교 전용 - 개선 방법 선택에 사용하지 않음

사용법:
  python scripts/evaluate.py --split tune distilbert-sst2 nlptown-stars
  python scripts/evaluate.py --split test distilbert-sst2 nlptown-stars
결과: reports/eval_<split>.json (요약 지표 + 리뷰별 예측, Git 에는 올리지 않음)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model import SentimentModel  # noqa: E402

DATA = ROOT / "data" / "eval" / "eval.jsonl"
REPORTS = ROOT / "reports"


def evaluate(name: str, rows: list[dict]) -> dict:
    m = SentimentModel(model_name=name)
    items, elapsed = [], 0.0
    for row in rows:
        started = time.perf_counter()
        r = m.analyze(row["text"])
        elapsed += time.perf_counter() - started
        items.append({"id": row["id"], "truth": row["label"], "pred": r["label"],
                      "p_pos": round(r["scores"]["positive"], 4), "correct": r["label"] == row["label"],
                      "tokens": r["tokens"], "truncated": r["truncated"]})
    n = len(items)
    f1s, recalls = [], {}
    for c in ("positive", "negative"):
        tp = sum(i["pred"] == c and i["truth"] == c for i in items)
        fp = sum(i["pred"] == c and i["truth"] != c for i in items)
        fn = sum(i["pred"] != c and i["truth"] == c for i in items)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        recalls[c] = rec
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"model": name, "n": n, "correct": sum(i["correct"] for i in items),
            "accuracy": sum(i["correct"] for i in items) / n, "macro_f1": sum(f1s) / 2,
            "recall_pos": recalls["positive"], "recall_neg": recalls["negative"],
            "avg_ms": round(elapsed / n * 1000, 1), "items": items}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tune", "test"], required=True)
    parser.add_argument("--show-text", action="store_true", help="오답 리뷰 본문 출력 (tune 오류 분석용)")
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == args.split]
    text_by_id = {r["id"]: r["text"] for r in rows}

    results = [evaluate(m, rows) for m in args.models]
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"eval_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "results": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print(f"\n[{args.split}] {len(rows)}건")
    print(f"{'model':28s} {'정확도':>12s} {'macro-F1':>9s} {'긍정재현':>8s} {'부정재현':>8s} {'ms/건':>7s}")
    for r in results:
        print(f"{r['model']:28s} {r['correct']:3d}/{r['n']} {r['accuracy']:6.1%} {r['macro_f1']:9.3f} "
              f"{r['recall_pos']:8.1%} {r['recall_neg']:8.1%} {r['avg_ms']:7.1f}")

    base = results[0]
    wrong = [i for i in base["items"] if not i["correct"]]
    print(f"\n{base['model']} 오답 {len(wrong)}건:")
    for i in wrong:
        fixed = [r["model"] for r in results[1:]
                 if next(x for x in r["items"] if x["id"] == i["id"])["correct"]]
        print(f"  {i['id']}: 정답 {i['truth']} → 예측 {i['pred']} (P(긍정)={i['p_pos']:.2f}, {i['tokens']}토큰)"
              f"{'  ✔ 고쳐짐: ' + ', '.join(fixed) if fixed else ''}")
        if args.show_text:
            print(f"      \"{text_by_id[i['id']][:300]}\"")
    for r in results[1:]:
        broke = [x["id"] for x in r["items"]
                 if not x["correct"] and next(b for b in base["items"] if b["id"] == x["id"])["correct"]]
        if broke:
            print(f"  ✘ {r['model']} 이 새로 틀린 리뷰: {', '.join(broke)}")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
