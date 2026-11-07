"""무중단 배포 검증용 독립 모니터 (python3 표준 라이브러리만 사용).

배포 스크립트와 별개의 프로세스로 실행되어, 고정 테스트 문장을 실제 분석 API(/api/analyze)에
1초 간격으로 보내고 요청마다 시작·종료 시각, 응답 시간, HTTP 상태, 분석 정상 여부,
응답한 서버 버전(버전/커밋/모델)을 기록합니다. 배포가 끝난 뒤에도 30초 더 요청한 다음 종료합니다.
요청은 스레드로 보내므로 응답이 늦어져도 1초 간격 스케줄은 유지됩니다.

하위 명령:
  run         모니터 실행 (백그라운드로 띄움)
  wait-ready  모니터가 기존 서비스의 정상 응답을 확인할 때까지 대기
  mark        배포 시작/종료 시각 기록
  report      모니터 종료를 기다린 뒤 전체 로그(UTF-8)와 요약(GitHub Step Summary) 생성
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
UNVERIFIED_GAP_SEC = 2.0  # 정상 응답 기록 사이가 이보다 길면 "검증되지 않은 구간"
PHASES = ("배포 전", "배포 중", "배포 후")


def ts(epoch: float, ms: bool = True) -> str:
    s = datetime.fromtimestamp(epoch, KST).strftime("%H:%M:%S.%f" if ms else "%H:%M:%S")
    return s[:-3] if ms else s


def full_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, KST).strftime("%Y-%m-%d %H:%M:%S KST")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- run
def post_json(url: str, payload: bytes, timeout: float) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read(), ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), f"HTTP {exc.code}"
    except Exception as exc:  # 연결 거부, 타임아웃 등
        return 0, b"", f"{type(exc).__name__}: {exc}"[:200]


class Monitor:
    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.out)
        self.payload = Path(args.input).read_bytes()
        json.loads(self.payload)  # 입력 파일 형식 확인
        self.lock = threading.Lock()
        self.log_f = (self.out / "requests.log").open("a", encoding="utf-8")
        self.jsonl_f = (self.out / "requests.jsonl").open("a", encoding="utf-8")
        self.ok_seen = False

    def one_request(self, seq: int) -> None:
        start = time.time()
        status, body, error = post_json(f"{self.args.url}/api/analyze", self.payload, self.args.timeout)
        end = time.time()
        rec = {"seq": seq, "start": start, "end": end, "latency_ms": round((end - start) * 1000, 1),
               "status": status, "ok": False, "version": "", "commit": "", "model": "",
               "label": "", "score": None, "error": error}
        if status == 200:
            try:
                d = json.loads(body)
                rec.update(version=d.get("version", ""), commit=d.get("commit", ""), model=d.get("model", ""),
                           label=d.get("label", ""), score=round(float(d.get("score", -1)), 4))
                rec["ok"] = (d.get("label") in ("positive", "negative") and 0.5 <= rec["score"] <= 1
                             and bool(rec["version"]) and bool(rec["commit"]))
                if not rec["ok"]:
                    rec["error"] = "분석 결과 형식 오류"
            except Exception as exc:
                rec["error"] = f"응답 파싱 실패: {exc}"
        line = (f"#{seq:04d} start={ts(start)} end={ts(end)} {rec['latency_ms']:7.1f}ms "
                f"HTTP {status:3d} {'OK  ' if rec['ok'] else 'FAIL'} "
                f"ver={rec['version']}/{rec['commit']}/{rec['model']} {rec['label']} {rec['error']}")
        with self.lock:
            self.log_f.write(line + "\n")
            self.log_f.flush()
            self.jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.jsonl_f.flush()
            if rec["ok"] and not self.ok_seen:
                self.ok_seen = True
                (self.out / "ready").write_text(str(end), encoding="utf-8")
        print(line, flush=True)

    def run(self) -> int:
        a = self.args
        (self.out / "monitor.json").write_text(json.dumps(
            {"pid": os.getpid(), "url": a.url, "input": a.input, "interval": a.interval,
             "post_seconds": a.post_seconds, "started": time.time()}, ensure_ascii=False), encoding="utf-8")
        threads: list[threading.Thread] = []
        t0 = time.time()
        seq = 0
        while True:
            due = t0 + seq * a.interval  # 절대 시각 기준 스케줄 (밀림 누적 방지)
            time.sleep(max(0.0, due - time.time()))
            seq += 1
            t = threading.Thread(target=self.one_request, args=(seq,), daemon=True)
            t.start()
            threads.append(t)
            end_mark = read_json(self.out / "deploy_end.json")
            if end_mark and time.time() >= end_mark["at"] + a.post_seconds:
                break
            if time.time() - t0 > a.max_seconds:
                print("최대 실행 시간 초과 → 모니터 종료", flush=True)
                break
        for t in threads:
            t.join(timeout=a.timeout + 1)
        (self.out / "monitor_done").write_text(str(time.time()), encoding="utf-8")
        return 0


def cmd_run(args) -> int:
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return Monitor(args).run()


# ---------------------------------------------------------------- helpers
def cmd_wait_ready(args) -> int:
    out = Path(args.out)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if (out / "ready").exists():
            print("모니터: 기존 서비스의 정상 응답 확인 → 배포 시작")
            return 0
        time.sleep(0.5)
    print("::warning::모니터가 정상 응답을 확인하지 못했습니다 (실행 중인 서비스 없음 = 첫 배포?). 배포는 계속 진행합니다.")
    (out / "no_baseline").write_text("1", encoding="utf-8")
    return 0


def cmd_mark(args) -> int:
    out = Path(args.out)
    data: dict = {"at": time.time()}
    if args.result is not None:
        data["result"] = args.result
    (out / f"{args.event}.json").write_text(json.dumps(data), encoding="utf-8")
    print(f"{args.event}: {full_ts(data['at'])}" + (f" (exit={args.result})" if args.result is not None else ""))
    return 0


# ---------------------------------------------------------------- report
def load_records(out: Path) -> list[dict]:
    path = out / "requests.jsonl"
    if not path.exists():
        return []
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(recs, key=lambda r: r["start"])


def cmd_report(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_json(out / "monitor.json") or {}
    # 배포 단계가 비정상 종료되어 종료 시각이 없으면 지금을 종료로 기록 (모니터가 멈추도록)
    if not (out / "deploy_end.json").exists():
        (out / "deploy_end.json").write_text(json.dumps({"at": time.time(), "result": "unknown"}), encoding="utf-8")
    end_mark = read_json(out / "deploy_end.json") or {"at": time.time()}
    deadline = time.time() + meta.get("post_seconds", 30) + 60
    while meta and not (out / "monitor_done").exists() and time.time() < deadline:
        time.sleep(1)

    recs = load_records(out)
    start_mark = read_json(out / "deploy_start.json")
    d_start = start_mark["at"] if start_mark else None
    d_end = end_mark["at"]
    deploy_result = end_mark.get("result")
    deploy_ok = str(deploy_result) == "0"
    expected = args.expected_commit[:7]

    for r in recs:
        if d_start is None or r["start"] < d_start:
            r["phase"] = "배포 전"
        else:
            r["phase"] = "배포 중" if r["start"] <= d_end else "배포 후"

    # 전체 로그 (UTF-8 CSV, 엑셀에서 한글이 깨지지 않도록 BOM 포함) - Artifacts 업로드용
    with (out / "requests.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seq", "단계", "시작", "종료", "응답시간_ms", "HTTP", "분석정상", "버전", "커밋", "모델",
                    "결과", "확신도", "오류"])
        for r in recs:
            w.writerow([r["seq"], r["phase"], ts(r["start"]), ts(r["end"]), r["latency_ms"], r["status"],
                        "Y" if r["ok"] else "N", r["version"], r["commit"], r["model"], r["label"],
                        r["score"], r["error"]])

    print("::group::전체 요청 기록")
    for r in recs:
        print(f"#{r['seq']:04d} [{r['phase']}] {ts(r['start'])} → {ts(r['end'])} {r['latency_ms']:7.1f}ms "
              f"HTTP {r['status']} {'OK' if r['ok'] else 'FAIL'} {r['version']}/{r['commit']}/{r['model']} "
              f"{r['label']} {r['error']}")
    print("::endgroup::")

    total = len(recs)
    ok = sum(r["ok"] for r in recs)
    fail = total - ok
    by_phase = {p: [r for r in recs if r["phase"] == p] for p in PHASES}
    starts = [r["start"] for r in recs]
    intervals = [b - a for a, b in zip(starts, starts[1:])]
    lat = [r["latency_ms"] for r in recs if r["ok"]]
    ok_ends = sorted(r["end"] for r in recs if r["ok"])
    gaps = [(a, b, b - a) for a, b in zip(ok_ends, ok_ends[1:])]
    max_gap = max(gaps, key=lambda g: g[2]) if gaps else None
    unverified = [g for g in gaps if g[2] > UNVERIFIED_GAP_SEC]

    versions: list[tuple[tuple[str, str, str], dict]] = []
    for r in sorted((r for r in recs if r["ok"]), key=lambda r: r["end"]):
        key = (r["version"], r["commit"], r["model"])
        if not versions or versions[-1][0] != key:
            versions.append((key, r))
    final_commit = versions[-1][0][1] if versions else ""
    switched = final_commit == expected
    no_baseline = (out / "no_baseline").exists()

    def vtxt(k):
        return f"v{k[0]} `{k[1]}` {k[2]}"

    L = [
        f"## 🛡️ 무중단 배포 검증 — {args.title}",
        "",
        "| 항목 | 내용 |",
        "|---|---|",
        f"| 검증 주소 | `POST {meta.get('url', '?')}/api/analyze` · 고정 입력 `{Path(meta.get('input', '')).name}` · "
        f"{meta.get('interval', 1):g}초 간격 · 배포 후 {meta.get('post_seconds', 30):g}초 추가 |",
    ]
    if recs:
        L.append(f"| 검증 시간 | {full_ts(recs[0]['start'])} ~ {ts(recs[-1]['end'], ms=False)} "
                 f"(총 {recs[-1]['end'] - recs[0]['start']:.0f}초) |")
    if d_start:
        L.append(f"| 배포 구간 | {ts(d_start, ms=False)} ~ {ts(d_end, ms=False)} ({d_end - d_start:.1f}초) · "
                 f"{'✅ 배포 성공' if deploy_ok else f'❌ 배포 실패 (exit={deploy_result})'} |")
    if versions:
        first_new = next((r for k, r in versions if k[1] == expected), None)
        chain = " → ".join(vtxt(k) for k, _ in versions)
        L.append(f"| 버전 전환 | {chain}" + (f" (새 버전 첫 응답 {ts(first_new['end'])})" if first_new else "")
                 + f" · 기대 커밋 `{expected}` {'✅' if switched else '❌'} |")
    L.append(f"| 요청 결과 | **총 {total}건 · 성공 {ok} · 실패 {fail}** "
             f"(배포 전 {len(by_phase['배포 전'])} / 중 {len(by_phase['배포 중'])} / 후 {len(by_phase['배포 후'])}) |")
    if intervals:
        L.append(f"| 실제 요청 간격 | 평균 {sum(intervals) / len(intervals):.3f}초 · 최소 {min(intervals):.3f} · "
                 f"최대 {max(intervals):.3f}초" + (f" · 응답시간 평균 {sum(lat) / len(lat):.0f}ms, 최대 {max(lat):.0f}ms" if lat else "") + " |")
    if max_gap:
        L.append(f"| 최대 기록 공백 | {max_gap[2]:.3f}초 ({ts(max_gap[0])} ~ {ts(max_gap[1])}, 연속한 정상 응답 사이) |")
    if unverified:
        spans = ", ".join(f"{ts(a)}~{ts(b)} ({g:.1f}초)" for a, b, g in unverified)
        L.append(f"| ⚠️ 검증되지 않은 구간 | {spans} — {UNVERIFIED_GAP_SEC:g}초 넘게 정상 응답 기록이 없음 |")
    else:
        L.append(f"| 검증되지 않은 구간 | 없음 (모든 정상 응답 사이 간격 ≤ {UNVERIFIED_GAP_SEC:g}초) |")

    L += ["", "**배포 전·중·후 대표 응답**", "",
          "| 단계 | # | 요청 시각 | HTTP | 응답시간 | 분석 | 응답 버전 |", "|---|---|---|---|---|---|---|"]
    for p, rs in by_phase.items():
        if not rs:
            L.append(f"| {p} | - | - | - | - | - | - |")
            continue
        picks = [rs[0]]
        if p == "배포 중":
            picks += [r for r in rs if not r["ok"]][:3]
            # 버전이 바뀐 직후 응답
            picks += [b for a, b in zip(rs, rs[1:]) if a["commit"] and b["commit"] and a["commit"] != b["commit"]][:1]
        picks.append(rs[-1])
        seen = set()
        for r in sorted(picks, key=lambda r: r["start"]):
            if r["seq"] in seen:
                continue
            seen.add(r["seq"])
            res = f"✅ {r['label']} {r['score']:.3f}" if r["ok"] else f"❌ {r['error'][:40]}"
            ver = f"v{r['version']} `{r['commit']}` {r['model']}" if r["version"] else "-"
            L.append(f"| {p} | {r['seq']} | {ts(r['start'])} | {r['status']} | {r['latency_ms']:.0f}ms | {res} | {ver} |")

    L.append("")
    passed = deploy_ok and fail == 0 and switched and total > 0 and not no_baseline and not unverified
    gap_txt = f"{max_gap[2]:.2f}초" if max_gap else "-"
    if no_baseline:
        L.append("> ℹ️ 배포 전에 실행 중인 서비스가 없어(첫 배포) 무중단 여부는 판단하지 않았습니다. "
                 "무중단 검증은 서비스가 떠 있는 상태의 다음 배포부터 나옵니다.")
    elif passed:
        L.append(f"> ✅ **관측 범위({ts(recs[0]['start'], ms=False)} ~ {ts(recs[-1]['end'], ms=False)})에서 실제 분석 요청 "
                 f"{total}건이 모두 정상 응답했고, 응답 버전이 `{versions[0][0][1]}` → `{final_commit}` 으로 전환되었습니다. "
                 f"이 범위에서는 서비스 중단 없이 재배포되었습니다.** "
                 f"단, 1초 간격 표본이므로 요청 사이(최대 기록 공백 {gap_txt})보다 짧은 순간의 중단까지 증명하지는 못합니다.")
    else:
        reasons = []
        if not deploy_ok:
            reasons.append(f"배포 실패(exit={deploy_result})")
        if fail:
            reasons.append(f"실패 요청 {fail}건")
        if not switched:
            reasons.append(f"새 버전 `{expected}` 미확인(최종 `{final_commit or '-'}`)")
        if unverified:
            reasons.append(f"검증되지 않은 구간 {len(unverified)}개")
        if not total:
            reasons.append("기록 없음")
        L.append(f"> ❌ **검증 실패: {', '.join(reasons)}** — 전체 기록은 Artifacts 의 `requests.csv` 참고")
    L += ["", "전체 요청 기록(UTF-8): Artifacts → `zero-downtime-monitor` (requests.csv · requests.log · requests.jsonl · summary.md)"]

    summary = "\n".join(L) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8")
    target = os.getenv("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(summary)
    print(summary)
    if no_baseline:
        return 0 if deploy_ok and switched else 1
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--input", required=True, help="고정 테스트 입력 JSON 파일")
    r.add_argument("--out", required=True)
    r.add_argument("--interval", type=float, default=1.0)
    r.add_argument("--post-seconds", type=float, default=30.0)
    r.add_argument("--timeout", type=float, default=10.0)
    r.add_argument("--max-seconds", type=float, default=1200.0)
    w = sub.add_parser("wait-ready")
    w.add_argument("--out", required=True)
    w.add_argument("--timeout", type=float, default=30.0)
    m = sub.add_parser("mark")
    m.add_argument("--out", required=True)
    m.add_argument("event", choices=["deploy_start", "deploy_end"])
    m.add_argument("--result")
    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    rp.add_argument("--expected-commit", required=True)
    rp.add_argument("--title", default="ReviewMood")
    args = p.parse_args()
    return {"run": cmd_run, "wait-ready": cmd_wait_ready, "mark": cmd_mark, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
