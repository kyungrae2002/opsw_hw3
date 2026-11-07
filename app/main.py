"""ReviewMood - 영어 리뷰 감성(긍정/부정) 분석 웹 서비스 (FastAPI)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model import MODEL_SPECS, InvalidTextError, SentimentModel

APP_VERSION = "1.1.0"
MAX_CHARS = 5000
STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reviewmood")

app = FastAPI(
    title="ReviewMood",
    version=APP_VERSION,
    description="영어 리뷰 문장을 넣으면 사전학습 모델이 긍정/부정을 분석합니다.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_model: SentimentModel | None = None
_load_error: str | None = None


def get_model() -> SentimentModel:
    """모델을 처음 요청 시 한 번만 로드 (lazy loading)."""
    global _model, _load_error
    if _model is None:
        try:
            _model = SentimentModel()
            _load_error = None
            logger.info("model loaded: %s", _model.model_name)
        except (FileNotFoundError, ValueError) as exc:
            _load_error = str(exc)
            raise HTTPException(status_code=503, detail=_load_error) from exc
    return _model


def deploy_info() -> dict:
    commit = os.getenv("GIT_COMMIT") or "local"
    return {"version": APP_VERSION, "commit": commit[:7]}


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        description=f"분석할 영어 리뷰 (최대 {MAX_CHARS}자)",
        examples=["The battery lasts all day and the screen is gorgeous. Totally worth the price!"],
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    try:
        m = get_model()
        model = {"loaded": True, "name": m.model_name, "source": m.spec.source}
    except HTTPException:
        model = {"loaded": False, "error": _load_error}
    return {"status": "ok", "model": model, **deploy_info()}


@app.get("/api/models")
def models() -> dict:
    """등록된 모델 설정 목록 (실험용 포함)."""
    return {name: {"source": s.source, "output": s.output, "neutral": s.neutral}
            for name, s in MODEL_SPECS.items()}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    text = req.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="분석할 문장을 입력해 주세요.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"문장이 너무 깁니다. {MAX_CHARS}자 이하로 입력해 주세요.")

    m = get_model()
    started = time.perf_counter()
    try:
        result = m.analyze(text)
    except InvalidTextError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info("analyze %d chars -> %s %.3f (%.1fms)", len(text), result["label"], result["score"], elapsed_ms)

    return {
        **result,
        "chars": len(text),
        "model": m.model_name,
        "inference_ms": elapsed_ms,
        "notice": "사전학습 모델의 추정치이며 정확하지 않을 수 있습니다. 영어 문장에 맞춰진 모델입니다.",
        **deploy_info(),  # 응답한 서버의 버전·커밋 (무중단 배포 검증용)
    }
