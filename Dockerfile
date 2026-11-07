FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# 1) 의존성 설치 (간접 의존성까지 정확한 버전으로 고정된 requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) 모델은 Git에 올리지 않고, 이미지 빌드 시 내려받아 SHA256으로 검증
#    (MODEL_NAME 을 비워 두면 app/model.py 의 DEFAULT_MODEL 사용)
ARG MODEL_NAME=
ENV MODEL_NAME=${MODEL_NAME}
COPY scripts/download_model.py scripts/verify_model.py scripts/
RUN python scripts/download_model.py

# 3) 앱 코드 복사 후, 모델이 실제로 로드·추론되는지 빌드 단계에서 검증 (실패하면 빌드 중단)
COPY app/ app/
RUN python scripts/verify_model.py

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# 배포된 커밋을 /api/health 와 분석 응답에 표시
ARG GIT_COMMIT=local
ENV GIT_COMMIT=${GIT_COMMIT} \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health')"

# exec 로 uvicorn 을 PID 1로 만들어 SIGTERM 을 바로 받게 함 (docker stop 이 30초씩 걸리지 않도록)
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
