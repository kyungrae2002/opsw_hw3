# ReviewMood — 리뷰 감성 분석 웹 서비스

영어 리뷰 문장을 넣으면 사전학습 모델(ONNX)이 **긍정/부정**을 분석해 주는 FastAPI 서비스입니다.
GitHub Actions 로 테스트 → Docker 이미지 빌드·푸시 → self-hosted runner 에서 **무중단(Blue-Green) 배포**까지 자동으로 진행합니다.

- 웹 화면: `http://localhost:8080`
- API 문서(Swagger): `http://localhost:8080/docs` → `POST /api/analyze` → **Try it out**
- 상태 확인: `GET /api/health` (모델 로드 여부, 앱 버전, 커밋)

## API

```bash
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The battery lasts all day. Totally worth the price!"}'
```

응답 예: `label`(positive/negative), `score`(확신도), `scores`, `detail`(모델 원본 출력), `model`, `version`, `commit`

## 모델

| 이름 | 원본 | 라이선스 | 출력 |
|---|---|---|---|
| `distilbert-sst2` (v1.0 기준선) | distilbert/distilbert-base-uncased-finetuned-sst-2-english | Apache-2.0 | 2클래스 |
| **`nlptown-stars` (v1.1 기본값)** | nlptown/bert-base-multilingual-uncased-sentiment (ONNX: Xenova, INT8) | MIT | 별점 1~5 |
| `twitter-roberta` | cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX: Xenova, INT8) | CC BY 4.0 | 3클래스 |

모델 교체 실험(tune 30 / test 70건, test 정확도 90.0% → 94.3%)은 [reports/model_update.md](reports/model_update.md) 참고.

모델 파일은 Git 에 올리지 않고 `scripts/download_model.py` 가 커밋 고정 URL 에서 받아 SHA256 으로 검증합니다.
Docker 빌드 중 `scripts/verify_model.py` 가 실제 추론을 해 보고, 실패하면 빌드가 중단됩니다.

## 구조

```
app/        FastAPI 앱 (main.py, model.py, static/)
scripts/    download_model.py, verify_model.py, prepare_eval_data.py, evaluate.py
deploy/     deploy.sh (blue-green + nginx), monitor.py (독립 모니터), test-input.json
reports/    model_update.md (모델 업데이트 실험 리포트)
tests/      pytest
.github/workflows/ci.yml   test → docker → deploy
```

## 로컬 개발

```bash
python3 -m venv venv && venv/bin/pip install -r requirements-dev.txt
venv/bin/python scripts/download_model.py
venv/bin/pytest
venv/bin/uvicorn app.main:app --port 8001
```

## 무중단 배포

외부 포트(8080)는 nginx 프록시 컨테이너만 열고, 앱은 `reviewmood-blue` / `reviewmood-green` 중 하나가 받습니다.
새 컨테이너가 모델을 로드하면 nginx 를 새 컨테이너로 바꾸고(graceful reload) 기존 컨테이너를 종료합니다.
배포 전부터 `deploy/monitor.py` 가 1초마다 실제 분석 API 를 호출하고, 결과를 Actions Summary 와 Artifacts 에 남깁니다.

로컬 리허설(실제 서비스와 분리된 18080 포트):

```bash
DEPLOY_NAME=rmtest DEPLOY_PORT=18080 bash deploy/deploy.sh <이미지>
```
