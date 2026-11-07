# 모델 업데이트 실험 리포트 (ReviewMood v1.0.0 → v1.1.0)

## 1. 요약

| 항목 | 기존 v1.0.0 | 개선 v1.1.0 | 변화 |
|---|---|---|---|
| 모델 | `distilbert-sst2` (영화 리뷰 문장 SST-2 학습) | `nlptown-stars` (상품 리뷰 별점 1~5 학습) | 사전학습 모델 교체 |
| **test 정확도 (70건)** | **90.0% (63/70)** | **94.3% (66/70)** | **+4.3%p (+3건)** |
| test macro-F1 | 0.899 | 0.943 | +0.044 |
| test 긍정 재현율 / 부정 재현율 | 90.6% / 89.5% | 96.9% / 92.1% | +6.3%p / +2.6%p |
| 평균 추론 시간 (로컬 M 시리즈 CPU) | 11.1 ms | 19.7 ms | 약 1.8배 느림 |
| 모델 파일 크기 | 268 MB (FP32) | 169 MB (INT8) | 이미지 약 100 MB 감소 |

- 대규모 재학습 없이 **같은 과제의 다른 사전학습 모델로 교체**해서 정확도를 올렸습니다.
- 개선 방법은 **tune 30건만 보고 선택**했고, test 70건은 선택이 끝난 뒤 **한 번만** 평가했습니다.
- test 개선 폭(+3건)은 표본이 작아 통계적으로 확실하다고 말하기 어렵습니다(4절 한계 참고).

## 2. 평가 데이터

| 항목 | 내용 |
|---|---|
| 출처 | Hugging Face [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) test split (40만 건) |
| 이용 조건 | **Apache License 2.0** (데이터셋 카드 기준) |
| 라벨 | 원래 별점 1~2점 = negative, 4~5점 = positive (**3점 리뷰는 데이터셋에서 제외됨**) |
| 입력 문장 | `제목. 본문` (서비스에 리뷰 전체를 붙여 넣는 상황과 같게) |
| 표본 | 고정 시드(2026)로 무작위 100건 → **tune 30건**(긍정 17/부정 13), **test 70건**(긍정 32/부정 38) |
| 길이 | 평균 약 100토큰, 최대 247토큰 → 모델 최대 길이(512) 안이라 **잘린 리뷰 0건** |
| 재현 | `python scripts/prepare_eval_data.py` → `python scripts/evaluate.py --split tune|test <모델...>` |

평가 데이터와 결과 원본(`data/`, `reports/eval_*.json`)은 Git에 올리지 않았습니다.

## 3. 실험 과정

### 3.1 기준 모델 오류 분석 (tune 30건)

기준 모델 `distilbert-sst2`: **tune 정확도 73.3% (22/30)**, 오답 8건

| 유형 | 건수 | 예시 (정답 → 예측, 기존 P(긍정)) |
|---|---|---|
| **장점과 단점이 섞인 긍정 리뷰를 부정으로 판정** | 5 | "Great vacuum pump. When I first hooked up … would not register …" (긍정 → 부정, 0.01) / "Convenient, pure, but costly …" (긍정 → 부정, 0.00) |
| 비꼬거나 돌려 말한 부정 리뷰를 긍정으로 판정 | 2 | "Ho-hum. Contrary to popular opinion … derivative rather than original" (부정 → 긍정, 0.64) |
| 라벨이 의심스러운 리뷰 | 1 | "couldn't use card b/c camera was a piece of garbage … Customer service was AWFUL!" (라벨은 긍정) |

관찰:
- 오답 대부분이 **문장 하나의 부정 표현에 끌려 확신도 99% 이상으로 틀리는** 경우였습니다. SST-2는 영화 리뷰의 *짧은 문장* 단위 데이터라, 여러 문장으로 장단점을 함께 쓰는 상품 리뷰와 성격이 다릅니다(도메인 차이).
- 길이 초과로 잘린 리뷰는 0건이어서 "긴 리뷰 잘림 방식 개선" 같은 전처리는 효과가 없다고 판단했습니다.

### 3.2 개선 시도 (최대 2회, tune 30건으로만 판단)

| 시도 | 내용 | tune 정확도 | 판단 |
|---|---|---|---|
| 기준 | `distilbert-sst2` | 73.3% (22/30) | - |
| **시도 1** | **모델 교체: `nlptown-stars`** (상품 리뷰 별점 1~5로 학습한 BERT, MIT). 별점 확률을 1~2점=부정, 4~5점=긍정, 3점=반반으로 합산 | **90.0% (27/30)** | 기준 오답 8건 중 5건 고침, 새로 틀린 것 0건 → **채택** |
| 시도 2 | 시도 1 + 후처리 변경 `nlptown-stars-polar`: 데이터셋에 3점 리뷰가 없다는 점을 반영해 3점 확률을 빼고 1~2점 vs 4~5점만 비교 | 90.0% (27/30) | 시도 1과 같음 → 복잡도만 늘어 **미채택** |

- 시도 1을 고른 이유: 오류의 주 원인이 도메인 차이(영화 문장 vs 상품 리뷰)였으므로, 상품 리뷰 별점으로 학습한 모델이 "장점과 단점이 섞인 리뷰"를 별점 4점 쪽으로 볼 것이라고 예상했습니다.
- 두 시도의 tune 결과가 같아서 더 단순한 시도 1을 **test를 보기 전에 최종 모델로 확정**했습니다.
- 그 밖에 준비해 둔 `twitter-roberta`(트윗 3클래스)는 시도 횟수 제한(2회) 때문에 평가하지 않았습니다.

### 3.3 최종 비교 (test 70건, 선택 이후 1회 평가)

| 모델 | 정확도 | macro-F1 | 긍정 재현율 | 부정 재현율 |
|---|---|---|---|---|
| 기준 `distilbert-sst2` | 90.0% (63/70) | 0.899 | 90.6% | 89.5% |
| **채택 `nlptown-stars`** | **94.3% (66/70)** | **0.943** | **96.9%** | **92.1%** |
| (참고) 시도 2 `nlptown-stars-polar` | 94.3% (66/70) | 0.943 | 96.9% | 92.1% |

리뷰별 변화 (test, 기준과 개선 모델 중 하나라도 틀린 9건):

| 리뷰 | 정답 | 기준 (P긍정) | 개선 (P긍정) | 변화 |
|---|---|---|---|---|
| "SIMPLEMENTE LO MEJOR. En este album …" (스페인어) | 긍정 | 부정 (0.04) | 긍정 (0.99) | ✅ 고쳐짐 (다국어 모델) |
| "Not for the Uninitiated. … useful video for people who are already …" | 부정 | 긍정 (0.92) | 부정 (0.45) | ✅ 고쳐짐 |
| "Dart (seam) shows through clothes. …" | 부정 | 긍정 (0.53) | 부정 (0.46) | ✅ 고쳐짐 |
| "the Warlocks of the kafir. …" | 긍정 | 부정 (0.06) | 긍정 (0.72) | ✅ 고쳐짐 |
| "Powder works great, NOT pills. …" | 긍정 | 부정 (0.04) | 긍정 (0.77) | ✅ 고쳐짐 (장단점 혼합) |
| "this book is fine. oh i wanted to LOVE this book-- it is FINE--but not amazing" | 부정 | 긍정 (1.00) | 긍정 (0.69) | 둘 다 틀림 |
| "Boooorrrrring. I was given this book as a gift …" | 부정 | 긍정 (0.90) | 긍정 (0.72) | 둘 다 틀림 |
| "Are you qualified? The premise here is …" | 부정 | 부정 (0.16) | 긍정 (0.50) | ❌ 새로 틀림 (경계값) |
| "All Killer, No Filler. While the insanely beautiful writing …" | 긍정 | 긍정 (1.00) | 부정 (0.29) | ❌ 새로 틀림 |

→ 고쳐진 5건, 새로 틀린 2건, 둘 다 틀린 2건. 순증가 +3건.

## 4. 한계 (정직하게 적음)

- **표본이 작습니다.** test 70건에서 1건 = 1.4%p입니다. 고쳐진 5건 vs 새로 틀린 2건을 McNemar 정확 검정으로 보면 p ≈ 0.45로, "개선 모델이 확실히 더 낫다"고 통계적으로 말할 수는 없습니다. 방향은 tune(+16.7%p)과 test(+4.3%p) 모두 같았습니다.
- **tune과 test의 개선 폭 차이가 큽니다.** tune에서 기준 모델이 유독 낮았던(73.3%) 영향이 있고, test의 기준 모델은 이미 90.0%였습니다. test를 본 뒤 다시 조정하지는 않았습니다(규칙 준수).
- **정밀도가 다릅니다.** 기준 모델은 FP32, 개선 모델은 INT8 양자화 ONNX입니다(용량 때문). 양자화가 정확도에 준 영향은 따로 분리하지 않았습니다.
- **속도가 느려졌습니다.** 리뷰 1건 평균 11 ms → 20 ms (BERT-base 12층 vs DistilBERT 6층). 서비스 응답에는 지장 없는 수준입니다.
- **라벨 노이즈**: tune 1건은 내용상 명백한 부정인데 긍정 라벨이라 어떤 모델도 맞히기 어렵습니다.
- **경계값**: 개선 모델이 P(긍정)=0.50 근처에서 판정하는 리뷰가 있습니다(별점 3점 쪽으로 확률이 몰리는 애매한 리뷰).
- 아마존 상품 리뷰 기준 결과이며, 다른 종류의 글(트윗, 뉴스 등)에서도 같은 결과가 나온다고 보장하지 않습니다.

## 5. 배포 반영

- `app/model.py`, `scripts/download_model.py` 의 `DEFAULT_MODEL` 을 `nlptown-stars` 로 변경, 앱 버전 `1.0.0 → 1.1.0`.
- 실행 환경(Python 3.12, 고정 라이브러리 버전), 주요 기능(웹 화면, `/api/analyze`, `/api/health`, `/docs`), 무중단 배포 구성(nginx + blue/green), 접속 주소(`http://localhost:8080`)는 그대로 유지.
- 배포 시 독립 모니터가 배포 전·중·후 1초 간격으로 실제 분석 API를 호출해 무중단 여부를 기록합니다(Actions Summary, Artifacts `zero-downtime-monitor`).

## 6. 사용 모델 출처와 라이선스

| 모델 | 출처 | 라이선스 |
|---|---|---|
| distilbert-sst2 | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (공식 ONNX) | Apache-2.0 |
| nlptown-stars | [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment), ONNX 변환본 [Xenova/bert-base-multilingual-uncased-sentiment](https://huggingface.co/Xenova/bert-base-multilingual-uncased-sentiment) `model_quantized.onnx` | MIT |

모든 파일은 Hugging Face 커밋 해시로 고정된 URL에서 받고 SHA256으로 검증합니다(`scripts/download_model.py`).
