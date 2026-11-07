#!/usr/bin/env bash
# Blue-Green 무중단 배포 스크립트 (self-hosted runner 에서 실행, macOS 기본 bash 3.2 호환)
#
#   [사용자] → localhost:8080 → nginx(reviewmood-proxy) → reviewmood-blue 또는 reviewmood-green
#
# 1) 새 이미지를 받아 "대기 중인 색" 컨테이너로 실행 (외부 포트 없이 내부 네트워크에만 연결)
# 2) 새 컨테이너에서 모델이 로드될 때까지 헬스체크 (실패 시 새 컨테이너만 지우고 기존 서비스 유지)
# 3) nginx 설정을 새 컨테이너로 바꾸고 graceful reload (진행 중인 요청은 끊기지 않음)
# 4) 프록시를 통해 새 버전 응답이 확인되면, 5초 뒤 기존 컨테이너 종료
#
# 사용법: deploy/deploy.sh <이미지:태그>
#   DEPLOY_PORT (기본 8080), DEPLOY_NAME (기본 reviewmood) 로 포트·이름을 바꿀 수 있음 (로컬 리허설용)
set -euo pipefail

IMAGE="${1:?사용법: deploy/deploy.sh <이미지:태그>}"
PORT="${DEPLOY_PORT:-8080}"
APP="${DEPLOY_NAME:-reviewmood}"
NET="$APP-net"
PROXY="$APP-proxy"
NGINX_IMAGE=nginx:1.27.5-alpine

log() { echo "[deploy $(date +%H:%M:%S)] $*"; }

log "이미지 받는 중: $IMAGE"
docker pull "$IMAGE" || docker image inspect "$IMAGE" >/dev/null  # 로컬 리허설용 이미지 허용

# 새 이미지가 응답할 커밋 (프록시 전환 확인용)
NEW_COMMIT=$(docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^GIT_COMMIT=' | head -1 | cut -d= -f2 | cut -c1-7)

docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null

# 프록시(nginx)가 없으면 생성 - 외부 포트는 프록시만 열어 둠
if ! docker ps --format '{{.Names}}' | grep -qx "$PROXY"; then
  docker rm -f "$PROXY" >/dev/null 2>&1 || true
  log "프록시 생성: localhost:$PORT"
  docker run -d --name "$PROXY" --network "$NET" -p "$PORT:80" \
    --restart unless-stopped "$NGINX_IMAGE" >/dev/null
fi

# 현재 트래픽을 받는 색 확인 (nginx 설정 기준)
ACTIVE=$(docker exec "$PROXY" cat /etc/nginx/conf.d/default.conf 2>/dev/null \
  | grep -o "$APP-\(blue\|green\)" | head -1 | sed "s/^$APP-//" || true)
if [ "$ACTIVE" = "blue" ]; then NEW=green; else NEW=blue; fi
log "현재 활성: ${ACTIVE:-없음} → 새 컨테이너: $APP-$NEW (커밋 ${NEW_COMMIT:-?})"

docker rm -f "$APP-$NEW" >/dev/null 2>&1 || true
docker run -d --name "$APP-$NEW" --network "$NET" --restart unless-stopped "$IMAGE" >/dev/null

# 새 컨테이너 헬스체크: 모델이 로드되어야 통과
log "헬스체크 대기 중..."
ok=0
for _ in $(seq 1 ${HEALTH_TRIES:-90}); do
  if docker exec "$APP-$NEW" python -c \
    "import urllib.request,sys; b=urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3).read(); sys.exit(0 if b'\"loaded\":true' in b else 1)" \
    >/dev/null 2>&1; then
    ok=1; break
  fi
  sleep 1
done
if [ "$ok" != 1 ]; then
  log "새 컨테이너가 정상 기동하지 않음 → 배포 중단 (기존 서비스 유지)"
  docker logs --tail 50 "$APP-$NEW" || true
  docker rm -f "$APP-$NEW" >/dev/null
  exit 1
fi
log "헬스체크 통과 (모델 로드됨)"

# nginx 전환 (설정 검증 후 graceful reload)
docker exec -i "$PROXY" sh -c 'cat > /etc/nginx/conf.d/default.conf' <<CONF
upstream app_backend {
    server $APP-$NEW:8000;
}
server {
    listen 80;
    client_max_body_size 1m;
    location / {
        proxy_pass http://app_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }
}
CONF
docker exec "$PROXY" nginx -t -q
docker exec "$PROXY" nginx -s reload

# reload 직후 잠깐은 이전 설정이 응답하므로, 프록시를 통해 새 커밋이 응답할 때까지 확인
switched=0
for _ in $(seq 1 40); do
  body=$(curl -fsS -m 3 "http://127.0.0.1:$PORT/api/health" 2>/dev/null || true)
  if echo "$body" | grep -q '"loaded":true' && { [ -z "$NEW_COMMIT" ] || echo "$body" | grep -q "\"commit\":\"$NEW_COMMIT\""; }; then
    switched=1; break
  fi
  sleep 0.5
done
if [ "$switched" != 1 ]; then
  log "오류: 프록시를 통한 새 버전 응답을 20초 안에 확인하지 못함 (기존 컨테이너는 남겨 둠)"
  exit 1
fi
log "트래픽 전환 완료: $APP-$NEW"

# 기존 컨테이너는 진행 중인 요청이 끝나도록 잠시 기다린 뒤 종료
if [ -n "$ACTIVE" ]; then
  sleep 5
  docker stop -t 30 "$APP-$ACTIVE" >/dev/null 2>&1 || true
  docker rm "$APP-$ACTIVE" >/dev/null 2>&1 || true
  log "기존 컨테이너 종료: $APP-$ACTIVE"
fi

docker image prune -f >/dev/null 2>&1 || true
log "배포 완료 → http://localhost:$PORT"
