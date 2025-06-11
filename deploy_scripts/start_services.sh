#!/bin/bash
# ~/17-newsum-ai/scripts/start_services.sh
# NewSum AI 서비스 시작 스크립트

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/scripts/logs"
mkdir -p "$LOG_DIR"

# FastAPI 서버 시작
nohup python "$PROJECT_ROOT/main.py" > "$LOG_DIR/server.log" 2>&1 &
echo $! > "$LOG_DIR/server.pid"
echo "   PID: $(cat $LOG_DIR/server.pid)"

sleep 3

# LLM 워커 시작
echo "⚡ LLM 워커 시작..."
nohup python "$PROJECT_ROOT/run_chat_worker.py" > "$LOG_DIR/worker.log" 2>&1 &
echo $! > "$LOG_DIR/worker.pid"
echo "   PID: $(cat $LOG_DIR/worker.pid)"

echo "path: ./scripts/status_services.sh"
echo "path: ./scripts/stop_services.sh"