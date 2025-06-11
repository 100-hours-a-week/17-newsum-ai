#!/bin/bash
# ~/17-newsum-ai/scripts/stop_services.sh
# NewSum AI 서비스 종료 스크립트

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/scripts/logs"

echo "🛑 NewSum AI 서비스 종료 중..."

# FastAPI 서버 종료
if [ -f "$LOG_DIR/server.pid" ]; then
    SERVER_PID=$(cat "$LOG_DIR/server.pid")
    if ps -p "$SERVER_PID" > /dev/null; then
        echo "📡 FastAPI 서버 종료 (PID: $SERVER_PID)..."
        kill "$SERVER_PID"
        sleep 2
        if ps -p "$SERVER_PID" > /dev/null; then
            echo "   강제 종료 중..."
            kill -9 "$SERVER_PID"
        fi
    fi
    rm -f "$LOG_DIR/server.pid"
fi

# LLM 워커 종료
if [ -f "$LOG_DIR/worker.pid" ]; then
    WORKER_PID=$(cat "$LOG_DIR/worker.pid")
    if ps -p "$WORKER_PID" > /dev/null; then
        echo "⚡ LLM 워커 종료 (PID: $WORKER_PID)..."
        kill "$WORKER_PID"
        sleep 2
        if ps -p "$WORKER_PID" > /dev/null; then
            echo "   강제 종료 중..."
            kill -9 "$WORKER_PID"
        fi
    fi
    rm -f "$LOG_DIR/worker.pid"
fi

echo "✅ 모든 서비스가 종료되었습니다."