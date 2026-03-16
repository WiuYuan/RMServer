# rmstart.sh
# DOC-BEGIN id=scripts/start#1 type=ops v=1
# summary: 启动 uvicorn 服务，以 nohup 后台方式运行，日志输出到 server.log，PID 写入 server.pid
# intent: 使用 nohup 保证 SSH 断开后进程不被杀死；记录 PID 供 stop/restart 脚本精确终止；
#   启动前检查是否已有实例运行，避免端口冲突
# DOC-END id=scripts/start#1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="server.pid"
LOG_FILE="server.log"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[WARN] Server already running (PID=$OLD_PID), please run rmstop.sh first"
        exit 1
    else
        echo "[INFO] Stale PID file found but process is gone, cleaning up"
        rm -f "$PID_FILE"
    fi
fi

echo "[INFO] Starting server..."
nohup python server.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[OK] Server started (PID=$(cat $PID_FILE)), log: $LOG_FILE"
