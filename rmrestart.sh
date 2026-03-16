# rmrestart.sh
# DOC-BEGIN id=scripts/restart#1 type=ops v=1
# summary: 顺序执行 stop.sh（销毁terminal+杀进程）然后 start.sh（启动新实例）
# intent: 确保 stop 完全结束后再 start，避免端口占用；中间 sleep 1s 给操作系统回收端口的时间
# DOC-END id=scripts/restart#1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========== Stopping server =========="
bash "$SCRIPT_DIR/rmstop.sh"

echo ""
echo "[INFO] Waiting 1 second..."
sleep 1

echo "========== Starting server =========="
bash "$SCRIPT_DIR/rmstart.sh"
