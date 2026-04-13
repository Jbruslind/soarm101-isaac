#!/usr/bin/env bash
# Debug script for remote inference when the robot does not move.
#
# Run ON the Jetson to verify OpenPi + Caddy are up and listening.
# Run FROM the client (with --host JETSON_IP) to verify connectivity and
# how to inspect the ROS2 bridge.
#
# Usage:
#   On Jetson:
#     ./scripts/debug_jetson.sh
#     ./scripts/debug_jetson.sh --logs 100
#   From client (machine where you run interactive_test.sh):
#     ./scripts/debug_jetson.sh --host 192.168.1.50
#     ./scripts/debug_jetson.sh --host jetson.local --port 8000
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_HOST=""
REMOTE_PORT="8443"
LOG_LINES=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --host) REMOTE_HOST="$2"; shift 2 ;;
        --port) REMOTE_PORT="$2"; shift 2 ;;
        --logs) LOG_LINES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "  SO-ARM VLA – Remote inference debug"
echo "=============================================="
echo ""

if [[ -n "$REMOTE_HOST" ]]; then
    # --- Run from CLIENT: check connectivity to Jetson ---
    echo ">>> Checking connectivity TO Jetson at ${REMOTE_HOST}:${REMOTE_PORT}"
    echo ""

    echo "1. TCP connect to ${REMOTE_HOST}:${REMOTE_PORT}"
    if command -v nc &>/dev/null; then
        if nc -zv -w3 "$REMOTE_HOST" "$REMOTE_PORT" 2>&1; then
            echo "   OK – port is open."
        else
            echo "   FAIL – cannot connect. Check firewall and that Caddy/OpenPi are running on Jetson."
        fi
    else
        echo "   (install netcat to test: apt install netcat-openbsd)"
    fi

    echo ""
    echo "2. HTTPS endpoint (Caddy proxy)"
    if [[ "$REMOTE_PORT" == "8443" ]]; then
        if curl -k -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "https://${REMOTE_HOST}:${REMOTE_PORT}" 2>/dev/null | grep -q .; then
            echo "   curl -k https://${REMOTE_HOST}:8443  -> check output above"
            curl -k -s --connect-timeout 3 "https://${REMOTE_HOST}:8443" 2>/dev/null | head -c 200 || true
            echo ""
        else
            echo "   curl -k https://${REMOTE_HOST}:8443  (run manually if needed)"
        fi
    fi

    echo ""
    echo "3. ROS2 bridge (run on the machine where you start interactive_test.sh)"
    echo "   After starting: ./scripts/interactive_test.sh --remote ${REMOTE_HOST} --port ${REMOTE_PORT}"
    echo "   In another terminal, watch bridge logs:"
    echo "     cd $PROJECT_DIR/docker"
    echo "     docker compose --profile interactive logs -f ros2-bridge"
    echo "   Look for:"
    echo "     [OpenVLA] TCP connect to ... succeeded"
    echo "     [OpenVLA] Inference ENABLED"
    echo "     [OpenVLA] OpenPi returned N action(s)"
    echo "     [OpenVLA] Publishing joint command #N"
    echo "   If you see 'Inference failed' or 'queue empty', the Jetson OpenPi may not be responding."
    echo ""
    echo "4. ROS2 topic (optional – from host with ROS2):"
    echo "   source /opt/ros/humble/setup.bash"
    echo "   ros2 topic echo /joint_commands --once"
    echo "   (Only works if ROS_DOMAIN_ID matches bridge, typically 42.)"
    echo ""
    exit 0
fi

# --- Run ON the Jetson: check OpenPi + Caddy ---
echo ">>> Running ON this machine (Jetson) – checking OpenPi stack"
echo ""

COMPOSE_FILE="$PROJECT_DIR/docker/docker-compose.jetson.yml"
if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "Expected $COMPOSE_FILE not found. Run from repo root or set PROJECT_DIR."
    exit 1
fi

cd "$PROJECT_DIR/docker"

echo "1. Docker containers (openpi + caddy)"
docker compose -f docker-compose.jetson.yml ps -a 2>/dev/null || true
echo ""

echo "2. Ports 8000 and 8443"
for port in 8000 8443; do
    if (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -q ":${port} "; then
        echo "   Port $port: listening"
    else
        echo "   Port $port: not listening (start stack: docker compose -f docker-compose.jetson.yml up -d)"
    fi
done
echo ""

echo "3. OpenPi server logs (last ${LOG_LINES} lines)"
docker compose -f docker-compose.jetson.yml logs --tail="$LOG_LINES" openpi-server 2>/dev/null || docker logs --tail="$LOG_LINES" soarm-openpi-jetson 2>/dev/null || echo "   (container not running or different name)"
echo ""

echo "4. Caddy logs (last 15 lines)"
docker compose -f docker-compose.jetson.yml logs --tail=15 caddy 2>/dev/null || docker logs --tail=15 soarm-caddy 2>/dev/null || echo "   (container not running)"
echo ""

echo "5. Local connectivity"
if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:8000 2>/dev/null | grep -q .; then
    echo "   http://127.0.0.1:8000 -> reachable (OpenPi direct)"
else
    echo "   http://127.0.0.1:8000 -> not reachable (OpenPi may be WebSocket-only or not ready)"
fi
if curl -k -s -o /dev/null -w "%{http_code}" --connect-timeout 2 https://127.0.0.1:8443 2>/dev/null | grep -q .; then
    echo "   https://127.0.0.1:8443 -> reachable (Caddy)"
else
    echo "   https://127.0.0.1:8443 -> not reachable"
fi
echo ""

echo "=============================================="
echo "  From your CLIENT run:"
echo "    ./scripts/debug_jetson.sh --host <JETSON_IP>"
echo "  to test connectivity and see bridge log commands."
echo "=============================================="
