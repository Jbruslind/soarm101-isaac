#!/usr/bin/env bash
# Manage the Sunshine streaming container (headless NVIDIA X display + Moonlight host).
#
# Usage:
#   ./scripts/sunshine.sh up        # build if needed and start in the background
#   ./scripts/sunshine.sh down      # stop and remove the container
#   ./scripts/sunshine.sh restart
#   ./scripts/sunshine.sh logs      # follow container logs
#   ./scripts/sunshine.sh status    # container state, X display, GPU encoder sessions
#   ./scripts/sunshine.sh xlog      # Xorg log from inside the container
#   ./scripts/sunshine.sh creds USER PASS   # set the Sunshine web UI login
#   ./scripts/sunshine.sh build     # rebuild the image (e.g. after bumping SUNSHINE_VERSION)
#
# After `up`, open https://<host-ip>:47990 to create a login (first time) and pair
# Moonlight clients with the PIN Moonlight shows. Then run any sim script with --gui.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/docker"

compose() { docker compose --profile gui "$@"; }

host_ip() {
    ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}'
}

preflight() {
    if [[ ! -c /dev/uinput ]]; then
        echo "WARNING: /dev/uinput not found; Moonlight keyboard/mouse input will not work."
        echo "         Fix: sudo modprobe uinput && echo uinput | sudo tee /etc/modules-load.d/uinput.conf"
    fi
    if [[ ! -d /run/udev ]]; then
        echo "WARNING: /run/udev not found on the host; input hotplug into Xorg will not work."
    fi
    if [[ "$(systemctl is-active soarm-sunshine-fw.service 2>/dev/null)" != active ]]; then
        echo "WARNING: Sunshine ports are not restricted to LAN/VPN clients."
        echo "         Fix: sudo ./scripts/sunshine-firewall.sh install"
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: NVIDIA driver not working on the host (nvidia-smi failed)." >&2
        exit 1
    fi
}

case "${1:-}" in
    up)
        preflight
        compose up -d --build sunshine
        echo ""
        echo "Waiting for the X display..."
        for _ in $(seq 1 90); do
            docker exec soarm-sunshine xdpyinfo >/dev/null 2>&1 && break
            if [[ -z "$(docker ps -q --filter name='^soarm-sunshine$' --filter status=running)" ]]; then
                echo "soarm-sunshine exited. Last log lines:" >&2
                docker logs --tail 40 soarm-sunshine >&2
                exit 1
            fi
            sleep 1
        done
        ip="$(host_ip)"
        echo ""
        echo "Sunshine is up."
        echo "  Web UI / pairing: https://${ip:-<host-ip>}:47990"
        echo "  Moonlight:        add host ${ip:-<host-ip>}, enter the PIN in the web UI, open 'SO-ARM Isaac Desktop'"
        echo "  Then launch sims with --gui, e.g.: ./scripts/isaac.sh --gui"
        ;;
    down) compose stop sunshine && compose rm -f sunshine ;;
    restart) compose restart sunshine ;;
    build) compose build sunshine ;;
    logs) docker logs -f soarm-sunshine ;;
    xlog) docker exec soarm-sunshine sh -c 'cat /var/log/Xorg.*.log' ;;
    creds)
        [[ $# -eq 3 ]] || { echo "Usage: $0 creds USER PASS"; exit 1; }
        docker exec soarm-sunshine sunshine /root/.config/sunshine/sunshine.conf --creds "$2" "$3"
        compose restart sunshine
        ;;
    status)
        docker ps -a --filter name='^soarm-sunshine$' --format 'container: {{.Names}}  {{.Status}}'
        if docker exec soarm-sunshine xdpyinfo >/dev/null 2>&1; then
            echo "display:   $(docker exec soarm-sunshine sh -c 'echo $DISPLAY; xdpyinfo | awk "/dimensions:/{print \$2}"' | paste -sd' ')"
        else
            echo "display:   not ready"
        fi
        nvidia-smi --query-gpu=encoder.stats.sessionCount,encoder.stats.averageFps --format=csv 2>/dev/null || true
        ;;
    *)
        sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
