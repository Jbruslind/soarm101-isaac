# shellcheck shell=bash
# Sourced by the sim launch scripts to implement --gui: Isaac Sim opens a normal
# window on the X display owned by the soarm-sunshine container, which streams it
# to Moonlight clients. See docs/SUNSHINE_STREAMING.md.

# gui_enable DOCKER_DIR
#   Verifies the Sunshine display is up and points `docker compose` at the GUI overlay
#   (docker-compose.gui.yml) for the rest of the calling script.
gui_enable() {
    local docker_dir="$1"
    if [[ -z "$(docker ps -q --filter name='^soarm-sunshine$' --filter status=running)" ]]; then
        echo "ERROR: --gui needs the Sunshine display container. Start it first:" >&2
        echo "         ./scripts/sunshine.sh up" >&2
        return 1
    fi
    if ! docker exec soarm-sunshine xdpyinfo >/dev/null 2>&1; then
        echo "ERROR: soarm-sunshine is running but its X display is not ready." >&2
        echo "       Check: ./scripts/sunshine.sh logs" >&2
        return 1
    fi
    # GUI runs share the host IPC namespace (ipc: host) for X11 MIT-SHM. A crashed Kit
    # process leaves its carb-* segments in the host /dev/shm, and the next launch then
    # aborts with "Failed to create shared memory named carb-RStringInternals-1".
    # Clear them when no other Isaac Sim container is running (soarm-sunshine also
    # uses ipc: host, so it sees the same /dev/shm and can remove the root-owned files).
    if [[ -z "$(docker ps -q --filter label=com.docker.compose.service=isaac-sim)" ]]; then
        docker exec soarm-sunshine sh -c 'rm -f /dev/shm/carb-* /dev/shm/sem.carb*' || true
    fi
    export COMPOSE_FILE="$docker_dir/docker-compose.yml:$docker_dir/docker-compose.gui.yml"
    local ip
    ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
    echo "GUI mode: Isaac Sim will open on the Sunshine display."
    echo "  Watch it from Moonlight: add host ${ip:-<this-host>} and open 'SO-ARM Isaac Desktop'."
}
