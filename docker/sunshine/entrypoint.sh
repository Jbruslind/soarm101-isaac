#!/bin/bash
# Start a headless NVIDIA Xorg display, Openbox, PulseAudio and Sunshine.
#
# Environment:
#   DISPLAY            X display to create (default :0). Isaac Sim --gui uses the same value.
#   DISPLAY_WIDTH      Virtual monitor width  (default 1920)
#   DISPLAY_HEIGHT     Virtual monitor height (default 1080)
#   DISPLAY_REFRESH    Virtual monitor refresh in Hz (default 60)
#   NVIDIA_OUTPUT      NVIDIA connector to fake a monitor on (default DFP-0)
#   SUNSHINE_NAME      Host name shown in Moonlight (default soarm-isaac)
#   SUNSHINE_USER / SUNSHINE_PASS
#                      If both set, (re)set the Sunshine web UI credentials at start.
set -euo pipefail

: "${DISPLAY:=:0}"
: "${DISPLAY_WIDTH:=1920}"
: "${DISPLAY_HEIGHT:=1080}"
: "${DISPLAY_REFRESH:=60}"
: "${NVIDIA_OUTPUT:=DFP-0}"
: "${SUNSHINE_NAME:=soarm-isaac}"
export DISPLAY

SUNSHINE_DIR="/root/.config/sunshine"
XORG_MODULE_CACHE="/var/cache/nvidia-xorg"
XORG_CONF="/etc/X11/soarm-xorg.conf"
DISPLAY_NUM="${DISPLAY#:}"
DISPLAY_NUM="${DISPLAY_NUM%%.*}"

log() { echo "[sunshine] $*"; }
die() { echo "[sunshine] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. GPU + driver checks
# ---------------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi not found. Run with the NVIDIA runtime and NVIDIA_DRIVER_CAPABILITIES=all."
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
PCI_BUS_ID="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr -d ' ')"
[[ -n "$DRIVER_VERSION" && -n "$PCI_BUS_ID" ]] || die "could not query the GPU via nvidia-smi."

# nvidia-smi reports DDDDDDDD:BB:DD.F in hex; Xorg wants PCI:bus@domain:dev:func in decimal.
IFS=':.' read -r _dom _bus _dev _fn <<<"$PCI_BUS_ID"
BUSID="PCI:$((16#$_bus))@$((16#${_dom: -4})):$((16#$_dev)):$((16#$_fn))"
log "GPU $PCI_BUS_ID (Xorg BusID $BUSID), driver $DRIVER_VERSION"

[[ -c /dev/uinput ]] || log "WARNING: /dev/uinput missing; Moonlight mouse/keyboard input will not work (host: sudo modprobe uinput)."

# ---------------------------------------------------------------------------
# 2. NVIDIA Xorg modules matching the host driver
# ---------------------------------------------------------------------------
find_module() {
    local name="$1" dir
    for dir in /usr/lib/x86_64-linux-gnu/nvidia/xorg /usr/lib/xorg/modules/drivers /usr/lib/xorg/modules/extensions; do
        compgen -G "$dir/$name*" >/dev/null && return 0
    done
    return 1
}

if find_module nvidia_drv.so && find_module libglxserver_nvidia.so; then
    log "Using NVIDIA Xorg modules provided by the container toolkit."
else
    cache="$XORG_MODULE_CACHE/$DRIVER_VERSION"
    if [[ ! -f "$cache/nvidia_drv.so" ]]; then
        run="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
        log "Fetching NVIDIA $DRIVER_VERSION driver bundle for its Xorg modules (one-time, cached)..."
        tmp="$(mktemp -d)"
        curl -fSL --retry 3 -o "$tmp/$run" \
            "https://download.nvidia.com/XFree86/Linux-x86_64/${DRIVER_VERSION}/${run}" \
            || die "download failed; mount the host's nvidia_drv.so/libglxserver_nvidia.so instead."
        sh "$tmp/$run" -x --target "$tmp/extract" >/dev/null
        mkdir -p "$cache"
        cp "$tmp/extract/nvidia_drv.so" "$cache/"
        cp "$tmp/extract/libglxserver_nvidia.so.$DRIVER_VERSION" "$cache/libglxserver_nvidia.so"
        rm -rf "$tmp"
    fi
    install -D -m 0644 "$cache/nvidia_drv.so" /usr/lib/xorg/modules/drivers/nvidia_drv.so
    install -D -m 0644 "$cache/libglxserver_nvidia.so" /usr/lib/xorg/modules/extensions/libglxserver_nvidia.so
    log "Installed NVIDIA Xorg modules for $DRIVER_VERSION."
fi

# ---------------------------------------------------------------------------
# 3. Virtual monitor + Xorg config
# ---------------------------------------------------------------------------
python3 /opt/soarm/make_edid.py "$DISPLAY_WIDTH" "$DISPLAY_HEIGHT" "$DISPLAY_REFRESH" /etc/X11/soarm-edid.bin
sed -e "s|@BUSID@|$BUSID|g" -e "s|@OUTPUT@|$NVIDIA_OUTPUT|g" /opt/soarm/xorg.conf.template > "$XORG_CONF"

# The socket dir is a volume shared with Isaac Sim containers; clear stale state from a previous run.
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix
rm -f "/tmp/.X11-unix/X${DISPLAY_NUM}" "/tmp/.X${DISPLAY_NUM}-lock"

PIDS=()
cleanup() {
    trap - TERM INT EXIT
    log "Stopping..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup TERM INT EXIT

# No -auth: any local client (including Isaac Sim containers via the shared socket) may connect.
# TCP is disabled, so this is limited to processes on this host.
log "Starting Xorg $DISPLAY at ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}@${DISPLAY_REFRESH} on $NVIDIA_OUTPUT"
Xorg "$DISPLAY" vt7 -novtswitch -sharevts -keeptty -noreset -nolisten tcp \
    -config "$XORG_CONF" -logfile "/var/log/Xorg.${DISPLAY_NUM}.log" &
PIDS+=($!)

for _ in $(seq 1 60); do
    xdpyinfo >/dev/null 2>&1 && break
    kill -0 "${PIDS[0]}" 2>/dev/null || { tail -n 40 "/var/log/Xorg.${DISPLAY_NUM}.log" >&2; die "Xorg exited during startup."; }
    sleep 0.5
done
xdpyinfo >/dev/null 2>&1 || die "Xorg did not become ready (see /var/log/Xorg.${DISPLAY_NUM}.log)."
log "X display ready: $(xdpyinfo | awk '/dimensions:/{print $2}')"

xset s off -dpms s noblank
xsetroot -solid '#20242a'
openbox &
PIDS+=($!)

# ---------------------------------------------------------------------------
# 4. Audio (Sunshine creates its own virtual sink on PulseAudio)
# ---------------------------------------------------------------------------
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
pulseaudio --daemonize=no --exit-idle-time=-1 --log-target=stderr --log-level=error &
PIDS+=($!)

# ---------------------------------------------------------------------------
# 5. Sunshine
# ---------------------------------------------------------------------------
mkdir -p "$SUNSHINE_DIR"
if [[ ! -f "$SUNSHINE_DIR/sunshine.conf" ]]; then
    cat > "$SUNSHINE_DIR/sunshine.conf" <<EOF
sunshine_name = $SUNSHINE_NAME
capture = x11
encoder = nvenc
output_name = 0
origin_web_ui_allowed = lan
file_apps = $SUNSHINE_DIR/apps.json
EOF
    log "Wrote default $SUNSHINE_DIR/sunshine.conf (persisted in the sunshine-config volume)."
fi
[[ -f "$SUNSHINE_DIR/apps.json" ]] || cp /opt/soarm/apps.json "$SUNSHINE_DIR/apps.json"

# The web UI's CSRF check only trusts localhost by default, so browsing to
# https://<host-ip>:47990 is rejected. Trust every host address and the hostname
# (host networking), plus SUNSHINE_CSRF_ORIGINS. Refreshed on every start.
CSRF_ORIGINS=""
for h in $(hostname -I) "$(hostname)" localhost 127.0.0.1; do
    CSRF_ORIGINS+="${CSRF_ORIGINS:+,}https://$h:47990"
done
[[ -n "${SUNSHINE_CSRF_ORIGINS:-}" ]] && CSRF_ORIGINS+=",$SUNSHINE_CSRF_ORIGINS"
sed -i '/^csrf_allowed_origins[[:space:]]*=/d' "$SUNSHINE_DIR/sunshine.conf"
echo "csrf_allowed_origins = $CSRF_ORIGINS" >> "$SUNSHINE_DIR/sunshine.conf"
log "Web UI CSRF allowed origins: $CSRF_ORIGINS"

if [[ -n "${SUNSHINE_USER:-}" && -n "${SUNSHINE_PASS:-}" ]]; then
    sunshine "$SUNSHINE_DIR/sunshine.conf" --creds "$SUNSHINE_USER" "$SUNSHINE_PASS" >/dev/null
    log "Web UI credentials set for user '$SUNSHINE_USER'."
fi

log "Starting Sunshine. Web UI: https://<this-host>:47990  (pair Moonlight clients there)"
sunshine "$SUNSHINE_DIR/sunshine.conf" &
PIDS+=($!)

# Exit (and let the restart policy recover) if any component dies.
set +e
wait -n "${PIDS[@]}"
status=$?
log "A display/stream process exited (status $status); shutting down."
exit "$status"
