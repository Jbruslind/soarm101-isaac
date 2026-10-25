# shellcheck shell=bash
# GPU helpers shared by the scripts in this directory (source it; nothing runs on its own).
#
# The project runs on any NVIDIA GPU. Instead of assuming a card size, scripts ask the driver
# how much memory is free and size their work to fit (batch size, co-running with Isaac Sim).

# gpu_mem_mib: print "<total> <free>" in MiB for the GPU the containers will use
# (the first one, or the first entry of NVIDIA_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES when set
# to an index). Prints nothing and returns 1 when nvidia-smi is not available.
gpu_mem_mib() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local idx="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
    idx="${idx%%,*}"
    [[ "$idx" =~ ^[0-9]+$ ]] || idx=0
    nvidia-smi --id="$idx" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' 'NR==1 {print $1, $2}'
}

# gpu_name: print the GPU's name (for logs only).
gpu_name() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1
}

# mib_to_gb <MiB>: print GB with one decimal.
mib_to_gb() { awk -v m="$1" 'BEGIN {printf "%.1f", m / 1024}'; }
