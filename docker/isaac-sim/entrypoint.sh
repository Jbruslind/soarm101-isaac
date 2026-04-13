#!/bin/bash
# Order matters: Isaac Sim ships Jazzy rosidl_generator_py *.so under the ROS2 bridge extension.
# If /opt/ros/jazzy/lib appears first, the linker loads Debian-built *.so while Python uses
# bridge-bundled msg classes → asserts in rcl_interfaces__msg__*_convert_from_py (core dump).

_BRIDGE_JAZZY_LIB="/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib"
_ROS_SYS_LIB=""
for _distro in jazzy humble; do
    if [ -d "/opt/ros/${_distro}/lib" ]; then
        _ROS_SYS_LIB="/opt/ros/${_distro}/lib"
        break
    fi
done

# Drop any /opt/ros segments from PYTHONPATH (avoid Debian ROS Python on sys.path).
if [ -n "${PYTHONPATH:-}" ]; then
    _new_pp=""
    _ifs_save="$IFS"
    IFS=':'
    # shellcheck disable=SC2206
    _parts=(${PYTHONPATH})
    IFS="$_ifs_save"
    for _p in "${_parts[@]}"; do
        [ -z "$_p" ] && continue
        case "$_p" in
            *"/opt/ros/"*) continue ;;
        esac
        _new_pp="${_new_pp:+${_new_pp}:}${_p}"
    done
    if [ -n "$_new_pp" ]; then
        export PYTHONPATH="$_new_pp"
    else
        unset PYTHONPATH
    fi
fi

_inherit="${LD_LIBRARY_PATH:-}"
_new_ld=""
if [ -d "$_BRIDGE_JAZZY_LIB" ]; then
    _new_ld="$_BRIDGE_JAZZY_LIB"
fi
if [ -n "$_ROS_SYS_LIB" ]; then
    _new_ld="${_new_ld:+${_new_ld}:}${_ROS_SYS_LIB}"
fi
if [ -n "$_inherit" ]; then
    _new_ld="${_new_ld:+${_new_ld}:}${_inherit}"
fi
export LD_LIBRARY_PATH="$_new_ld"

# Do not export ROS_DISTRO here — Isaac Sim uses it to detect a sourced system ROS workspace.

exec "$@"
