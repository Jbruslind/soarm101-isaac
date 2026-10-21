# Sunshine game-stream host + headless Xorg display for Isaac Sim GUI sessions.
#
# This container owns an X server (:0 by default) running on the NVIDIA GPU with a
# virtual monitor (fake EDID), a minimal window manager, and Sunshine. Isaac Sim
# containers started with --gui connect to that X server through the shared
# x11-unix volume, and Sunshine streams the display to Moonlight clients (NVENC).
#
# The NVIDIA Xorg modules (nvidia_drv.so, libglxserver_nvidia.so) must exactly match
# the host driver; the entrypoint picks them up from the NVIDIA container toolkit or
# fetches the matching driver bundle at first start (cached in a volume).
FROM ubuntu:24.04

ARG SUNSHINE_VERSION=2026.914.233613

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3-minimal \
        libcap2-bin \
        udev \
        xserver-xorg-core \
        xserver-xorg-input-libinput \
        x11-xserver-utils \
        x11-utils \
        openbox \
        pulseaudio \
        libglvnd0 libgl1 libegl1 libgles2 libvulkan1 \
    && curl -fsSL -o /tmp/sunshine.deb \
        "https://github.com/LizardByte/Sunshine/releases/download/v${SUNSHINE_VERSION}/sunshine_${SUNSHINE_VERSION}-1+ubuntu24.04_amd64.deb" \
    && apt-get install -y --no-install-recommends /tmp/sunshine.deb \
    && rm -rf /tmp/sunshine.deb /var/lib/apt/lists/*

COPY make_edid.py /opt/soarm/make_edid.py
COPY xorg.conf.template /opt/soarm/xorg.conf.template
COPY apps.json /opt/soarm/apps.json
COPY openbox-rc.xml /etc/xdg/openbox/rc.xml
COPY entrypoint.sh /opt/soarm/entrypoint.sh
RUN chmod +x /opt/soarm/entrypoint.sh /opt/soarm/make_edid.py

# Sunshine web UI / pairing (TCP 47984 47989 47990 48010, UDP 47998-48000).
# The service runs with host networking, so these are informational only.
EXPOSE 47984 47989 47990 48010 47998/udp 47999/udp 48000/udp

ENTRYPOINT ["/opt/soarm/entrypoint.sh"]
