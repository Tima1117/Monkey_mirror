#!/usr/bin/env bash
# Один раз создаёт виртуальную камеру «Monkey Cam» (/dev/video10). Запускать через sudo.
set -e
if [ "$(id -u)" -ne 0 ]; then echo "Запусти так: sudo ./setup-linux.sh"; exit 1; fi

if command -v apt-get >/dev/null; then
  apt-get install -y libegl1 libgles2 python3-venv
fi

if ! modinfo v4l2loopback >/dev/null 2>&1; then
  echo "Ставлю v4l2loopback…"
  if command -v apt-get >/dev/null; then
    apt-get install -y "linux-headers-$(uname -r)" || true
    apt-get install -y v4l2loopback-dkms
  elif command -v dnf >/dev/null; then dnf install -y v4l2loopback mesa-libEGL
  elif command -v pacman >/dev/null; then pacman -S --needed --noconfirm v4l2loopback-dkms linux-headers mesa
  else echo "Не знаю твой пакетный менеджер: установи v4l2loopback вручную."; exit 1
  fi
fi

OPTS='devices=1 video_nr=10 card_label="Monkey Cam" exclusive_caps=1'
modprobe -r v4l2loopback 2>/dev/null || true
if ! modprobe v4l2loopback devices=1 video_nr=10 card_label="Monkey Cam" exclusive_caps=1; then
  echo "Модуль не загрузился. Если включён Secure Boot — подпиши модуль (DKMS обычно предлагает это при установке) или отключи Secure Boot."
  exit 1
fi

# Чтобы камера появлялась сама после перезагрузки.
echo v4l2loopback > /etc/modules-load.d/monkey-cam.conf
echo "options v4l2loopback $OPTS" > /etc/modprobe.d/monkey-cam.conf
echo "Готово: виртуальная камера /dev/video10 «Monkey Cam». Теперь запусти ./run.sh"
