#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[Monkey Cam] Первый запуск: ставлю зависимости, это займёт пару минут…"
  if ! python3 -m venv .venv; then
    rm -rf .venv
    echo "Не получилось создать окружение Python. На Ubuntu/Debian: sudo apt install python3-venv"
    exit 1
  fi
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { rm -rf .venv; echo "Не получилось поставить зависимости. Проверь интернет и запусти ещё раз."; exit 1; }
fi
exec .venv/bin/python monkey_cam.py "$@"
