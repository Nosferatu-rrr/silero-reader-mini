#!/bin/bash
# Переход в папку проекта
cd "$(dirname "$0")"

# Запуск python напрямую (без nohup и &, чтобы не терять контекст и ошибки)
./venv/bin/python main.py
