# Silero Reader Mini

Приложение для озвучивания текста (TTS) на моделях Silero. Интерфейс на Tkinter, работа с `.txt`, экспорт в MP3.

## Возможности

*   Чтение с любого предложения — клик по тексту задаёт начало.
*   Запоминание текста и позиции между запусками.
*   Числа озвучиваются словами, латиница транслитерируется в кириллицу.
*   Открытие `.txt` в UTF-8 и Windows-1251.
*   Экспорт в MP3 с прогрессом на кнопке.
*   Подсветка текущей фразы и автоскролл.
*   Настройка голоса, скорости и пауз.
*   Смена версии модели в меню «Настройки → Модель».

## Установка

```bash
sudo apt install ffmpeg libportaudio2 python3-tk
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python /usr/bin/python3
uv pip install -r requirements.txt
uv pip install --index-url https://download.pytorch.org/whl/cpu --reinstall-package torch torch
```

Окружение создаётся на системном Python: он использует Tk с Xft, поэтому шрифты отображаются корректно.

## Запуск

```bash
./run.sh
```

## Версии модели

Список версий — словарь `AVAILABLE_MODELS` в начале `main.py`. Выбранная версия хранится в `settings.json`. Пункт «Настройки → Модель → Проверить обновления» подтягивает новые версии из официального `models.yml`.

## Технологии

Silero TTS v5, Tkinter, sounddevice, pydub, num2words.

## Лицензия

MIT