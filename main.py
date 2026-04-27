import os
import re
import threading
import time
import json
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# --- Конфигурация и пути ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "v4_ru.pt")
MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
SETTINGS_PATH = os.path.join(PROJECT_DIR, "settings.json")
SAMPLE_RATE = 48000

class SileroEngine:
    def __init__(self, model_path):
        global torch, np, sd, AudioSegment
        import torch
        import numpy as np
        import sounddevice as sd
        from pydub import AudioSegment
        
        self.device = torch.device('cpu')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(MODEL_URL, model_path)
            
        import torch.package
        self.model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        self.model.to(self.device)
        self.speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']

    def clean_text_for_engine(self, text):
        text = re.sub(r'https?://\S+', ' ссылка ', text)
        text = text.replace('•', ' ').replace('·', ' ').replace('—', '-')
        text = re.sub(r'[^а-яА-ЯёЁ0-9\s.,!?-——:;()"+]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def trim_silence(self, audio, threshold=0.01):
        import numpy as np
        mask = np.abs(audio) > threshold
        if not np.any(mask): return audio
        start = np.argmax(mask)
        end = len(audio) - np.argmax(mask[::-1])
        return audio[start:end]

    def apply_tts(self, text, speaker, speed):
        cleaned_text = self.clean_text_for_engine(text)
        if not cleaned_text: return None
        audio_tensor = self.model.apply_tts(text=cleaned_text, speaker=speaker, sample_rate=SAMPLE_RATE)
        audio = audio_tensor.numpy()
        audio = self.trim_silence(audio)
        if speed != 1.0:
            import numpy as np
            from pydub import AudioSegment
            y = (audio * 32767).astype(np.int16)
            sound = AudioSegment(y.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
            new_sample_rate = int(sound.frame_rate * speed)
            sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
            sound = sound.set_frame_rate(SAMPLE_RATE)
            return np.array(sound.get_array_of_samples()).astype(np.float32) / 32767.0
        return audio

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Silero Reader Mini")
        self.root.geometry("800x650")
        self.root.minsize(700, 450)
        
        self.engine = None
        self.is_playing = False
        self.is_paused = False
        self.stop_requested = False
        self.audio_queue = queue.Queue(maxsize=3)
        
        self.settings = self.load_settings()
        self.setup_menu()
        self.setup_ui()
        self.load_engine()

    def load_settings(self):
        defaults = {"voice": "baya", "speed": 1.0, "pause_sent": 0.1, "pause_para": 0.4}
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, 'r') as f:
                    return {**defaults, **json.load(f)}
            except: pass
        return defaults

    def save_settings(self, *args):
        try:
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(self.settings, f)
        except: pass

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(label="Голос и паузы", command=self.open_settings_dialog)

    def open_settings_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Настройки")
        dialog.geometry("400x320")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Голос:").grid(row=0, column=0, sticky=tk.W, pady=5)
        voice_var = tk.StringVar(value=self.settings["voice"])
        voice_combo = ttk.Combobox(frame, textvariable=voice_var, state="readonly", values=self.engine.speakers if self.engine else [])
        voice_combo.grid(row=0, column=1, sticky=tk.EW, pady=5)
        ttk.Label(frame, text="Скорость:").grid(row=1, column=0, sticky=tk.W, pady=5)
        speed_scale = ttk.Scale(frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL)
        speed_scale.set(self.settings["speed"])
        speed_scale.grid(row=1, column=1, sticky=tk.EW, pady=5)
        ttk.Label(frame, text="Пауза (предл.):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ps_scale = ttk.Scale(frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL)
        ps_scale.set(self.settings["pause_sent"])
        ps_scale.grid(row=2, column=1, sticky=tk.EW, pady=5)
        ttk.Label(frame, text="Пауза (абзац):").grid(row=3, column=0, sticky=tk.W, pady=5)
        pp_scale = ttk.Scale(frame, from_=0.0, to=2.0, orient=tk.HORIZONTAL)
        pp_scale.set(self.settings["pause_para"])
        pp_scale.grid(row=3, column=1, sticky=tk.EW, pady=5)
        frame.columnconfigure(1, weight=1)
        def save_and_close():
            self.settings["voice"] = voice_var.get()
            self.settings["speed"] = round(speed_scale.get(), 2)
            self.settings["pause_sent"] = round(ps_scale.get(), 2)
            self.settings["pause_para"] = round(pp_scale.get(), 2)
            self.save_settings()
            dialog.destroy()
        ttk.Button(frame, text="Применить", command=save_and_close).grid(row=4, column=0, columnspan=2, pady=20)

    def setup_ui(self):
        self.status_var = tk.StringVar(value="Запуск...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Инструментальная панель
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        btn_config = [
            ("🗑 Очистить", self.clear_text),
            ("📋 Вставить", self.paste_text),
            ("▶ Читать", self.start_reading),
            ("⏸ Пауза", self.toggle_pause),
            ("⏹ Стоп", self.stop_reading),
            ("💾 В MP3", self.export_mp3)
        ]
        
        self.btns = {}
        for text, cmd in btn_config:
            btn = ttk.Button(toolbar, text=text, command=cmd, width=12)
            btn.pack(side=tk.LEFT, padx=2)
            self.btns[text] = btn
        
        self.btns["▶ Читать"].config(state=tk.DISABLED)
        self.btns["💾 В MP3"].config(state=tk.DISABLED)
        
        # Фрейм для текста со скроллбаром
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_area = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 12), undo=True)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.configure(yscrollcommand=scrollbar.set)
        
        # Настройка тега для подсветки
        self.text_area.tag_configure("highlight", background="yellow", foreground="black")

    def clear_text(self):
        self.text_area.tag_remove("highlight", "1.0", tk.END)
        self.text_area.delete("1.0", tk.END)

    def paste_text(self):
        try:
            text = self.root.clipboard_get()
            if text: self.text_area.insert(tk.INSERT, text)
        except: pass

    def load_engine(self):
        def _load():
            try:
                self.engine = SileroEngine(MODEL_PATH)
                self.root.after(0, self._on_engine_ready)
            except:
                self.root.after(0, lambda: self.status_var.set("Ошибка загрузки"))
        threading.Thread(target=_load, daemon=True).start()

    def _on_engine_ready(self):
        self.btns["▶ Читать"].config(state=tk.NORMAL)
        self.btns["💾 В MP3"].config(state=tk.NORMAL)
        self.status_var.set("Готов")

    def split_to_segments(self, text):
        segments = []
        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            if not line.strip(): continue
            sentences = re.split(r'(?<=[.!?])\s+', line)
            curr_col = 0
            for i, s in enumerate(sentences):
                s_trimmed = s.strip()
                if s_trimmed:
                    start_col = line.find(s, curr_col)
                    end_col = start_col + len(s)
                    is_last = (i == len(sentences) - 1)
                    segments.append({
                        "text": s_trimmed,
                        "pause": self.settings["pause_para"] if is_last else self.settings["pause_sent"],
                        "start_idx": f"{line_num}.{start_col}",
                        "end_idx": f"{line_num}.{end_col}"
                    })
                    curr_col = end_col
        return segments

    def set_highlight(self, start, end):
        self.text_area.tag_remove("highlight", "1.0", tk.END)
        self.text_area.tag_add("highlight", start, end)
        self.text_area.see(start) # Прокрутка к тексту

    def start_reading(self):
        if self.is_playing or not self.engine: return
        text = self.text_area.get("1.0", tk.END)
        if not text.strip(): return
        
        self.is_playing = True
        self.is_paused = False
        self.stop_requested = False
        self.btns["▶ Читать"].config(state=tk.DISABLED)
        
        while not self.audio_queue.empty(): self.audio_queue.get()
        segments = self.split_to_segments(text)
        
        threading.Thread(target=self._producer_loop, args=(segments,), daemon=True).start()
        threading.Thread(target=self._consumer_loop, args=(segments,), daemon=True).start()

    def _producer_loop(self, segments):
        for seg in segments:
            if self.stop_requested: break
            try:
                audio = self.engine.apply_tts(seg["text"], self.settings["voice"], self.settings["speed"])
                self.audio_queue.put((audio, seg))
            except: self.audio_queue.put((None, seg))

    def _consumer_loop(self, segments):
        import sounddevice as sd
        total = len(segments)
        for i in range(total):
            if self.stop_requested: break
            audio, seg = self.audio_queue.get()
            
            while self.is_paused:
                if self.stop_requested: break
                time.sleep(0.1)
            
            # Подсветка в основном потоке
            self.root.after(0, lambda s=seg: self.set_highlight(s["start_idx"], s["end_idx"]))
            
            self.status_var.set(f"Читаю: {i+1} из {total}")
            if audio is not None:
                sd.play(audio, SAMPLE_RATE)
                sd.wait()
                if seg["pause"] > 0: time.sleep(seg["pause"])
        
        self.is_playing = False
        self.root.after(0, lambda: self.btns["▶ Читать"].config(state=tk.NORMAL))
        self.root.after(0, lambda: self.text_area.tag_remove("highlight", "1.0", tk.END))
        self.status_var.set("Готов")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btns["⏸ Пауза"].config(text="▶ Продолжить" if self.is_paused else "⏸ Пауза")

    def stop_reading(self):
        self.stop_requested = True
        self.is_paused = False
        try:
            import sounddevice as sd
            sd.stop()
        except: pass

    def export_mp3(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if not text or not self.engine: return
        file_path = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 Audio", "*.mp3")])
        if not file_path: return
        self.status_var.set("Экспорт...")
        def _export():
            try:
                from pydub import AudioSegment
                import numpy as np
                segments = self.split_to_segments(text)
                combined = AudioSegment.empty()
                for seg in segments:
                    try:
                        audio_data = self.engine.apply_tts(seg["text"], self.settings["voice"], self.settings["speed"])
                        if audio_data is not None:
                            y = (audio_data * 32767).astype(np.int16)
                            combined += AudioSegment(y.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
                            if seg["pause"] > 0:
                                combined += AudioSegment.silent(duration=int(seg["pause"] * 1000), frame_rate=SAMPLE_RATE)
                    except: continue
                combined.export(file_path, format="mp3")
                self.root.after(0, lambda: messagebox.showinfo("Успех", "Сохранено!"))
                self.status_var.set("Готов")
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        threading.Thread(target=_export, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    if 'clam' in style.theme_names(): style.theme_use('clam')
    app = App(root)
    root.mainloop()
