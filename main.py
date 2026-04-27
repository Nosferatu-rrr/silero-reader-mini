import os
import re
import threading
import time
import json
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from concurrent.futures import ThreadPoolExecutor

# --- Конфигурация и пути ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "v4_ru.pt")
MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
SETTINGS_PATH = os.path.join(PROJECT_DIR, "settings.json")
SAMPLE_RATE = 48000

class SileroEngine:
    def __init__(self, model_path):
        global torch, np, sd, AudioSegment, num2words
        import torch
        import numpy as np
        import sounddevice as sd
        from pydub import AudioSegment
        from num2words import num2words
        
        self.device = torch.device('cpu')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(MODEL_URL, model_path)
            
        import torch.package
        self.model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        self.model.to(self.device)
        self.speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']
        self.lock = threading.Lock()

    def replace_numbers(self, text):
        def replace(match):
            try: return num2words(match.group(), lang='ru')
            except: return match.group()
        return re.sub(r'\d+', replace, text)

    def clean_text_for_engine(self, text):
        text = re.sub(r'https?://\S+', ' ссылка ', text)
        text = self.replace_numbers(text)
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
        with self.lock:
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
        self.current_seg_idx = self.settings.get("last_index", 0)
        
        self.setup_menu()
        self.setup_ui()
        self.load_engine()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_settings(self):
        defaults = {"voice": "baya", "speed": 1.0, "pause_sent": 0.1, "pause_para": 0.4, "last_index": 0, "last_text": ""}
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, 'r') as f:
                    return {**defaults, **json.load(f)}
            except: pass
        return defaults

    def save_settings(self):
        try:
            self.settings["last_text"] = self.text_area.get("1.0", tk.END).strip()
            self.settings["last_index"] = self.current_seg_idx
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(self.settings, f)
        except: pass

    def on_close(self):
        self.stop_requested = True
        try:
            import sounddevice as sd
            sd.stop()
        except: pass
        self.save_settings()
        self.root.destroy()

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Открыть .txt", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.on_close)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(label="Голос и паузы", command=self.open_settings_dialog)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                content = ""
                # Пробуем разные кодировки
                for enc in ['utf-8', 'windows-1251']:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError: continue
                
                if content:
                    self.clear_text()
                    self.text_area.insert("1.0", content)
                    self.text_area.edit_modified(False)
                    self.status_var.set(f"Загружен файл: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {e}")

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
            btn = ttk.Button(toolbar, text=text, command=cmd, width=15)
            btn.pack(side=tk.LEFT, padx=2)
            self.btns[text] = btn
        
        self.btns["▶ Читать"].config(state=tk.DISABLED)
        self.btns["💾 В MP3"].config(state=tk.DISABLED)
        
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.text_area = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 12), undo=True)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        if self.settings["last_text"]:
            self.text_area.insert("1.0", self.settings["last_text"])
            self.text_area.edit_modified(False)
            if self.current_seg_idx > 0:
                self.btns["▶ Читать"].config(text="▶ Продолжить")

        self.text_area.bind("<<Modified>>", self.on_text_modified)
        self.text_area.bind("<ButtonRelease-1>", self.on_click_set_position)
        
        self.context_menu = tk.Menu(self.text_area, tearoff=0)
        self.context_menu.add_command(label="Вырезать", command=lambda: self.text_area.event_generate("<<Cut>>"))
        self.context_menu.add_command(label="Копировать", command=lambda: self.text_area.event_generate("<<Copy>>"))
        self.context_menu.add_command(label="Вставить", command=lambda: self.text_area.event_generate("<<Paste>>"))
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Выделить всё", command=lambda: self.text_area.tag_add("sel", "1.0", tk.END))
        self.text_area.bind("<Button-3>", lambda e: self.context_menu.post(e.x_root, e.y_root))

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.configure(yscrollcommand=scrollbar.set)
        self.text_area.tag_configure("highlight", background="yellow", foreground="black")

    def on_click_set_position(self, event):
        if self.is_playing: return
        cursor_idx = self.text_area.index(tk.INSERT)
        text = self.text_area.get("1.0", tk.END)
        segments = self.split_to_segments(text)
        for i, seg in enumerate(segments):
            if self.text_area.compare(cursor_idx, ">=", seg["start_idx"]) and \
               self.text_area.compare(cursor_idx, "<", seg["end_idx"]):
                self.current_seg_idx = i
                self.btns["▶ Читать"].config(text="▶ Продолжить")
                self.status_var.set(f"Начнем с предложения №{i+1}")
                self.save_settings()
                break

    def on_text_modified(self, event):
        if self.text_area.edit_modified():
            self.current_seg_idx = 0
            self.btns["▶ Читать"].config(text="▶ Читать")
            self.text_area.edit_modified(False)

    def clear_text(self):
        self.stop_reading()
        self.current_seg_idx = 0
        self.text_area.tag_remove("highlight", "1.0", tk.END)
        self.text_area.delete("1.0", tk.END)
        self.btns["▶ Читать"].config(text="▶ Читать")
        self.save_settings()

    def paste_text(self):
        try:
            text = self.root.clipboard_get()
            if text: 
                self.text_area.insert(tk.INSERT, text)
                self.current_seg_idx = 0
                self.btns["▶ Читать"].config(text="▶ Читать")
                self.save_settings()
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
        self.text_area.see(start)

    def start_reading(self):
        if self.is_playing or not self.engine: return
        text = self.text_area.get("1.0", tk.END)
        if not text.strip(): return
        self.is_playing = True
        self.is_paused = False
        self.stop_requested = False
        self.btns["▶ Читать"].config(state=tk.DISABLED, text="▶ Читать")
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except: break
        segments = self.split_to_segments(text)
        if self.current_seg_idx >= len(segments): self.current_seg_idx = 0
        threading.Thread(target=self._producer_loop, args=(segments[self.current_seg_idx:],), daemon=True).start()
        threading.Thread(target=self._consumer_loop, args=(segments, self.current_seg_idx), daemon=True).start()

    def _producer_loop(self, segments_to_read):
        speaker = self.settings["voice"]
        speed = self.settings["speed"]
        for seg in segments_to_read:
            if self.stop_requested: break
            try:
                audio = self.engine.apply_tts(seg["text"], speaker, speed)
                while not self.stop_requested:
                    try:
                        self.audio_queue.put((audio, seg), timeout=0.1)
                        break
                    except queue.Full: continue
            except:
                if not self.stop_requested: self.audio_queue.put((None, seg), timeout=0.1)

    def _consumer_loop(self, segments, start_idx):
        import sounddevice as sd
        total = len(segments)
        try:
            for i in range(start_idx, total):
                if self.stop_requested: break
                audio, seg = None, None
                while not self.stop_requested:
                    try:
                        audio, seg = self.audio_queue.get(timeout=0.1)
                        break
                    except queue.Empty: continue
                if self.stop_requested or (audio is None and seg is None): break
                while self.is_paused and not self.stop_requested: time.sleep(0.1)
                if self.stop_requested: break
                self.current_seg_idx = i
                if i % 5 == 0: self.save_settings()
                self.root.after(0, lambda s=seg: self.set_highlight(s["start_idx"], s["end_idx"]))
                self.status_var.set(f"Читаю: {i+1} из {total}")
                if audio is not None:
                    sd.play(audio, SAMPLE_RATE)
                    while sd.get_stream().active and not self.stop_requested: time.sleep(0.05)
                    if self.stop_requested: sd.stop(); break
                    if seg["pause"] > 0: time.sleep(seg["pause"])
            if not self.stop_requested: self.current_seg_idx = 0
        finally:
            self.is_playing = False
            self.save_settings()
            self.root.after(0, self._reset_ui_after_reading)

    def _reset_ui_after_reading(self):
        text = "▶ Продолжить" if self.current_seg_idx > 0 else "▶ Читать"
        self.btns["▶ Читать"].config(state=tk.NORMAL, text=text)
        self.text_area.tag_remove("highlight", "1.0", tk.END)
        self.status_var.set("Готов")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btns["⏸ Пауза"].config(text="▶ Продолжить" if self.is_paused else "⏸ Пауза")

    def stop_reading(self):
        self.stop_requested = True
        self.is_paused = False
        self.current_seg_idx = 0 
        self.btns["▶ Читать"].config(text="▶ Читать")
        try:
            while not self.audio_queue.empty(): self.audio_queue.get_nowait()
            self.audio_queue.put_nowait((None, None))
        except: pass
        try: import sounddevice as sd; sd.stop()
        except: pass
        self.save_settings()

    def export_mp3(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if not text or not self.engine: return
        file_path = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 Audio", "*.mp3")])
        if not file_path: return
        self.btns["💾 В MP3"].config(state=tk.DISABLED)
        self.status_var.set("Запуск экспорта...")
        def _export():
            try:
                from pydub import AudioSegment
                import numpy as np
                segments = self.split_to_segments(text)
                total = len(segments)
                results = [None] * total
                def process_segment(idx):
                    seg = segments[idx]
                    try:
                        audio_data = self.engine.apply_tts(seg["text"], self.settings["voice"], self.settings["speed"])
                        processed_count = sum(1 for r in results if r is not None)
                        percent = int((processed_count / total) * 100)
                        self.root.after(0, lambda p=percent: self.btns["💾 В MP3"].config(text=f"💾 [{p}%]"))
                        return audio_data
                    except: return None
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(process_segment, i): i for i in range(total)}
                    for future in futures:
                        idx = futures[future]
                        results[idx] = future.result()
                combined = AudioSegment.empty()
                for i, audio_data in enumerate(results):
                    if audio_data is not None:
                        y = (audio_data * 32767).astype(np.int16)
                        combined += AudioSegment(y.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
                        if segments[i]["pause"] > 0:
                            combined += AudioSegment.silent(duration=int(segments[i]["pause"] * 1000), frame_rate=SAMPLE_RATE)
                combined.export(file_path, format="mp3")
                self.root.after(0, lambda: messagebox.showinfo("Успех", f"Сохранено: {os.path.basename(file_path)}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
            finally:
                self.root.after(0, lambda: self.btns["💾 В MP3"].config(state=tk.NORMAL, text="💾 В MP3"))
                self.root.after(0, lambda: self.status_var.set("Готов"))
        threading.Thread(target=_export, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk(className="SileroReader")
    style = ttk.Style()
    if 'clam' in style.theme_names(): style.theme_use('clam')
    app = App(root)
    root.mainloop()
