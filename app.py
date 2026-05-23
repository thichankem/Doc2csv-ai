"""Doc2CSV-AI - Tkinter desktop GUI (DPI-aware, with system monitor)."""
import os
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from src.ollama_client import is_running, list_models
from src.pipeline import Pipeline
from src.sysmon import SystemMonitor

SUPPORTED_FILETYPES = [
    ("Documents", "*.pdf *.docx *.doc *.txt *.md"),
    ("PDF", "*.pdf"),
    ("Word", "*.docx *.doc"),
    ("Text", "*.txt *.md"),
    ("All files", "*.*"),
]

PREFERRED_MODELS = (
    "qwen3.5:9b", "llama3.1:8b", "deepseek-r1:8b",
    "deepseek-r1:14b", "gemma4:e4b",
)

INSTRUCTION_PLACEHOLDER = (
    "Ví dụ: Tóm tắt đoạn văn sau bằng tiếng Việt trong 3-5 câu.\n"
    "Hoặc: Trích xuất tất cả định nghĩa thuật ngữ theo dạng 'Thuật ngữ: định nghĩa'.\n"
    "(Để TRỐNG = chế độ auto Q&A, model tự sinh nhiều cặp hỏi-đáp mỗi chunk)"
)


# ---------------------------------------------------------------------------
# DPI awareness — must run before Tk() to get sharp rendering on HiDPI screens.
# ---------------------------------------------------------------------------
def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    import ctypes
    for fn in (
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),   # Per-Monitor v2
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(1),   # Per-Monitor v1
        lambda: ctypes.windll.user32.SetProcessDPIAware(),         # System
    ):
        try:
            fn()
            return
        except (AttributeError, OSError):
            continue


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "--"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec:02d}s"
    h, rem = divmod(s, 3600)
    m = rem // 60
    return f"{h}h {m:02d}m"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Doc2CSV-AI · Trích xuất dữ liệu training")
        root.geometry("980x860")
        root.minsize(880, 740)

        self.files: list[str] = []
        self.worker: threading.Thread | None = None
        self.stop_flag = False

        self.sysmon = SystemMonitor()
        self._sysmon_job: Optional[str] = None
        self._placeholder_on = True

        self._build_ui()
        self.refresh_models()
        self._schedule_sysmon()

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        # ===== Section 1: Files =====
        fbox = ttk.LabelFrame(main, text="  1. Files đầu vào (.pdf / .docx / .doc / .txt)  ")
        fbox.pack(fill="x", pady=(0, 10))

        toolbar = ttk.Frame(fbox)
        toolbar.pack(fill="x", **pad)
        ttk.Button(toolbar, text="Thêm file...", command=self.add_files).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Xóa đã chọn", command=self.remove_selected).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Xóa hết", command=self.clear_files).pack(side="left")
        self.lbl_count = ttk.Label(toolbar, text="0 file")
        self.lbl_count.pack(side="right")

        list_frame = ttk.Frame(fbox)
        list_frame.pack(fill="both", expand=False, **pad)
        self.lst_files = tk.Listbox(
            list_frame, height=5, selectmode="extended",
            activestyle="dotbox",
            font=("Segoe UI", 10),
            highlightthickness=0, borderwidth=1, relief="solid",
        )
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.lst_files.yview)
        self.lst_files.config(yscrollcommand=sb.set)
        self.lst_files.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ===== Section 2: Instruction =====
        ibox = ttk.LabelFrame(main, text="  2. Instruction (cùng dùng cho mỗi chunk → cột 'instruction')  ")
        ibox.pack(fill="x", pady=(0, 10))

        instr_frame = ttk.Frame(ibox)
        instr_frame.pack(fill="x", **pad)
        self.txt_instr = tk.Text(
            instr_frame, height=4, wrap="word",
            font=("Segoe UI", 10),
            highlightthickness=0, borderwidth=1, relief="solid",
            padx=8, pady=6,
        )
        sb_i = ttk.Scrollbar(instr_frame, orient="vertical", command=self.txt_instr.yview)
        self.txt_instr.config(yscrollcommand=sb_i.set)
        self.txt_instr.pack(side="left", fill="both", expand=True)
        sb_i.pack(side="right", fill="y")

        self.txt_instr.insert("1.0", INSTRUCTION_PLACEHOLDER)
        self.txt_instr.config(foreground="gray")
        self.txt_instr.bind("<FocusIn>", self._instr_focus_in)
        self.txt_instr.bind("<FocusOut>", self._instr_focus_out)

        # ===== Section 3: Config =====
        cbox = ttk.LabelFrame(main, text="  3. Cấu hình  ")
        cbox.pack(fill="x", pady=(0, 10))

        r1 = ttk.Frame(cbox); r1.pack(fill="x", **pad)
        ttk.Label(r1, text="Model Ollama:").pack(side="left")
        self.cmb_model = ttk.Combobox(r1, state="readonly", width=28)
        self.cmb_model.pack(side="left", padx=8)
        ttk.Button(r1, text="↻ Refresh", command=self.refresh_models).pack(side="left")
        self.lbl_ollama = ttk.Label(r1, text="", foreground="gray")
        self.lbl_ollama.pack(side="left", padx=12)

        r2 = ttk.Frame(cbox); r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Output CSV:").pack(side="left")
        self.var_out = tk.StringVar(value=str(Path.cwd() / "output" / "training_data.csv"))
        ttk.Entry(r2, textvariable=self.var_out, font=("Segoe UI", 10)).pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(r2, text="...", command=self.choose_output, width=4).pack(side="left")

        r3 = ttk.Frame(cbox); r3.pack(fill="x", **pad)
        ttk.Label(r3, text="Chunk size (từ):").pack(side="left")
        self.var_chunk = tk.IntVar(value=1500)
        ttk.Spinbox(r3, from_=300, to=8000, increment=100,
                    textvariable=self.var_chunk, width=8).pack(side="left", padx=6)

        ttk.Label(r3, text="Samples/chunk (auto):").pack(side="left", padx=(18, 0))
        self.var_samples = tk.IntVar(value=3)
        ttk.Spinbox(r3, from_=1, to=10, textvariable=self.var_samples, width=6).pack(side="left", padx=6)

        ttk.Label(r3, text="Temp:").pack(side="left", padx=(18, 0))
        self.var_temp = tk.DoubleVar(value=0.3)
        ttk.Spinbox(r3, from_=0.0, to=1.5, increment=0.1,
                    textvariable=self.var_temp, width=6, format="%.2f").pack(side="left", padx=6)

        ttk.Label(r3, text="num_ctx:").pack(side="left", padx=(18, 0))
        self.var_ctx = tk.IntVar(value=8192)
        ttk.Spinbox(r3, from_=2048, to=32768, increment=1024,
                    textvariable=self.var_ctx, width=8).pack(side="left", padx=6)

        # ===== Action buttons =====
        abox = ttk.Frame(main)
        abox.pack(fill="x", pady=(0, 10))
        self.btn_start = ttk.Button(abox, text="▶  Bắt đầu trích xuất", command=self.start, style="Accent.TButton")
        self.btn_start.pack(side="left", padx=(0, 8), ipadx=6, ipady=2)
        self.btn_stop = ttk.Button(abox, text="⏹  Dừng", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", ipadx=4, ipady=2)
        ttk.Button(abox, text="📂  Mở thư mục output", command=self.open_output_dir).pack(
            side="right", ipadx=4, ipady=2
        )

        # ===== Bottom split: Progress (left) + System monitor (right) =====
        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(0, 10))
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=2)

        # Progress
        pbox = ttk.LabelFrame(bottom, text="  4. Tiến trình  ")
        pbox.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.pb = ttk.Progressbar(pbox, mode="determinate")
        self.pb.pack(fill="x", padx=10, pady=(10, 6))
        self.lbl_status = ttk.Label(pbox, text="Sẵn sàng.", font=("Segoe UI", 10, "bold"))
        self.lbl_status.pack(anchor="w", padx=10, pady=2)
        self.lbl_eta = ttk.Label(pbox, text="ETA: --", foreground="#0066cc")
        self.lbl_eta.pack(anchor="w", padx=10, pady=2)
        self.lbl_substatus = ttk.Label(pbox, text="", foreground="gray", font=("Consolas", 9))
        self.lbl_substatus.pack(anchor="w", padx=10, pady=(2, 10))

        # System monitor
        sbox = ttk.LabelFrame(bottom, text="  5. Tài nguyên hệ thống  ")
        sbox.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_sysmon(sbox)

        # ===== Section 6: Log =====
        lbox = ttk.LabelFrame(main, text="  6. Log  ")
        lbox.pack(fill="both", expand=True)
        log_frame = ttk.Frame(lbox)
        log_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.txt_log = tk.Text(
            log_frame, height=10, wrap="word",
            font=("Consolas", 9),
            highlightthickness=0, borderwidth=1, relief="solid",
            padx=8, pady=6,
        )
        sb2 = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt_log.yview)
        self.txt_log.config(yscrollcommand=sb2.set, state="disabled")
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")

    def _build_sysmon(self, parent: ttk.LabelFrame) -> None:
        wrap = ttk.Frame(parent, padding=(10, 10))
        wrap.pack(fill="both", expand=True)
        wrap.columnconfigure(1, weight=1)

        def row(r: int, label: str) -> tuple[ttk.Progressbar, ttk.Label]:
            ttk.Label(wrap, text=label, width=6).grid(row=r, column=0, sticky="w", pady=3)
            pb = ttk.Progressbar(wrap, mode="determinate", maximum=100)
            pb.grid(row=r, column=1, sticky="ew", padx=(8, 8))
            val = ttk.Label(wrap, text="--", width=16, anchor="e", font=("Consolas", 9))
            val.grid(row=r, column=2, sticky="e")
            return pb, val

        self.pb_cpu, self.lbl_cpu = row(0, "CPU")
        self.pb_ram, self.lbl_ram = row(1, "RAM")
        self.pb_gpu, self.lbl_gpu = row(2, "GPU")
        self.pb_vram, self.lbl_vram = row(3, "VRAM")

        gpu_name = self.sysmon.gpu_name or "Không phát hiện GPU NVIDIA"
        self.lbl_gpu_name = ttk.Label(wrap, text=gpu_name, foreground="gray", font=("Segoe UI", 9))
        self.lbl_gpu_name.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

    # ---------------------------------------------------------------- sysmon
    def _schedule_sysmon(self) -> None:
        try:
            s = self.sysmon.sample()
            self.pb_cpu["value"] = s.cpu_pct
            self.lbl_cpu.config(text=f"{s.cpu_pct:5.1f} %")
            self.pb_ram["value"] = s.ram_pct
            self.lbl_ram.config(text=f"{s.ram_used_gb:4.1f}/{s.ram_total_gb:.1f} GB")
            if s.gpu_pct is not None:
                self.pb_gpu["value"] = s.gpu_pct
                self.lbl_gpu.config(text=f"{s.gpu_pct:5.1f} %")
                self.pb_vram["value"] = s.vram_pct or 0.0
                self.lbl_vram.config(text=f"{s.vram_used_gb:4.2f}/{s.vram_total_gb:.2f} GB")
            else:
                self.lbl_gpu.config(text="n/a")
                self.lbl_vram.config(text="n/a")
        except Exception:
            pass
        self._sysmon_job = self.root.after(1000, self._schedule_sysmon)

    # ------------------------------------------------------------ placeholder
    def _instr_focus_in(self, _evt) -> None:
        if self._placeholder_on:
            self.txt_instr.delete("1.0", "end")
            self.txt_instr.config(foreground="black")
            self._placeholder_on = False

    def _instr_focus_out(self, _evt) -> None:
        if not self.txt_instr.get("1.0", "end-1c").strip():
            self.txt_instr.delete("1.0", "end")
            self.txt_instr.insert("1.0", INSTRUCTION_PLACEHOLDER)
            self.txt_instr.config(foreground="gray")
            self._placeholder_on = True

    def _get_instruction(self) -> str:
        if self._placeholder_on:
            return ""
        return self.txt_instr.get("1.0", "end-1c").strip()

    # ------------------------------------------------------------ file ops
    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Chọn file", filetypes=SUPPORTED_FILETYPES)
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.lst_files.insert("end", p)
        self.lbl_count.config(text=f"{len(self.files)} file")

    def remove_selected(self) -> None:
        for idx in reversed(self.lst_files.curselection()):
            self.lst_files.delete(idx)
            del self.files[idx]
        self.lbl_count.config(text=f"{len(self.files)} file")

    def clear_files(self) -> None:
        self.files.clear()
        self.lst_files.delete(0, "end")
        self.lbl_count.config(text="0 file")

    def choose_output(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="training_data.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if p:
            self.var_out.set(p)

    def open_output_dir(self) -> None:
        out = Path(self.var_out.get()).parent
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out))
        except AttributeError:
            messagebox.showinfo("Output dir", str(out))

    # ------------------------------------------------------------- Ollama
    def refresh_models(self) -> None:
        if not is_running():
            self.lbl_ollama.config(text="⚠ Ollama không chạy (localhost:11434)", foreground="red")
            self.cmb_model["values"] = []
            self.cmb_model.set("")
            return
        models = list_models()
        self.cmb_model["values"] = models
        if not models:
            self.cmb_model.set("")
            self.lbl_ollama.config(text="⚠ Chưa có model nào", foreground="orange")
            return
        chosen = ""
        for pref in PREFERRED_MODELS:
            if pref in models:
                chosen = pref
                break
        self.cmb_model.set(chosen or models[0])
        self.lbl_ollama.config(text=f"✓ {len(models)} model có sẵn", foreground="#0a8a3a")

    # ---------------------------------------------------- logging & progress
    def _append_log(self, msg: str) -> None:
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    def log(self, msg: str) -> None:
        self.root.after(0, self._append_log, msg)

    def update_progress(self, cur: int, total: int, eta: Optional[float] = None) -> None:
        def _apply():
            self.pb["maximum"] = max(total, 1)
            self.pb["value"] = cur
            pct = (cur / total * 100) if total else 0.0
            self.lbl_status.config(text=f"Tổng: {cur}/{total} chunks  ({pct:.1f}%)")
            if eta is None or eta <= 0:
                self.lbl_eta.config(text="ETA: --")
            else:
                self.lbl_eta.config(text=f"ETA: {format_duration(eta)}")
        self.root.after(0, _apply)

    def update_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.lbl_substatus.config(text=msg))

    # -------------------------------------------------------------- run/stop
    def start(self) -> None:
        if not self.files:
            messagebox.showwarning("Thiếu file", "Vui lòng thêm ít nhất một file.")
            return
        model = self.cmb_model.get().strip()
        if not model:
            messagebox.showwarning("Thiếu model", "Vui lòng chọn model Ollama.")
            return
        out = self.var_out.get().strip()
        if not out:
            messagebox.showwarning("Thiếu output", "Vui lòng chọn đường dẫn output CSV.")
            return

        if Path(out).exists():
            if not messagebox.askyesno(
                "File đã tồn tại",
                f"File '{Path(out).name}' đã có. Tiếp tục sẽ APPEND thêm dòng mới vào cuối file.\nTiếp tục?",
            ):
                return

        instruction = self._get_instruction()
        mode = "Custom instruction" if instruction else "Auto Q&A"
        self.log(f"\n=== Bắt đầu run | Mode: {mode} | Model: {model} ===")

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.stop_flag = False
        self.pb["value"] = 0
        self.lbl_eta.config(text="ETA: tính toán...")

        pipe = Pipeline(
            files=list(self.files),
            model=model,
            output_csv=out,
            instruction=instruction,
            chunk_words=int(self.var_chunk.get()),
            samples_per_chunk=int(self.var_samples.get()),
            temperature=float(self.var_temp.get()),
            num_ctx=int(self.var_ctx.get()),
            on_log=self.log,
            on_progress=self.update_progress,
            on_status=self.update_status,
            should_stop=lambda: self.stop_flag,
        )

        def runner():
            try:
                pipe.run()
            except Exception as e:
                self.log(f"❌ Lỗi không mong đợi: {e}")
            finally:
                self.root.after(0, self._on_done)

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_flag = True
        self.log("⏸ Đang dừng sau token / chunk hiện tại...")
        self.btn_stop.config(state="disabled")

    def _on_done(self) -> None:
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        if self.stop_flag:
            self.lbl_status.config(text="Đã dừng.")
        else:
            self.lbl_status.config(text="Hoàn tất.")
        self.lbl_eta.config(text="ETA: --")

    def _on_close(self) -> None:
        self.stop_flag = True
        if self._sysmon_job is not None:
            try:
                self.root.after_cancel(self._sysmon_job)
            except Exception:
                pass
        try:
            self.sysmon.shutdown()
        except Exception:
            pass
        self.root.destroy()


def _setup_fonts() -> None:
    # Set the global Tk named fonts so all widgets render with Segoe UI / Consolas.
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                 "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
                 "TkIconFont", "TkTooltipFont"):
        try:
            tkfont.nametofont(name).configure(family="Segoe UI", size=10)
        except tk.TclError:
            pass
    try:
        tkfont.nametofont("TkFixedFont").configure(family="Consolas", size=10)
    except tk.TclError:
        pass


def _setup_style(root: tk.Tk) -> None:
    style = ttk.Style()
    for theme in ("vista", "winnative", "clam"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break

    style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
    style.configure("TButton", padding=(10, 5))
    style.configure("Accent.TButton", padding=(14, 6), font=("Segoe UI", 10, "bold"))
    style.configure("TProgressbar", thickness=18)


def main() -> None:
    enable_dpi_awareness()
    root = tk.Tk()
    _setup_fonts()
    _setup_style(root)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
