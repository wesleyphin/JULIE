from __future__ import annotations

import datetime as dt
import csv
import importlib
import json
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from zoneinfo import ZoneInfo

from config import CONFIG

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_CSV_NAME = "es_master_outrights.parquet"
NY_TZ = ZoneInfo("America/New_York")

BACKTEST_SELECTABLE_STRATEGIES = [
    "RegimeAdaptiveStrategy",
    "VIXReversionStrategy",
    "ImpulseBreakoutStrategy",
    "DynamicEngineStrategy",
    "DynamicEngine3Strategy",
    "IntradayDipStrategy",
    "AuctionReversionStrategy",
    "LiquiditySweepStrategy",
    "ValueAreaBreakoutStrategy",
    "ConfluenceStrategy",
    "SMTStrategy",
    "SmoothTrendAsiaStrategy",
    "MLPhysicsStrategy",
    "ManifoldStrategy",
    "OrbStrategy",
    "ICTModelStrategy",
]

BACKTEST_SELECTABLE_FILTERS = [
    "NewsFilter",
    "PreCandidateGate",
    "RegimeManifold",
    "FixedSLTP",
    "TrendDayTier",
    "TargetFeasibility",
    "RejectionFilter",
    "DirectionalLossBlocker",
    "ImpulseFilter",
    "HTF_FVG",
    "StructureBlocker",
    "RegimeBlocker",
    "PenaltyBoxBlocker",
    "MemorySRFilter",
    "BankLevelQuarterFilter",
    "LegacyTrend",
    "TrendFilter",
    "ChopFilter",
    "ExtensionFilter",
    "VolatilityGuardrail",
    "MLVolRegimeGuard",
    "FilterArbitrator",
]

_BT_MODULE = None


def _load_backtest_module():
    global _BT_MODULE
    if _BT_MODULE is None:
        started = time.perf_counter()
        _BT_MODULE = importlib.import_module("backtest_mes_et")
        elapsed = time.perf_counter() - started
        try:
            print(f"[BacktestUI] Loaded backtest engine in {elapsed:.2f}s")
        except Exception:
            pass
    return _BT_MODULE


def format_dt(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


class BacktestUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Backtest Control")
        self.root.geometry("1180x900")
        self.root.minsize(1080, 760)
        self.root.option_add("*Font", "{Segoe UI} 10")

        self.queue: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()

        self.df_cache = None
        self.df_path: Optional[Path] = None
        self.range_df = None
        self.last_report_path: Optional[Path] = None

        self.csv_path_var = tk.StringVar(value=DEFAULT_CSV_NAME)
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.symbol_var = tk.StringVar()
        self.export_de3_decisions_var = tk.BooleanVar(value=False)
        self.de3_decisions_top_k_var = tk.StringVar(value="5")
        self.de3_decisions_out_var = tk.StringVar(value="reports/de3_decisions.csv")

        self.range_var = tk.StringVar(value="Range: -")
        self.contracts_var = tk.StringVar(value="Contracts: -")
        self.status_var = tk.StringVar(value="Idle")

        self.realized_var = tk.StringVar(value="0.00")
        self.unrealized_var = tk.StringVar(value="0.00")
        self.total_var = tk.StringVar(value="0.00")
        self.trades_var = tk.StringVar(value="0")
        self.wins_var = tk.StringVar(value="0")
        self.losses_var = tk.StringVar(value="0")
        self.winrate_var = tk.StringVar(value="0.00%")
        self.drawdown_var = tk.StringVar(value="0.00")
        self.time_var = tk.StringVar(value="-")

        self.progress = tk.DoubleVar(value=0.0)
        self.report_text = None
        self.recent_text = None
        self.run_log_text = None
        self._last_progress_log_bucket = -1
        self.strategy_select_all_var = tk.BooleanVar(value=True)
        self.filter_select_all_var = tk.BooleanVar(value=True)
        self.strategy_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=True) for name in BACKTEST_SELECTABLE_STRATEGIES
        }
        self.filter_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=True) for name in BACKTEST_SELECTABLE_FILTERS
        }

        self._apply_dark_theme()
        self._build_layout()
        self._poll_queue()

    def _bt(self):
        return _load_backtest_module()

    def _apply_dark_theme(self) -> None:
        self.colors = {
            "bg": "#090909",
            "panel": "#111111",
            "panel_alt": "#171717",
            "text": "#E8E8E8",
            "muted": "#B8B8B8",
            "accent": "#18C47D",
            "accent_active": "#22D88D",
            "border": "#2A2A2A",
            "entry": "#0F0F0F",
        }
        self.root.configure(bg=self.colors["bg"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=self.colors["bg"], foreground=self.colors["text"])
        style.configure(
            "TFrame",
            background=self.colors["bg"],
        )
        style.configure(
            "TLabelframe",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            borderwidth=1,
            relief="solid",
            bordercolor=self.colors["border"],
        )
        style.configure(
            "TLabelframe.Label",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        style.configure(
            "TEntry",
            fieldbackground=self.colors["entry"],
            foreground=self.colors["text"],
            insertcolor=self.colors["text"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["entry"],
            background=self.colors["panel_alt"],
            foreground=self.colors["text"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["text"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.colors["entry"])],
            foreground=[("readonly", self.colors["text"])],
            background=[("readonly", self.colors["panel_alt"])],
        )
        style.configure(
            "TButton",
            background=self.colors["panel_alt"],
            foreground=self.colors["text"],
            bordercolor=self.colors["border"],
            padding=(10, 6),
        )
        style.map(
            "TButton",
            background=[("active", "#242424"), ("disabled", "#131313")],
            foreground=[("disabled", "#777777")],
        )
        style.configure(
            "Accent.TButton",
            background=self.colors["accent"],
            foreground="#050505",
            bordercolor=self.colors["accent"],
            padding=(12, 7),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.colors["accent_active"]), ("disabled", "#1A5C45")],
            foreground=[("disabled", "#0A271E")],
        )
        style.configure(
            "TCheckbutton",
            background=self.colors["panel"],
            foreground=self.colors["text"],
        )
        style.map(
            "TCheckbutton",
            background=[("active", self.colors["panel_alt"])],
            foreground=[("disabled", "#777777")],
        )
        style.configure(
            "TNotebook",
            background=self.colors["panel"],
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background="#141414",
            foreground=self.colors["muted"],
            padding=(10, 6),
            bordercolor=self.colors["border"],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#1F1F1F"), ("active", "#1A1A1A")],
            foreground=[("selected", self.colors["text"])],
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#1A1A1A",
            background=self.colors["accent"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )

        # Combo popup colors.
        self.root.option_add("*TCombobox*Listbox*Background", self.colors["entry"])
        self.root.option_add("*TCombobox*Listbox*Foreground", self.colors["text"])
        self.root.option_add("*TCombobox*Listbox*selectBackground", self.colors["accent"])
        self.root.option_add("*TCombobox*Listbox*selectForeground", "#050505")

    def _toggle_all_strategies(self) -> None:
        state = bool(self.strategy_select_all_var.get())
        for var in self.strategy_vars.values():
            var.set(state)

    def _toggle_all_filters(self) -> None:
        state = bool(self.filter_select_all_var.get())
        for var in self.filter_vars.values():
            var.set(state)

    def _sync_select_all_flags(self) -> None:
        self.strategy_select_all_var.set(all(var.get() for var in self.strategy_vars.values()))
        self.filter_select_all_var.set(all(var.get() for var in self.filter_vars.values()))

    def _selected_strategies(self) -> set[str]:
        return {name for name, var in self.strategy_vars.items() if var.get()}

    def _selected_filters(self) -> set[str]:
        return {name for name, var in self.filter_vars.items() if var.get()}

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        self._scroll_canvas = tk.Canvas(
            outer,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0,
        )
        v_scroll = ttk.Scrollbar(outer, orient="vertical", command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=v_scroll.set)
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")

        main = ttk.Frame(self._scroll_canvas, padding=14)
        self._scroll_window_id = self._scroll_canvas.create_window((0, 0), window=main, anchor="nw")

        def _on_main_configure(_event) -> None:
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

        def _on_canvas_configure(event) -> None:
            try:
                self._scroll_canvas.itemconfigure(self._scroll_window_id, width=event.width)
            except Exception:
                pass

        def _on_mousewheel(event) -> str:
            widget_class = ""
            try:
                widget_class = str(event.widget.winfo_class())
            except Exception:
                widget_class = ""
            if widget_class in {"Text", "Entry", "TEntry", "TCombobox", "Combobox", "Listbox"}:
                return ""
            delta = 0
            if getattr(event, "delta", 0):
                delta = -int(event.delta / 120)
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            if delta != 0:
                self._scroll_canvas.yview_scroll(delta, "units")
                return "break"
            return ""

        main.bind("<Configure>", _on_main_configure)
        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)

        file_frame = ttk.LabelFrame(main, text="Data Source")
        file_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(file_frame, text="CSV:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(file_frame, textvariable=self.csv_path_var, width=60).grid(
            row=0, column=1, sticky="we", padx=6, pady=6
        )
        ttk.Button(file_frame, text="Browse", command=self.browse_csv).grid(
            row=0, column=2, padx=6, pady=6
        )
        ttk.Button(file_frame, text="Preview Range", command=self.preview_range).grid(
            row=0, column=3, padx=6, pady=6
        )
        file_frame.columnconfigure(1, weight=1)

        range_frame = ttk.LabelFrame(main, text="Date Range")
        range_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(range_frame, text="Start (YYYY-MM-DD or YYYY-MM-DD HH:MM):").grid(
            row=0, column=0, sticky="w", padx=6, pady=6
        )
        ttk.Entry(range_frame, textvariable=self.start_var, width=30).grid(
            row=0, column=1, sticky="w", padx=6, pady=6
        )
        ttk.Label(range_frame, text="End:").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        ttk.Entry(range_frame, textvariable=self.end_var, width=30).grid(
            row=0, column=3, sticky="w", padx=6, pady=6
        )

        ttk.Label(range_frame, textvariable=self.range_var).grid(
            row=1, column=0, columnspan=4, sticky="w", padx=6, pady=4
        )
        ttk.Label(range_frame, textvariable=self.contracts_var).grid(
            row=2, column=0, columnspan=4, sticky="w", padx=6, pady=4
        )

        symbol_frame = ttk.LabelFrame(main, text="Contract")
        symbol_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(symbol_frame, text="Symbol:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.symbol_combo = ttk.Combobox(symbol_frame, textvariable=self.symbol_var, state="readonly", width=20)
        self.symbol_combo.grid(row=0, column=1, sticky="w", padx=6, pady=6)

        de3_export_frame = ttk.LabelFrame(main, text="DE3 Decision Journal (Optional)")
        de3_export_frame.pack(fill="x", padx=5, pady=5)
        ttk.Checkbutton(
            de3_export_frame,
            text="Export DE3 decisions",
            variable=self.export_de3_decisions_var,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Label(de3_export_frame, text="Top-K:").grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Entry(de3_export_frame, textvariable=self.de3_decisions_top_k_var, width=8).grid(
            row=0, column=2, sticky="w", padx=6, pady=6
        )
        ttk.Label(de3_export_frame, text="Output CSV:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(de3_export_frame, textvariable=self.de3_decisions_out_var, width=64).grid(
            row=1, column=1, columnspan=2, sticky="we", padx=6, pady=6
        )
        ttk.Button(
            de3_export_frame,
            text="Browse",
            command=self.browse_de3_decisions_out,
        ).grid(row=1, column=3, padx=6, pady=6)
        de3_export_frame.columnconfigure(1, weight=1)

        self._build_scope_selectors(main)

        controls = ttk.Frame(main)
        controls.pack(fill="x", padx=5, pady=10)
        self.run_btn = ttk.Button(
            controls,
            text="Run Backtest",
            command=self.start_backtest,
            style="Accent.TButton",
        )
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop_backtest, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.export_csv_btn = ttk.Button(
            controls,
            text="Save Recent Backtest CSV",
            command=self.export_latest_backtest_csv,
        )
        self.export_csv_btn.pack(side="left", padx=6)
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=12)

        stats = ttk.LabelFrame(main, text="Performance")
        stats.pack(fill="x", expand=False, padx=5, pady=5)

        grid = ttk.Frame(stats)
        grid.pack(fill="x", padx=8, pady=8)

        self._stat_row(grid, 0, "Realized PnL:", self.realized_var, "Unrealized:", self.unrealized_var)
        self._stat_row(grid, 1, "Total PnL:", self.total_var, "Max Drawdown:", self.drawdown_var)
        self._stat_row(grid, 2, "Trades:", self.trades_var, "Winrate:", self.winrate_var)
        self._stat_row(grid, 3, "Wins:", self.wins_var, "Losses:", self.losses_var)
        self._stat_row(grid, 4, "Current Time:", self.time_var, "", tk.StringVar(value=""))

        progress_frame = ttk.Frame(stats)
        progress_frame.pack(fill="x", padx=8, pady=(4, 10))
        ttk.Label(progress_frame, text="Progress:").pack(side="left", padx=6)
        ttk.Progressbar(progress_frame, maximum=100, variable=self.progress).pack(
            side="left", fill="x", expand=True, padx=6
        )

        diagnostics = ttk.LabelFrame(main, text="Diagnostics")
        diagnostics.pack(fill="both", expand=True, padx=5, pady=5)

        notebook = ttk.Notebook(diagnostics)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)

        report_frame = ttk.Frame(notebook)
        trades_frame = ttk.Frame(notebook)
        runlog_frame = ttk.Frame(notebook)
        notebook.add(report_frame, text="Loss Drivers")
        notebook.add(trades_frame, text="Recent Trades")
        notebook.add(runlog_frame, text="Run Log")

        self.report_text = self._make_scrolling_text(report_frame)
        self.recent_text = self._make_scrolling_text(trades_frame)
        self.run_log_text = self._make_scrolling_text(runlog_frame)
        self._append_text(self.run_log_text, "UI initialized. Ready.\n")

    def _build_scope_selectors(self, parent: ttk.Frame) -> None:
        scope = ttk.LabelFrame(parent, text="Backtest Scope")
        scope.pack(fill="x", padx=5, pady=5)
        scope.columnconfigure(0, weight=1)
        scope.columnconfigure(1, weight=1)

        strat_frame = ttk.LabelFrame(scope, text="Strategies")
        filter_frame = ttk.LabelFrame(scope, text="Filters")
        strat_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        filter_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        for idx in range(3):
            strat_frame.columnconfigure(idx, weight=1)
        for idx in range(3):
            filter_frame.columnconfigure(idx, weight=1)

        ttk.Checkbutton(
            strat_frame,
            text="Select All",
            variable=self.strategy_select_all_var,
            command=self._toggle_all_strategies,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        ttk.Checkbutton(
            filter_frame,
            text="Select All",
            variable=self.filter_select_all_var,
            command=self._toggle_all_filters,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))

        for idx, name in enumerate(BACKTEST_SELECTABLE_STRATEGIES):
            row = 1 + (idx // 3)
            col = idx % 3
            ttk.Checkbutton(
                strat_frame,
                text=name.replace("Strategy", ""),
                variable=self.strategy_vars[name],
                command=self._sync_select_all_flags,
            ).grid(row=row, column=col, sticky="w", padx=6, pady=2)

        for idx, name in enumerate(BACKTEST_SELECTABLE_FILTERS):
            row = 1 + (idx // 3)
            col = idx % 3
            ttk.Checkbutton(
                filter_frame,
                text=name,
                variable=self.filter_vars[name],
                command=self._sync_select_all_flags,
            ).grid(row=row, column=col, sticky="w", padx=6, pady=2)

    def _stat_row(
        self,
        parent: ttk.Frame,
        row: int,
        label_left: str,
        var_left: tk.StringVar,
        label_right: str,
        var_right: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text=label_left).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Label(parent, textvariable=var_left, width=16).grid(row=row, column=1, sticky="w", padx=6, pady=3)
        if label_right:
            ttk.Label(parent, text=label_right).grid(row=row, column=2, sticky="w", padx=20, pady=3)
            ttk.Label(parent, textvariable=var_right, width=16).grid(row=row, column=3, sticky="w", padx=6, pady=3)

    def _make_scrolling_text(self, parent: ttk.Frame) -> tk.Text:
        text = tk.Text(
            parent,
            wrap="none",
            height=10,
            bg=self.colors["entry"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            selectbackground="#2D5A47",
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
        )
        scroll_y = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        scroll_x = ttk.Scrollbar(parent, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        text.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        text.configure(state="disabled")
        return text

    def _set_text(self, widget: tk.Text, content: str) -> None:
        try:
            y_top, y_bottom = widget.yview()
            follow_tail = float(y_bottom) >= 0.995
        except Exception:
            y_top = 0.0
            follow_tail = True
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")
        if follow_tail:
            widget.yview_moveto(1.0)
        else:
            widget.yview_moveto(float(y_top))

    def _append_text(self, widget: tk.Text, content: str, max_lines: int = 3000) -> None:
        try:
            _, y_bottom = widget.yview()
            follow_tail = float(y_bottom) >= 0.995
        except Exception:
            follow_tail = True
        widget.configure(state="normal")
        widget.insert("end", content)
        # Keep log bounded for responsiveness.
        try:
            line_count = int(widget.index("end-1c").split(".")[0])
            if line_count > max_lines:
                drop_lines = line_count - max_lines
                widget.delete("1.0", f"{drop_lines + 1}.0")
        except Exception:
            pass
        widget.configure(state="disabled")
        if follow_tail:
            widget.yview_moveto(1.0)

    def _log_ui(self, message: str) -> None:
        if self.run_log_text is None:
            return
        now = dt.datetime.now().strftime("%H:%M:%S")
        self._append_text(self.run_log_text, f"[{now}] {message}\n")

    def _save_report_file(
        self,
        stats: dict,
        symbol: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> Optional[Path]:
        try:
            bt = self._bt()
            return bt.save_backtest_report(stats, symbol, start_time, end_time)
        except Exception:
            return None

    def browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)

    def browse_de3_decisions_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save DE3 decisions CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=Path(self.de3_decisions_out_var.get() or "de3_decisions.csv").name,
        )
        if path:
            self.de3_decisions_out_var.set(path)

    def preview_range(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Backtest", "A task is already running.")
            return
        self.status_var.set("Loading data...")
        self._log_ui("Preview requested.")
        self.worker = threading.Thread(target=self._preview_worker, daemon=True)
        self.worker.start()

    def start_backtest(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Backtest", "A task is already running.")
            return
        self.status_var.set("Running backtest...")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.cancel_event.clear()
        self.progress.set(0.0)
        self._last_progress_log_bucket = -1
        self._log_ui("Backtest started.")
        self.worker = threading.Thread(target=self._backtest_worker, daemon=True)
        self.worker.start()

    def stop_backtest(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Stopping...")
        self._log_ui("Stop requested.")

    def _resolve_report_dir(self) -> Path:
        return Path(__file__).resolve().parent / "backtest_reports"

    def _report_is_completed(self, report_path: Path) -> bool:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        cancelled = bool(summary.get("cancelled", False))
        trades = payload.get("trade_log", [])
        has_trade_log = isinstance(trades, list)
        return bool((not cancelled) and has_trade_log)

    def _latest_backtest_report_path(self) -> Optional[Path]:
        report_dir = self._resolve_report_dir()
        if not report_dir.exists():
            return None
        candidates = sorted(
            [
                p
                for p in report_dir.glob("backtest_*.json")
                if not p.name.startswith("backtest_live_")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        for path in candidates:
            if self._report_is_completed(path):
                return path
        # Fallback to latest file if no completed report is found.
        return candidates[0]

    def _convert_report_json_to_csv(self, report_path: Path) -> Path:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        trades = payload.get("trade_log", []) or []
        if not isinstance(trades, list):
            raise ValueError(f"Report has invalid trade_log payload: {report_path.name}")
        out_path = report_path.with_name(f"converted_{report_path.stem}.csv")
        header = [
            "Trade #",
            "Type",
            "Date and time",
            "Signal",
            "Price USD",
            "Position size (qty)",
            "Position size (value)",
            "Net P&L USD",
            "MFE points",
            "MAE points",
            "Cumulative P&L USD",
        ]
        cumulative = 0.0
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for idx, trade in enumerate(trades, start=1):
                entry_price = float(trade.get("entry_price", 0.0) or 0.0)
                qty = float(trade.get("size", 0.0) or 0.0)
                pnl_net = float(trade.get("pnl_net", 0.0) or 0.0)
                cumulative += pnl_net
                writer.writerow(
                    [
                        idx,
                        "Trade",
                        str(trade.get("entry_time", "") or ""),
                        str(trade.get("side", "") or ""),
                        entry_price,
                        qty,
                        round(entry_price * qty, 4),
                        round(pnl_net, 4),
                        float(trade.get("mfe_points", 0.0) or 0.0),
                        float(trade.get("mae_points", 0.0) or 0.0),
                        round(cumulative, 4),
                    ]
                )
        return out_path

    def export_latest_backtest_csv(self) -> None:
        try:
            report_path: Optional[Path] = None
            if (
                self.last_report_path is not None
                and self.last_report_path.exists()
                and self._report_is_completed(self.last_report_path)
            ):
                report_path = self.last_report_path
            else:
                report_path = self._latest_backtest_report_path()
            if report_path is None:
                messagebox.showinfo("Backtest", "No backtest report JSON found to export.")
                return
            csv_path = self._convert_report_json_to_csv(report_path)
            self.status_var.set(f"CSV saved: {csv_path.name}")
            self._log_ui(f"CSV exported from {report_path.name}: {csv_path.name}")
            messagebox.showinfo(
                "Backtest",
                f"Source report:\n{report_path}\n\nSaved CSV:\n{csv_path}",
            )
        except Exception as exc:
            self._log_worker_exception(exc, context="export_csv")
            messagebox.showerror(
                "Backtest",
                f"Failed to export CSV: {exc}\n\nDetails saved to backtest_ui_error.log",
            )

    def _resolve_csv_path(self) -> Path:
        path = Path(self.csv_path_var.get()).expanduser()
        if path.is_file():
            return path
        base_dir = Path(__file__).resolve().parent
        alt = base_dir / path
        return alt

    def _load_df(self) -> pd.DataFrame:
        path = self._resolve_csv_path()
        if self.df_cache is not None and self.df_path == path:
            return self.df_cache
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        cache_dir = CONFIG.get("BACKTEST_CACHE_DIR", "cache")
        use_cache = bool(CONFIG.get("BACKTEST_USE_CACHE", True))
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir)
            if not cache_path.is_absolute():
                cache_path = Path(__file__).resolve().parent / cache_path
        bt = self._bt()
        df = bt.load_csv_cached(path, cache_dir=cache_path, use_cache=use_cache)
        self.df_cache = df
        self.df_path = path
        return df

    def _parse_range(self, df: pd.DataFrame) -> tuple[dt.datetime, dt.datetime]:
        bt = self._bt()
        start_raw = self.start_var.get().strip()
        end_raw = self.end_var.get().strip()
        start_time = bt.parse_user_datetime(start_raw, bt.NY_TZ, is_end=False) if start_raw else df.index.min()
        end_time = bt.parse_user_datetime(end_raw, bt.NY_TZ, is_end=True) if end_raw else df.index.max()
        if start_time > end_time:
            raise ValueError("Start must be before end.")
        return start_time, end_time

    def _preview_worker(self) -> None:
        try:
            df = self._load_df()
            start_time, end_time = self._parse_range(df)
            range_df = df[(df.index >= start_time) & (df.index <= end_time)]
            if range_df.empty:
                raise ValueError("No rows in the selected range.")
            bt = self._bt()
            bar_minutes = bt.infer_bar_minutes(range_df.index)
            symbol_mode = str(CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single").lower()
            symbols = sorted(range_df["symbol"].dropna().unique().tolist())
            if not symbols:
                raise ValueError("No symbols found in range.")
            if symbol_mode != "single" and len(symbols) > 1:
                symbols = ["AUTO_BY_DAY"]
            self.queue.put(
                {
                    "type": "preview",
                    "start": start_time,
                    "end": end_time,
                    "symbols": symbols,
                    "bar_minutes": bar_minutes,
                }
            )
        except Exception as exc:
            self._log_worker_exception(exc, context="preview")
            self.queue.put({"type": "error", "message": str(exc)})

    def _backtest_worker(self) -> None:
        try:
            self.queue.put({"type": "status", "message": "Loading data/caches..."})
            df = self._load_df()
            start_time, end_time = self._parse_range(df)
            source_df = df[df.index <= end_time]
            range_df = source_df[(source_df.index >= start_time) & (source_df.index <= end_time)]
            if range_df.empty:
                raise ValueError("No rows in the selected range.")
            symbol_mode = str(CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single").lower()
            symbol_method = CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume")
            symbols = sorted(range_df["symbol"].dropna().unique().tolist())
            if not symbols:
                raise ValueError("No symbols found in range.")
            selected = self.symbol_var.get().strip()
            symbol_df = source_df
            bt = self._bt()
            if symbol_mode != "single" and len(symbols) > 1:
                symbol_df, selected, _ = bt.apply_symbol_mode(
                    source_df, symbol_mode, symbol_method
                )
                if symbol_df.empty:
                    raise ValueError("No rows found after auto symbol selection.")
                selected_test_df = symbol_df[(symbol_df.index >= start_time) & (symbol_df.index <= end_time)]
                if selected_test_df.empty:
                    raise ValueError("No rows found in selected range after auto symbol selection.")
                symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
            else:
                if selected not in symbols:
                    selected = bt.choose_symbol(range_df, None)
                symbol_df = source_df[source_df["symbol"] == selected]
                if symbol_df.empty:
                    raise ValueError("No rows found for selected symbol.")
                selected_test_df = symbol_df[(symbol_df.index >= start_time) & (symbol_df.index <= end_time)]
                if selected_test_df.empty:
                    raise ValueError("No rows found in selected range for selected symbol.")
                symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
            source_attrs = getattr(source_df, "attrs", {}) or {}
            symbol_df = bt.attach_backtest_symbol_context(
                symbol_df,
                selected,
                symbol_mode,
                source_key=source_attrs.get("source_cache_key"),
                source_label=source_attrs.get("source_label"),
                source_path=source_attrs.get("source_path"),
            )

            selected_strategies = self._selected_strategies()
            selected_filters = self._selected_filters()
            enabled_strategies = (
                None
                if len(selected_strategies) == len(BACKTEST_SELECTABLE_STRATEGIES)
                else selected_strategies
            )
            enabled_filters = (
                None if len(selected_filters) == len(BACKTEST_SELECTABLE_FILTERS) else selected_filters
            )
            de3_export_enabled = bool(self.export_de3_decisions_var.get())
            try:
                de3_top_k = max(1, int(self.de3_decisions_top_k_var.get().strip() or "5"))
            except Exception:
                de3_top_k = 5
            de3_out_path = str(self.de3_decisions_out_var.get().strip() or "reports/de3_decisions.csv")

            def progress_cb(payload: dict) -> None:
                payload = payload.copy()
                payload_type = str(payload.get("type", "progress") or "progress").strip().lower()
                if payload_type not in {"progress", "status"}:
                    payload_type = "progress"
                payload["type"] = payload_type
                payload["symbol"] = selected
                self.queue.put(payload)

            self.queue.put({"type": "status", "message": "Initializing backtest engine..."})
            stats = bt.run_backtest(
                symbol_df,
                start_time,
                end_time,
                progress_cb=progress_cb,
                cancel_event=self.cancel_event,
                enabled_strategies=enabled_strategies,
                enabled_filters=enabled_filters,
                export_de3_decisions=de3_export_enabled,
                de3_decisions_top_k=de3_top_k,
                de3_decisions_out=de3_out_path,
            )
            stats["symbol_mode"] = symbol_mode
            self.queue.put(
                {
                    "type": "done",
                    "stats": stats,
                    "symbol": selected,
                    "start": start_time,
                    "end": end_time,
                }
            )
        except Exception as exc:
            self._log_worker_exception(exc, context="backtest")
            self.queue.put({"type": "error", "message": str(exc)})

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _log_worker_exception(self, exc: Exception, context: str) -> None:
        log_path = Path(__file__).resolve().parent / "backtest_ui_error.log"
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n=== Backtest UI error ({}) ===\n".format(context))
                handle.write("{}\n".format(str(exc)))
                handle.write(traceback.format_exc())
        except Exception:
            pass

    def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        if msg_type == "error":
            self.status_var.set("Error")
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            message = msg.get("message", "Unknown error")
            self._log_ui(f"ERROR: {message}")
            messagebox.showerror(
                "Backtest",
                f"{message}\n\nDetails saved to backtest_ui_error.log",
            )
            return
        if msg_type == "status":
            message = str(msg.get("message", "") or "").strip()
            if message:
                self.status_var.set(message)
                self._log_ui(message)
            return
        if msg_type == "preview":
            start = msg["start"]
            end = msg["end"]
            symbols = msg["symbols"]
            bar_minutes = msg.get("bar_minutes")
            if bar_minutes:
                self.range_var.set(f"Range: {start} to {end} (NY) | Bar: {bar_minutes}m")
            else:
                self.range_var.set(f"Range: {start} to {end} (NY)")
            self.contracts_var.set(f"Contracts in range: {', '.join(symbols)}")
            self.symbol_combo["values"] = symbols
            if not self.symbol_var.get() or self.symbol_var.get() not in symbols:
                self.symbol_var.set(symbols[0])
            if not self.start_var.get().strip():
                self.start_var.set(format_dt(start))
            if not self.end_var.get().strip():
                self.end_var.set(format_dt(end))
            self.status_var.set("Ready")
            self._log_ui(
                f"Preview loaded: {start} -> {end} | symbols={', '.join(symbols)}"
            )
            return
        if msg_type == "progress":
            self.realized_var.set(f"{msg.get('equity', 0.0):.2f}")
            self.unrealized_var.set(f"{msg.get('unrealized', 0.0):.2f}")
            self.total_var.set(f"{msg.get('total', 0.0):.2f}")
            self.trades_var.set(str(msg.get("trades", 0)))
            self.wins_var.set(str(msg.get("wins", 0)))
            self.losses_var.set(str(msg.get("losses", 0)))
            self.winrate_var.set(f"{msg.get('winrate', 0.0):.2f}%")
            self.drawdown_var.set(f"{msg.get('max_drawdown', 0.0):.2f}")
            ts = msg.get("time")
            if isinstance(ts, dt.datetime):
                self.time_var.set(format_dt(ts))
            bar_index = msg.get("bar_index", 0)
            total_bars = msg.get("total_bars", 0) or 1
            pct = min(100.0, (bar_index / total_bars) * 100.0)
            self.progress.set(pct)
            bucket = int(pct // 10)
            if bucket > self._last_progress_log_bucket:
                self._last_progress_log_bucket = bucket
                self._log_ui(
                    f"Progress {pct:.1f}% | trades={msg.get('trades', 0)} | "
                    f"winrate={msg.get('winrate', 0.0):.2f}% | pnl={msg.get('equity', 0.0):.2f}"
                )
            report = msg.get("report")
            if report is not None and self.report_text is not None:
                self._set_text(self.report_text, report)
            recent = msg.get("recent_trades")
            if recent is not None and self.recent_text is not None:
                self._set_text(self.recent_text, "\n".join(recent))
            return
        if msg_type == "done":
            stats = msg.get("stats", {})
            label = "Cancelled" if stats.get("cancelled") else "Finished"
            report_path = None
            start_time = msg.get("start")
            end_time = msg.get("end")
            symbol = msg.get("symbol", "")
            if isinstance(start_time, dt.datetime) and isinstance(end_time, dt.datetime):
                report_path = self._save_report_file(stats, symbol, start_time, end_time)
            if report_path is not None:
                self.last_report_path = report_path
            if report_path is not None:
                self.status_var.set(f"{label} (saved {report_path.name})")
            else:
                self.status_var.set(label)
            self._log_ui(
                f"{label}: trades={stats.get('trades', 0)} wins={stats.get('wins', 0)} "
                f"losses={stats.get('losses', 0)} winrate={stats.get('winrate', 0.0):.2f}% "
                f"net={stats.get('equity', 0.0):.2f}"
            )
            de3_export_meta = stats.get("de3_decisions_export", {}) or {}
            if bool(de3_export_meta.get("enabled")):
                csv_path = str(de3_export_meta.get("path", "") or "").strip()
                trade_path = str(de3_export_meta.get("trade_attribution_path", "") or "").strip()
                if csv_path:
                    self._log_ui(f"DE3 decisions CSV: {csv_path}")
                if trade_path:
                    self._log_ui(f"DE3 trade attribution CSV: {trade_path}")
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            return


def main() -> None:
    root = tk.Tk()
    BacktestUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
