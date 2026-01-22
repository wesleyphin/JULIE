import datetime as dt
import queue
import threading
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from backtest_mes_et import (
    DEFAULT_CSV_NAME,
    NY_TZ,
    choose_symbol,
    load_csv,
    parse_user_datetime,
    run_backtest,
    save_backtest_report,
)


def format_dt(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


class BacktestUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Backtest UI")
        self.root.geometry("980x820")

        self.queue: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()

        self.df_cache = None
        self.df_path: Optional[Path] = None
        self.range_df = None

        self.csv_path_var = tk.StringVar(value=DEFAULT_CSV_NAME)
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.symbol_var = tk.StringVar()

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

        self._build_layout()
        self._poll_queue()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

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

        controls = ttk.Frame(main)
        controls.pack(fill="x", padx=5, pady=10)
        self.run_btn = ttk.Button(controls, text="Run Backtest", command=self.start_backtest)
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop_backtest, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
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
        notebook.add(report_frame, text="Loss Drivers")
        notebook.add(trades_frame, text="Recent Trades")

        self.report_text = self._make_scrolling_text(report_frame)
        self.recent_text = self._make_scrolling_text(trades_frame)

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
        text = tk.Text(parent, wrap="none", height=10)
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
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _save_report_file(
        self,
        stats: dict,
        symbol: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> Optional[Path]:
        try:
            return save_backtest_report(stats, symbol, start_time, end_time)
        except Exception:
            return None

    def browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)

    def preview_range(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Backtest", "A task is already running.")
            return
        self.status_var.set("Loading data...")
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
        self.worker = threading.Thread(target=self._backtest_worker, daemon=True)
        self.worker.start()

    def stop_backtest(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Stopping...")

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
        df = load_csv(path)
        self.df_cache = df
        self.df_path = path
        return df

    def _parse_range(self, df: pd.DataFrame) -> tuple[dt.datetime, dt.datetime]:
        start_raw = self.start_var.get().strip()
        end_raw = self.end_var.get().strip()
        start_time = parse_user_datetime(start_raw, NY_TZ, is_end=False) if start_raw else df.index.min()
        end_time = parse_user_datetime(end_raw, NY_TZ, is_end=True) if end_raw else df.index.max()
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
            symbols = sorted(range_df["symbol"].dropna().unique().tolist())
            if not symbols:
                raise ValueError("No symbols found in range.")
            self.queue.put(
                {
                    "type": "preview",
                    "start": start_time,
                    "end": end_time,
                    "symbols": symbols,
                }
            )
        except Exception as exc:
            self.queue.put({"type": "error", "message": str(exc)})

    def _backtest_worker(self) -> None:
        try:
            df = self._load_df()
            start_time, end_time = self._parse_range(df)
            range_df = df[(df.index >= start_time) & (df.index <= end_time)]
            if range_df.empty:
                raise ValueError("No rows in the selected range.")
            symbols = sorted(range_df["symbol"].dropna().unique().tolist())
            if not symbols:
                raise ValueError("No symbols found in range.")
            selected = self.symbol_var.get().strip()
            if selected not in symbols:
                selected = choose_symbol(range_df, None)
            symbol_df = range_df[range_df["symbol"] == selected]
            if symbol_df.empty:
                raise ValueError("No rows found for selected symbol.")

            def progress_cb(payload: dict) -> None:
                payload = payload.copy()
                payload["type"] = "progress"
                payload["symbol"] = selected
                self.queue.put(payload)

            stats = run_backtest(
                symbol_df,
                start_time,
                end_time,
                progress_cb=progress_cb,
                cancel_event=self.cancel_event,
            )
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
            self.queue.put({"type": "error", "message": str(exc)})

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        if msg_type == "error":
            self.status_var.set("Error")
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            messagebox.showerror("Backtest", msg.get("message", "Unknown error"))
            return
        if msg_type == "preview":
            start = msg["start"]
            end = msg["end"]
            symbols = msg["symbols"]
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
            self.progress.set(min(100.0, (bar_index / total_bars) * 100.0))
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
                self.status_var.set(f"{label} (saved {report_path.name})")
            else:
                self.status_var.set(label)
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            return


def main() -> None:
    root = tk.Tk()
    BacktestUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
