import argparse
import subprocess
import sys


def _append_if(args: list, flag: str, value) -> None:
    if value is None or value == "":
        return
    args.extend([flag, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all continuation-related trainers in sequence.")
    parser.add_argument("--csv", default=None, help="CSV path passed to all trainers.")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--experimental-window",
        action="store_true",
        help="Train all trainers on configured experimental window (2011-01-01 .. 2017-12-31).",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Suffix appended to all trainer output artifacts (e.g. _exp2011_2017).",
    )
    parser.add_argument("--recent-start", default=None, help="Recent window start date (YYYY-MM-DD).")
    parser.add_argument("--recent-end", default=None, help="Recent window end date (YYYY-MM-DD).")
    parser.add_argument(
        "--recent-mode",
        default=None,
        choices=("intersect", "union", "recent_only"),
        help="How to combine full vs recent results.",
    )
    parser.add_argument("--no-recent", action="store_true", help="Disable recency window.")
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Use proxy signals for the continuation allowlist trainer.",
    )
    parser.add_argument("--cache-dir", default=None, help="Cache directory for parquet/features.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache read/write.")
    args = parser.parse_args()

    common_args: list[str] = []
    _append_if(common_args, "--csv", args.csv)
    _append_if(common_args, "--start", args.start)
    _append_if(common_args, "--end", args.end)
    _append_if(common_args, "--artifact-suffix", args.artifact_suffix)
    _append_if(common_args, "--recent-start", args.recent_start)
    _append_if(common_args, "--recent-end", args.recent_end)
    _append_if(common_args, "--recent-mode", args.recent_mode)
    _append_if(common_args, "--cache-dir", args.cache_dir)
    if args.no_recent:
        common_args.append("--no-recent")
    if args.no_cache:
        common_args.append("--no-cache")
    if args.experimental_window:
        common_args.append("--experimental-window")

    allowlist_cmd = [sys.executable, "train_continuation_allowlist.py", *common_args]
    if args.proxy:
        allowlist_cmd.append("--proxy")

    sltp_cmd = [sys.executable, "train_continuation_sltp.py", *common_args]
    flip_cmd = [sys.executable, "train_flip_confidence.py", *common_args]

    for cmd in (allowlist_cmd, sltp_cmd, flip_cmd):
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
