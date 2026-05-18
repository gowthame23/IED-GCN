"""manage_outputs.py

Small utility to centralize model outputs in the workspace and create a live view.

Commands:
    init-view            Create `outputs_view/` and populate with links to final_outputs.
  collect [--dry]      Move matching output files from workspace root into `archived_outputs/` and replace originals with symlinks to the archive (keeps original paths working).
  link <src> <dst>     Create a symlink or copy fallback.

Notes:
- On Windows, creating symlinks may require admin or developer mode. If symlink creation fails the script will copy the file instead and print a warning.
- Symlinks provide real-time view: editing the original file updates the linked view automatically.

Example:
  python manage_outputs.py init-view
  python manage_outputs.py collect
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
OUTPUT_VIEW = WORKSPACE / "outputs_view"
ARCHIVE = WORKSPACE / "archived_outputs"
KNOWN_OUTPUT_DIRS = [WORKSPACE / "final_outputs"]
DEFAULT_PATTERNS = ["*.npy", "*.json", "*.tif", "*.png", "*.csv"]


def ensure_dirs():
    OUTPUT_VIEW.mkdir(exist_ok=True)
    ARCHIVE.mkdir(exist_ok=True)


def try_symlink_or_copy(src: Path, dst: Path) -> None:
    """Create a live file view at dst pointing to src.

    Hard links work on Windows without elevated privileges and keep the file contents
    in sync. If that fails, try a symlink, then fall back to copying.
    """
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.link(src, dst)
        print(f"hardlink -> {dst} -> {src}")
    except Exception as exc:  # pragma: no cover - environment dependent
        try:
            dst.symlink_to(src)
            print(f"symlink -> {dst} -> {src}")
        except Exception as e:
            try:
                shutil.copy2(src, dst)
                print(f"copied -> {dst} (source: {src}) [link failed: {exc}; symlink failed: {e}]")
            except Exception as copy_exc:
                print(f"ERROR copying {src} to {dst}: {copy_exc}")


def init_view() -> None:
    ensure_dirs()
    created = 0
    for out_dir in KNOWN_OUTPUT_DIRS:
        if not out_dir.exists():
            continue
        for p in out_dir.iterdir():
            if p.is_file():
                dst = OUTPUT_VIEW / p.name
                try_symlink_or_copy(p, dst)
                created += 1
    print(f"init-view complete: {created} links created in {OUTPUT_VIEW}")


def collect(dry: bool = False, patterns: list[str] | None = None) -> None:
    """Move files from workspace root matching patterns into ARCHIVE and leave symlink at original location.

    This helps to centralize files and keep original paths working via symlink.
    """
    ensure_dirs()
    patterns = patterns or DEFAULT_PATTERNS
    moved = 0
    for pattern in patterns:
        for src in WORKSPACE.glob(pattern):
            # skip files already in archive or outputs_view
            if ARCHIVE in src.parents or OUTPUT_VIEW in src.parents:
                continue
            if not src.is_file():
                continue
            dst = ARCHIVE / src.name
            print(f"found: {src}")
            if dry:
                print(f"dry-run: would move {src} -> {dst}")
                moved += 1
                continue
            # move then create symlink at original location pointing to archive
            try:
                if dst.exists():
                    # if destination exists, backup with suffix
                    dst = ARCHIVE / (src.stem + "_dup" + src.suffix)
                shutil.move(str(src), str(dst))
                try:
                    src.symlink_to(dst)
                    print(f"moved and symlinked: {src} -> {dst}")
                except Exception as exc:
                    # fallback: copy back a small pointer file with path
                    # but better to copy the file back so original path still has file
                    shutil.copy2(dst, src)
                    print(f"moved and copied back (symlink failed): {src} <- {dst} [{exc}]")
                moved += 1
            except Exception as e:
                print(f"ERROR moving {src} -> {dst}: {e}")
    print(f"collect complete: {moved} items processed. Archive: {ARCHIVE}")


def link(src: str, dst: str) -> None:
    s = Path(src)
    d = Path(dst)
    if not s.exists():
        print(f"source does not exist: {s}")
        return
    try_symlink_or_copy(s, d)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manage workspace outputs: create view and archive files centrally.")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("init-view", help="create outputs_view/ with links to known output folders")
    c = sub.add_parser("collect", help="move matched files from workspace root into archived_outputs/ and leave symlinks")
    c.add_argument("--dry", action="store_true", help="dry run: show actions without moving")
    c.add_argument("--pattern", action="append", help="glob pattern to collect (can be used multiple times)")
    l = sub.add_parser("link", help="create a symlink or copy fallback: link <src> <dst>")
    l.add_argument("src")
    l.add_argument("dst")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "init-view":
        init_view()
        return 0
    if args.cmd == "collect":
        collect(dry=getattr(args, "dry", False), patterns=getattr(args, "pattern", None))
        return 0
    if args.cmd == "link":
        link(args.src, args.dst)
        return 0
    print("No command provided. Use --help for usage.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
