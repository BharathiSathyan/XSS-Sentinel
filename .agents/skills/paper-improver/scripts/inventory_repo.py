#!/usr/bin/env python3
"""
inventory_repo.py

Deterministic first-pass scan of a codebase to find files likely to contain
experimental evidence (results, metrics, logs, configs, notebooks, figures).

Usage:
    python inventory_repo.py <repo_root> [--out repo_inventory.md]

Prints (and optionally writes) a markdown summary grouped by category, so an
agent can reason over a short, reliable list instead of browsing the repo
ad hoc.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Patterns are intentionally broad and cheap (name/extension based).
# Refine per-project if needed.
CATEGORY_PATTERNS = {
    "results_metrics": [
        ".json", ".csv", ".tsv", ".yaml", ".yml",
    ],
    "logs": [
        ".log", ".out", ".err",
    ],
    "notebooks": [
        ".ipynb",
    ],
    "figures_tables": [
        ".png", ".pdf", ".svg", ".eps",
    ],
    "checkpoints": [
        ".pt", ".pth", ".ckpt", ".h5", ".safetensors",
    ],
    "code": [
        ".py", ".sh",
    ],
}

# Directory/file name fragments that raise suspicion this file/folder is
# relevant to results (used to prioritize within the noisy "code"/"json" pools).
KEYWORDS = [
    "result", "results", "metric", "metrics", "eval", "evaluation",
    "log", "logs", "experiment", "exp", "ablation", "baseline",
    "checkpoint", "ckpt", "run", "runs", "output", "outputs",
    "score", "scores", "report", "table", "figure", "fig",
]

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "site-packages", ".ipynb_checkpoints",
}


def categorize(path: Path):
    ext = path.suffix.lower()
    for category, exts in CATEGORY_PATTERNS.items():
        if ext in exts:
            return category
    return None


def is_keyword_hit(path: Path) -> bool:
    lowered = str(path).lower()
    return any(kw in lowered for kw in KEYWORDS)


def scan(repo_root: Path):
    inventory = {cat: [] for cat in CATEGORY_PATTERNS}
    inventory["other_keyword_hits"] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(repo_root)
            category = categorize(fpath)
            keyword_hit = is_keyword_hit(fpath)

            if category:
                # Only keep code files if they look results-relevant, to avoid
                # dumping the entire source tree.
                if category == "code" and not keyword_hit:
                    continue
                try:
                    size = fpath.stat().st_size
                except OSError:
                    size = None
                inventory[category].append({
                    "path": str(rel),
                    "size_bytes": size,
                    "keyword_hit": keyword_hit,
                })
            elif keyword_hit:
                inventory["other_keyword_hits"].append({"path": str(rel)})

    return inventory


def to_markdown(inventory: dict, repo_root: Path) -> str:
    lines = [f"# Repo inventory: `{repo_root}`\n"]
    total = sum(len(v) for v in inventory.values())
    lines.append(f"Scanned files matched: **{total}**\n")

    order = [
        "results_metrics", "logs", "notebooks", "figures_tables",
        "checkpoints", "code", "other_keyword_hits",
    ]
    for category in order:
        items = inventory.get(category, [])
        if not items:
            continue
        lines.append(f"\n## {category.replace('_', ' ').title()} ({len(items)})\n")
        # Prioritize keyword hits first
        items_sorted = sorted(items, key=lambda x: not x.get("keyword_hit", False))
        for item in items_sorted[:200]:  # cap noisy categories
            tag = " ⭐" if item.get("keyword_hit") else ""
            size = item.get("size_bytes")
            size_str = f" ({size} bytes)" if size is not None else ""
            lines.append(f"- `{item['path']}`{size_str}{tag}")
        if len(items_sorted) > 200:
            lines.append(f"- ... and {len(items_sorted) - 200} more")

    lines.append(
        "\n\n_⭐ = filename/path contains a results-related keyword "
        "(result, metric, eval, log, experiment, ablation, baseline, "
        "checkpoint, run, output, score, report, table, figure). "
        "Review these first when building the claim-evidence map._"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=str, help="Path to the codebase root")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write markdown output")
    parser.add_argument("--json", action="store_true", help="Also print raw JSON instead of markdown")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        print(f"Error: {repo_root} does not exist", file=sys.stderr)
        sys.exit(1)

    inventory = scan(repo_root)

    if args.json:
        print(json.dumps(inventory, indent=2))
        return

    md = to_markdown(inventory, repo_root)
    print(md)

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\n[written to {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()