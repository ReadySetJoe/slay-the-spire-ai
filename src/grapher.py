import json
import logging
import os

logger = logging.getLogger(__name__)

_ROLL_WINDOW = 20


def generate_graphs(
    log_path: str = "data/run_log.jsonl",
    scores_path: str = "data/card_scores.json",
    output_dir: str = "data/graphs",
) -> None:
    """Read run_log.jsonl and write a performance summary PNG to output_dir."""
    records = _load_jsonl(log_path)
    if len(records) < 2:
        logger.info("Not enough runs to graph (%d recorded).", len(records))
        return

    try:
        import matplotlib
        matplotlib.use("Agg")  # file-only backend, no display needed
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping graphs.")
        return

    os.makedirs(output_dir, exist_ok=True)

    n = len(records)
    run_nums = list(range(1, n + 1))
    floors    = [r["floor_reached"] for r in records]
    relics    = [r["relic_count"] for r in records]
    wins      = [1 if r["result"] == "win" else 0 for r in records]
    hp_ratio  = [r["current_hp"] / max(r["max_hp"], 1) for r in records]
    deck_sizes = [r["deck_size"] for r in records]

    w = min(_ROLL_WINDOW, max(1, n // 5))

    def rolling(values):
        arr = np.array(values, dtype=float)
        out = np.empty_like(arr)
        for i in range(len(arr)):
            out[i] = arr[max(0, i - w + 1):i + 1].mean()
        return out

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"Slay the Spire AI — {n} runs",
        fontsize=13, fontweight="bold",
    )

    def scatter_panel(ax, y, title, ylabel, color="steelblue"):
        ax.scatter(run_nums, y, alpha=0.25, s=8, color=color, zorder=2)
        ax.plot(run_nums, rolling(y), color="darkorange", linewidth=1.8,
                label=f"rolling avg (w={w})", zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Run #", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    scatter_panel(axes[0, 0], floors,     "Floor Reached",       "Floor",          "steelblue")
    scatter_panel(axes[0, 1], wins,       "Win Rate",            "Win (1) / Loss", "seagreen")
    scatter_panel(axes[1, 0], relics,     "Relics at Run End",   "Relics",         "mediumpurple")
    scatter_panel(axes[1, 1], hp_ratio,   "Final HP Ratio",      "HP / Max HP",    "tomato")
    scatter_panel(axes[2, 0], deck_sizes, "Deck Size at Run End","Cards",           "peru")

    _card_score_panel(axes[2, 1], scores_path)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "performance.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Performance graphs saved to %s", out_path)


def _card_score_panel(ax, scores_path: str) -> None:
    """Horizontal bar chart of top-5 and bottom-5 cards by EMA score."""
    if not os.path.exists(scores_path):
        ax.text(0.5, 0.5, "No card score data yet",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title("Card Scores (EMA)", fontsize=10)
        ax.axis("off")
        return

    with open(scores_path) as f:
        data = json.load(f)

    if not data:
        ax.axis("off")
        return

    sorted_cards = sorted(data.items(), key=lambda x: x[1].get("ema", 0.5))
    bottom = sorted_cards[:5]
    top = sorted_cards[-5:][::-1]  # best first
    combined = top + bottom

    labels = [f"{cid} ({d.get('picks', 0)}×)" for cid, d in combined]
    values = [d.get("ema", 0.5) for _, d in combined]
    colors = ["seagreen"] * len(top) + ["tomato"] * len(bottom)

    y_pos = range(len(combined))
    ax.barh(list(y_pos), values, color=colors, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("EMA Score", fontsize=9)
    ax.set_title("Card Scores — top 5 / bottom 5", fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
