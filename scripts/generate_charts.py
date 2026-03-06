"""
generate_charts.py
Produces all business insight charts from the Trustpilot dataset.
Output: charts/ directory (PNG files)
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "data.csv"
CATEGORIES_CSV = ROOT / "data" / "catgories.csv"
CHARTS_DIR = ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BRAND_BLUE = "#1A4A8A"
BRAND_GREEN = "#2ECC71"
BRAND_RED = "#E74C3C"
BRAND_ORANGE = "#E67E22"
BRAND_GRAY = "#95A5A6"
BRAND_DARK = "#2C3E50"

PALETTE_COOL = ["#1A4A8A", "#2980B9", "#3498DB", "#85C1E9", "#AED6F1"]
PALETTE_TRAFFIC = [BRAND_RED, BRAND_ORANGE, BRAND_GRAY, BRAND_GREEN, BRAND_BLUE]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.color": "#DEE2E6",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_data():
    rows = []
    with DATA_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_category_meta():
    meta = {}
    with CATEGORIES_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            meta[row["category_id"]] = row
    return meta


def deduplicate(rows):
    seen = set()
    unique = []
    for r in rows:
        bid = r["business_unit_id"]
        if bid and bid not in seen:
            seen.add(bid)
            unique.append(r)
    return unique


def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def safe_int(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def trust_bucket(ts):
    if ts >= 4.5:
        return "4.5 – 5.0\nExcellent"
    elif ts >= 4.0:
        return "4.0 – 4.4\nGood"
    elif ts >= 3.0:
        return "3.0 – 3.9\nAverage"
    elif ts >= 2.0:
        return "2.0 – 2.9\nPoor"
    else:
        return "0.1 – 1.9\nBad"


TRUST_BUCKET_ORDER = [
    "0.1 – 1.9\nBad",
    "2.0 – 2.9\nPoor",
    "3.0 – 3.9\nAverage",
    "4.0 – 4.4\nGood",
    "4.5 – 5.0\nExcellent",
]


def save(fig, name):
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Chart 1 — Market Size by Top-Level Category
# ---------------------------------------------------------------------------
def chart_market_size(unique_rows, cat_meta):
    cat_to_parent = {}
    parent_name = {}
    for cid, m in cat_meta.items():
        if m["level"] == "top":
            cat_to_parent[cid] = cid
            parent_name[cid] = m["display_name"]
        else:
            cat_to_parent[cid] = m.get("parent_category_id", cid)

    counts = defaultdict(int)
    for r in unique_rows:
        p = cat_to_parent.get(r["category_id"])
        if p and p in parent_name:
            counts[parent_name[p]] += 1

    labels, values = zip(*sorted(counts.items(), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(labels, values, color=BRAND_BLUE, height=0.65)

    for bar, v in zip(bars, values):
        ax.text(v + 80, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", ha="left", fontsize=9, color=BRAND_DARK)

    ax.set_xlabel("Number of Businesses Listed")
    ax.set_title("Chart 1 — How Many Businesses Compete in Each Sector?")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    save(fig, "01_market_size_by_category.png")


# ---------------------------------------------------------------------------
# Chart 2 — Average Trust Score by Top-Level Category
# ---------------------------------------------------------------------------
def chart_trust_by_category(unique_rows, cat_meta):
    cat_to_parent = {}
    parent_name = {}
    for cid, m in cat_meta.items():
        if m["level"] == "top":
            cat_to_parent[cid] = cid
            parent_name[cid] = m["display_name"]
        else:
            cat_to_parent[cid] = m.get("parent_category_id", cid)

    scores = defaultdict(list)
    for r in unique_rows:
        ts = safe_float(r["trust_score"])
        if ts and ts > 0:
            p = cat_to_parent.get(r["category_id"])
            if p and p in parent_name:
                scores[parent_name[p]].append(ts)

    avg = {k: sum(v) / len(v) for k, v in scores.items() if len(v) >= 50}
    labels, values = zip(*sorted(avg.items(), key=lambda x: x[1]))

    colors = [BRAND_RED if v < 3.4 else BRAND_ORANGE if v < 3.6 else BRAND_BLUE for v in values]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(labels, values, color=colors, height=0.65)

    for bar, v in zip(bars, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", ha="left", fontsize=9, color=BRAND_DARK)

    ax.axvline(x=3.5, color=BRAND_GRAY, linewidth=1.2, linestyle="--", label="Sector median (3.5)")
    ax.set_xlabel("Average Trust Score (0 – 5)")
    ax.set_title("Chart 2 — Which Sectors Earn Customer Trust and Which Underperform?")
    ax.set_xlim(0, 5.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "02_avg_trust_score_by_category.png")


# ---------------------------------------------------------------------------
# Chart 3 — Trust Score Distribution (all active businesses)
# ---------------------------------------------------------------------------
def chart_trust_distribution(unique_rows):
    bucket_counts = Counter()
    for r in unique_rows:
        ts = safe_float(r["trust_score"])
        if ts and ts > 0:
            bucket_counts[trust_bucket(ts)] += 1

    labels = TRUST_BUCKET_ORDER
    values = [bucket_counts.get(l, 0) for l in labels]
    colors = PALETTE_TRAFFIC

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 300,
                f"{v:,}", ha="center", va="bottom", fontsize=10, color=BRAND_DARK)

    total = sum(values)
    ax.set_ylabel("Number of Businesses")
    ax.set_title("Chart 3 — The Trust Landscape: Where Do Businesses Stand Today?")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Annotate pct above bar
    for bar, v in zip(bars, values):
        pct = 100 * v / total
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2500,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=9, color="#666666")

    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    save(fig, "03_trust_score_distribution.png")


# ---------------------------------------------------------------------------
# Chart 4 — Review Volume Segmentation (maturity ladder)
# ---------------------------------------------------------------------------
def chart_review_segmentation(unique_rows):
    segments = [
        ("No Reviews\n(0)", 0, 0),
        ("Emerging\n(1 – 9)", 1, 9),
        ("Developing\n(10 – 99)", 10, 99),
        ("Established\n(100 – 999)", 100, 999),
        ("Leading\n(1k – 10k)", 1000, 9999),
        ("Dominant\n(10k+)", 10000, 10**9),
    ]
    counts = Counter()
    for r in unique_rows:
        n = safe_int(r["number_of_reviews"])
        for label, lo, hi in segments:
            if lo <= n <= hi:
                counts[label] += 1
                break

    labels = [s[0] for s in segments]
    values = [counts[l] for l in labels]
    colors = [BRAND_GRAY, "#E8A87C", BRAND_ORANGE, "#27AE60", BRAND_BLUE, BRAND_DARK]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    total = sum(values)
    for bar, v in zip(bars, values):
        pct = 100 * v / total
        ax.text(bar.get_x() + bar.get_width() / 2, v + 400,
                f"{v:,}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9.5, color=BRAND_DARK)

    ax.set_ylabel("Number of Businesses")
    ax.set_title("Chart 4 — The Review Maturity Ladder: How Engaged Are Businesses?")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(values) * 1.22)
    fig.tight_layout()
    save(fig, "04_review_maturity_ladder.png")


# ---------------------------------------------------------------------------
# Chart 5 — No-Review Rate by Top-Level Category
# ---------------------------------------------------------------------------
def chart_no_review_rate(unique_rows, cat_meta):
    cat_to_parent = {}
    parent_name = {}
    for cid, m in cat_meta.items():
        if m["level"] == "top":
            cat_to_parent[cid] = cid
            parent_name[cid] = m["display_name"]
        else:
            cat_to_parent[cid] = m.get("parent_category_id", cid)

    total = defaultdict(int)
    no_rev = defaultdict(int)
    for r in unique_rows:
        p = cat_to_parent.get(r["category_id"])
        if not p or p not in parent_name:
            continue
        name = parent_name[p]
        total[name] += 1
        if safe_int(r["number_of_reviews"]) == 0:
            no_rev[name] += 1

    rates = {k: 100 * no_rev[k] / total[k] for k in total if total[k] >= 50}
    labels, values = zip(*sorted(rates.items(), key=lambda x: x[1]))
    colors = [BRAND_RED if v >= 50 else BRAND_ORANGE if v >= 35 else BRAND_BLUE for v in values]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(labels, values, color=colors, height=0.65)

    for bar, v in zip(bars, values):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", ha="left", fontsize=9, color=BRAND_DARK)

    ax.axvline(x=30, color=BRAND_GRAY, linewidth=1.2, linestyle="--", label="30% threshold")
    ax.set_xlabel("Share of Businesses With Zero Reviews (%)")
    ax.set_title("Chart 5 — Untapped Potential: Sectors With the Highest Review Gaps")
    ax.set_xlim(0, 80)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "05_no_review_rate_by_category.png")


# ---------------------------------------------------------------------------
# Chart 6 — Recommended Rate by Trust Score Band
# ---------------------------------------------------------------------------
def chart_recommended_rate(unique_rows):
    rec = defaultdict(lambda: {"rec": 0, "total": 0})
    for r in unique_rows:
        ts = safe_float(r["trust_score"])
        if not ts or ts <= 0:
            continue
        b = trust_bucket(ts)
        rec[b]["total"] += 1
        if r["is_recommended_in_categories"] == "True":
            rec[b]["rec"] += 1

    labels = TRUST_BUCKET_ORDER
    values = [100 * rec[l]["rec"] / rec[l]["total"] if rec[l]["total"] else 0 for l in labels]
    totals = [rec[l]["total"] for l in labels]
    colors = PALETTE_TRAFFIC

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    for bar, v, n in zip(bars, values, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                f"{v:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=9, color=BRAND_DARK)

    ax.set_ylabel("% Recommended by Trustpilot")
    ax.set_title("Chart 6 — The Recommendation Threshold: Trust Score Determines Visibility")
    ax.set_ylim(0, 32)
    fig.tight_layout()
    save(fig, "06_recommended_rate_by_trust_band.png")


# ---------------------------------------------------------------------------
# Chart 7 — Average Review Count by Trust Score Band
# ---------------------------------------------------------------------------
def chart_reviews_vs_trust(unique_rows):
    rev_by_band = defaultdict(list)
    for r in unique_rows:
        ts = safe_float(r["trust_score"])
        if not ts or ts <= 0:
            continue
        n = safe_int(r["number_of_reviews"])
        rev_by_band[trust_bucket(ts)].append(n)

    labels = TRUST_BUCKET_ORDER
    medians = [int(np.median(rev_by_band[l])) if rev_by_band[l] else 0 for l in labels]
    avgs = [int(np.mean(rev_by_band[l])) if rev_by_band[l] else 0 for l in labels]

    x = np.arange(len(labels))
    width = 0.38
    colors_avg = PALETTE_TRAFFIC
    colors_med = [c + "99" for c in PALETTE_TRAFFIC]  # transparent version not easy; use lighter

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_avg = ax.bar(x - width / 2, avgs, width, label="Average reviews", color=PALETTE_TRAFFIC)
    bars_med = ax.bar(x + width / 2, medians, width, label="Median reviews", color=BRAND_GRAY, alpha=0.7)

    for bar, v in zip(bars_avg, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 10,
                f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=BRAND_DARK)
    for bar, v in zip(bars_med, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 10,
                f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=BRAND_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Chart 7 — High-Trust Businesses Attract Dramatically More Reviews")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    fig.tight_layout()
    save(fig, "07_review_volume_vs_trust_score.png")


# ---------------------------------------------------------------------------
# Chart 8 — Top 15 Businesses by Review Volume
# ---------------------------------------------------------------------------
def chart_top_businesses(unique_rows):
    active = [r for r in unique_rows if safe_int(r["number_of_reviews"]) > 0]
    top = sorted(active, key=lambda r: safe_int(r["number_of_reviews"]), reverse=True)[:15]

    labels = [r["display_name"] for r in top]
    values = [safe_int(r["number_of_reviews"]) for r in top]
    scores = [safe_float(r["trust_score"]) or 0 for r in top]

    colors = [BRAND_GREEN if s >= 4.5 else BRAND_BLUE if s >= 4.0 else BRAND_ORANGE for s in scores]

    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.65)

    for bar, v, s in zip(bars, values[::-1], scores[::-1]):
        ax.text(v + 5000, bar.get_y() + bar.get_height() / 2,
                f"{v:,}  (score {s})", va="center", ha="left", fontsize=9, color=BRAND_DARK)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BRAND_GREEN, label="Score 4.5+ (Excellent)"),
        Patch(facecolor=BRAND_BLUE, label="Score 4.0–4.4 (Good)"),
        Patch(facecolor=BRAND_ORANGE, label="Score < 4.0"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.set_xlabel("Total Number of Reviews")
    ax.set_title("Chart 8 — The Review Leaders: Who Has Built the Largest Review Presence?")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, max(values) * 1.3)
    fig.tight_layout()
    save(fig, "08_top_15_businesses_by_reviews.png")


# ---------------------------------------------------------------------------
# Chart 9 — Stars Rating Distribution (stacked: active vs no-review)
# ---------------------------------------------------------------------------
def chart_stars_distribution(unique_rows):
    star_labels = ["1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"]
    star_counts = Counter()
    for r in unique_rows:
        s = r["stars"]
        if s in star_labels:
            star_counts[s] += 1

    values = [star_counts.get(s, 0) for s in star_labels]
    x_labels = ["1★", "1.5★", "2★", "2.5★", "3★", "3.5★", "4★", "4.5★", "5★"]

    # Color: red for low, blue for high
    star_colors = [
        BRAND_RED, BRAND_RED, BRAND_ORANGE, BRAND_ORANGE,
        BRAND_GRAY, BRAND_BLUE, BRAND_BLUE, BRAND_GREEN, BRAND_GREEN,
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x_labels, values, color=star_colors, width=0.65)

    total = sum(values)
    for bar, v in zip(bars, values):
        pct = 100 * v / total
        ax.text(bar.get_x() + bar.get_width() / 2, v + 200,
                f"{v:,}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8.5, color=BRAND_DARK)

    ax.set_ylabel("Number of Businesses")
    ax.set_title("Chart 9 — Star Rating Breakdown: The Full Quality Spectrum")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(values) * 1.22)
    fig.tight_layout()
    save(fig, "09_stars_rating_distribution.png")


# ---------------------------------------------------------------------------
# Chart 10 — Subcategory Deep-Dive: Trust vs. Review Density (bubble-style bar)
# ---------------------------------------------------------------------------
def chart_subcategory_trust_vs_activity(unique_rows, cat_meta):
    """
    For the top 20 subcategories by business count, show avg trust score
    and color by review engagement rate (% with >9 reviews).
    """
    sub_rows = defaultdict(list)
    for r in unique_rows:
        m = cat_meta.get(r["category_id"])
        if m and m["level"] == "sub":
            sub_rows[m["display_name"]].append(r)

    # Keep top 20 by count
    top_subs = sorted(sub_rows.items(), key=lambda x: len(x[1]), reverse=True)[:20]

    labels = []
    avg_scores = []
    engagement_rates = []

    for name, rrows in top_subs:
        scored = [safe_float(r["trust_score"]) for r in rrows if safe_float(r["trust_score"]) and safe_float(r["trust_score"]) > 0]
        avg_ts = sum(scored) / len(scored) if scored else 0
        engaged = sum(1 for r in rrows if safe_int(r["number_of_reviews"]) >= 10)
        eng_rate = 100 * engaged / len(rrows) if rrows else 0
        labels.append(name)
        avg_scores.append(avg_ts)
        engagement_rates.append(eng_rate)

    # Sort by avg trust score
    order = sorted(range(len(labels)), key=lambda i: avg_scores[i])
    labels = [labels[i] for i in order]
    avg_scores = [avg_scores[i] for i in order]
    engagement_rates = [engagement_rates[i] for i in order]

    norm = plt.Normalize(min(engagement_rates), max(engagement_rates))
    cmap = plt.cm.RdYlGn
    colors = [cmap(norm(e)) for e in engagement_rates]

    fig, ax = plt.subplots(figsize=(13, 9))
    bars = ax.barh(labels, avg_scores, color=colors, height=0.65)

    for bar, v, e in zip(bars, avg_scores, engagement_rates):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}  ({e:.0f}% active)", va="center", ha="left", fontsize=8.5, color=BRAND_DARK)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("Review Engagement Rate (%\nwith 10+ reviews)", fontsize=9)

    ax.axvline(x=3.5, color=BRAND_GRAY, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Average Trust Score")
    ax.set_title("Chart 10 — Subcategory Intelligence: Trust Score vs. Review Engagement")
    ax.set_xlim(0, 5.5)
    fig.tight_layout()
    save(fig, "10_subcategory_trust_vs_engagement.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    rows = load_data()
    cat_meta = load_category_meta()
    unique_rows = deduplicate(rows)

    print(f"  Total rows       : {len(rows):,}")
    print(f"  Unique businesses: {len(unique_rows):,}")
    print(f"  Categories       : {len(cat_meta):,}")
    print()
    print("Generating charts...")

    chart_market_size(unique_rows, cat_meta)
    chart_trust_by_category(unique_rows, cat_meta)
    chart_trust_distribution(unique_rows)
    chart_review_segmentation(unique_rows)
    chart_no_review_rate(unique_rows, cat_meta)
    chart_recommended_rate(unique_rows)
    chart_reviews_vs_trust(unique_rows)
    chart_top_businesses(unique_rows)
    chart_stars_distribution(unique_rows)
    chart_subcategory_trust_vs_activity(unique_rows, cat_meta)

    print()
    print(f"Done. All charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
