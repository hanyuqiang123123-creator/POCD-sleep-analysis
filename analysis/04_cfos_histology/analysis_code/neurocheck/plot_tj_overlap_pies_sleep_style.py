from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


INPUT_XLSX = Path(r"F:\Sleep\eXdata\NeuroCheck\TJ.xlsx")
OUTDIR = Path(r"F:\Sleep\outputs\neurocheck_tj_pies")
BASE = OUTDIR / "Fig4H_TJ_overlap_pies_sleep_style"

BLUE = "#4DBBD5"
ORANGE = "#F39B7F"
WHITE = "#FFFFFF"
BLACK = "#000000"


def read_sheet_summary(sheet_name: str, marker_col: str, overlap_col: str) -> dict:
    df = pd.read_excel(INPUT_XLSX, sheet_name=sheet_name, header=None)
    header_row = df.iloc[1].tolist()
    data = df.iloc[2:].copy()
    data.columns = header_row
    sum_row = data[data["NO."] == "SUM"].iloc[0]
    cfos_total = float(sum_row["C-FOS"])
    marker_total = float(sum_row[marker_col])
    overlap = float(sum_row[overlap_col])
    return {
        "marker": sheet_name,
        "cfos_total": cfos_total,
        "marker_total": marker_total,
        "overlap": overlap,
        "overlap_over_cfos": overlap / cfos_total,
        "overlap_over_marker": overlap / marker_total,
    }


def draw_pie(ax, fraction: float, color: str, startangle: float, text_xy=(0, -0.05), text_color=BLACK):
    wedges, texts = ax.pie(
        [fraction, 1 - fraction],
        colors=[color, WHITE],
        startangle=startangle,
        counterclock=False,
        wedgeprops={"edgecolor": BLACK, "linewidth": 0.55},
    )
    for text in texts:
        text.set_visible(False)
    ax.text(
        text_xy[0],
        text_xy[1],
        f"{fraction * 100:.2f}%",
        ha="center",
        va="center",
        fontsize=3.8,
        color=text_color,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()


def add_legend_pair(ax, y: float, marker: str):
    x0 = 0.02
    sw = 0.055
    sh = 0.12
    ax.add_patch(plt.Rectangle((x0, y), sw, sh, color=BLUE, transform=ax.transAxes, clip_on=False))
    ax.text(
        x0 + sw + 0.018,
        y + sh / 2,
        f"{marker}+ overlay c-fos+/c-fos+",
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=3.8,
    )
    ax.add_patch(plt.Rectangle((x0, y - 0.36), sw, sh, color=ORANGE, transform=ax.transAxes, clip_on=False))
    ax.text(
        x0 + sw + 0.018,
        y - 0.36 + sh / 2,
        f"{marker}+ overlay c-fos+/{marker}+",
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=3.8,
    )


def save_ai_clean(fig, base: Path):
    for artist in fig.findobj():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)
        if hasattr(artist, "set_clip_path"):
            artist.set_clip_path(None)
    fig.savefig(f"{base}_ai_clean.svg", facecolor="white", edgecolor="none")
    fig.savefig(f"{base}_ai_clean.pdf", facecolor="white", edgecolor="none")
    fig.savefig(f"{base}_ai_clean.png", dpi=600, facecolor="white", edgecolor="none")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summaries = [
        read_sheet_summary("CART", "CART", "C-FOS/CART"),
        read_sheet_summary("CCK", "CCK", "C-FOS/CCK"),
    ]
    source_rows = []
    for item in summaries:
        source_rows.extend(
            [
                {
                    "marker": item["marker"],
                    "pie": f"{item['marker']}+ overlay c-fos+ / c-fos+",
                    "numerator_overlap": item["overlap"],
                    "denominator": item["cfos_total"],
                    "fraction": item["overlap_over_cfos"],
                    "percent": item["overlap_over_cfos"] * 100,
                    "color": BLUE,
                },
                {
                    "marker": item["marker"],
                    "pie": f"{item['marker']}+ overlay c-fos+ / {item['marker']}+",
                    "numerator_overlap": item["overlap"],
                    "denominator": item["marker_total"],
                    "fraction": item["overlap_over_marker"],
                    "percent": item["overlap_over_marker"] * 100,
                    "color": ORANGE,
                },
            ]
        )
    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(OUTDIR / "Fig4H_TJ_overlap_pies_sleep_style_source_data.csv", index=False)
    source_df.to_csv(OUTDIR / "Fig4H_TJ_overlap_pies_sleep_style_stats.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.compression": 0,
            "font.size": 4.2,
        }
    )

    cart, cck = summaries
    fig = plt.figure(figsize=(2.18, 1.92))
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1, 0.42, 1, 0.42],
        width_ratios=[1, 0.08, 1, 0.02],
        left=0.08,
        right=0.98,
        top=0.93,
        bottom=0.06,
        wspace=0.38,
        hspace=0.08,
    )
    ax_label = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_label.set_axis_off()
    ax_label.text(0.005, 0.965, "H", fontsize=5.8, fontweight="bold", ha="left", va="top")

    ax_cart_cfos = fig.add_subplot(gs[0, 0])
    ax_cart_marker = fig.add_subplot(gs[0, 2])
    ax_cart_legend = fig.add_subplot(gs[1, 0:3])
    ax_cart_legend.set_axis_off()

    ax_cck_cfos = fig.add_subplot(gs[2, 0])
    ax_cck_marker = fig.add_subplot(gs[2, 2])
    ax_cck_legend = fig.add_subplot(gs[3, 0:3])
    ax_cck_legend.set_axis_off()

    draw_pie(ax_cart_cfos, cart["overlap_over_cfos"], BLUE, startangle=88, text_xy=(0, -0.07))
    draw_pie(
        ax_cart_marker,
        cart["overlap_over_marker"],
        ORANGE,
        startangle=90,
        text_xy=(0.02, -0.12),
        text_color=WHITE,
    )
    add_legend_pair(ax_cart_legend, 0.68, "CART")

    draw_pie(ax_cck_cfos, cck["overlap_over_cfos"], BLUE, startangle=88, text_xy=(0.28, -0.08))
    draw_pie(
        ax_cck_marker,
        cck["overlap_over_marker"],
        ORANGE,
        startangle=90,
        text_xy=(0.0, -0.14),
        text_color=WHITE,
    )
    add_legend_pair(ax_cck_legend, 0.68, "CCK")

    fig.savefig(f"{BASE}.svg", bbox_inches="tight")
    fig.savefig(f"{BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{BASE}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{BASE}.tiff", dpi=600, bbox_inches="tight")
    save_ai_clean(fig, BASE)
    plt.close(fig)

    print(source_df.to_string(index=False))
    print(f"Wrote {BASE}.*")


if __name__ == "__main__":
    main()
