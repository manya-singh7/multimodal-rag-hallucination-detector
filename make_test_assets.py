"""Generate the deliberate caption-vs-image mismatch test images.

Run once:  python make_test_assets.py
Creates test_assets/ with:
  decline_chart.png  - revenue clearly FALLING   (used with the WRONG caption "revenue growth")
  growth_chart.png   - revenue clearly RISING    (control: caption matches)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEARS = [2019, 2020, 2021, 2022, 2023]


def make(path, values, title):
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=120)
    ax.plot(YEARS, values, marker="o", linewidth=2.5)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue ($M)")
    ax.set_xticks(YEARS)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("test_assets", exist_ok=True)
    make("test_assets/decline_chart.png", [120, 105, 88, 70, 52], "Annual Revenue")
    make("test_assets/growth_chart.png", [52, 70, 88, 105, 120], "Annual Revenue")
    print("wrote test_assets/decline_chart.png and growth_chart.png")
