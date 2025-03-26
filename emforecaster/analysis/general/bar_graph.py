import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_tos_barchart(csv_file, dataset, output_path=None, dpi=600):
    """
    Create a professional bar chart visualization of TOS scores.
    Args:
        csv_file (str): Path to the CSV file containing model scores
        dataset (str): Dataset name to look up title from map
        output_path (str, optional): Path to save the figure. If None, displays the plot
        dpi (int): DPI for saved figure
    """
    # Dataset mapping
    map = {
        "rf_emf_det": {
            "title": "Department of Electronic Engineering 2022 (Rome, Italy)",
        },
        "rf_emf_det2": {
            "title": "Department of Electronic Engineering 2023 (Rome, Italy)",
        },
        "rf_emf_ptv": {
            "title": "Unversity Hospital 2022 (Rome, Italy)",
        },
        "rf_emf_ptv2": {
            "title": "Unversity Hospital 2023 (Rome, Italy)",
        },
        "rf_emf_tur": {
            "title": "Polytechnic of Turin (Turin, Italy)",
        },
        "rf_emf_tur2": {
            "title": "Train Station (Turin, Italy)",
        },
        "rf_emf": {
            "title": "Altinordu District (Ordu City, Turkey)",
        },
    }

    # Set style parameters
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 20,  # Increased from 16
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
        }
    )

    # Read and sort data
    df = pd.read_csv(csv_file)
    df = df.sort_values("TOS", ascending=True)  # Sort by TOS score

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create bars
    bars = ax.barh(range(len(df)), df["TOS"], height=0.6)

    # Customize colors - use a professional color palette
    colors = [
        "#2978A0",
        "#315F8D",
        "#393E7B",
        "#2B4162",
        "#1E325C",
        "#142952",
        "#0B1F47",
    ]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.85)

    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=20,
        )

    # Customize axis
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [name.replace("RF_EMF_", "") for name in df["model"]], fontsize=20
    )  # Increased from 16

    # Set title and labels with dataset name
    title = f"{map[dataset]['title']}"
    ax.set_title(r"\textbf{" + title + "}", pad=20)
    ax.set_xlabel(r"\textbf{Tradeoff Score}")
    ax.tick_params(axis="x", labelsize=20)

    # Customize grid
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Set x-axis limits with some padding
    ax.set_xlim(0, max(df["TOS"]) * 1.15)

    # Add subtle background color
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    # Adjust layout
    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    from argparse import ArgumentParser
    import os

    # Parse arguments
    parser = ArgumentParser(description="Create a professional bar chart of TOS scores")
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset. Options: 'rf_emf_det', 'rf_emf_ptv', 'rf_emf_ptv2', 'rf_emf_tur'",
    )
    parser.add_argument("pred_len", type=int, help="Prediction length of forecast")
    parser.add_argument("alpha", type=float, help="Significance level")
    args = parser.parse_args()

    input_path = f"../neptune/results/conformal/tos_{args.dataset}_pred_len{args.pred_len}_alpha{args.alpha}.csv"
    output_dir = f"../../../figures/general/conformal"
    output_path = os.path.join(
        output_dir, f"tos_{args.dataset}_pred_len{args.pred_len}_alpha{args.alpha}.eps"
    )
    os.makedirs(output_dir, exist_ok=True)
    create_tos_barchart(input_path, args.dataset, output_path)
