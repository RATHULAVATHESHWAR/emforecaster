import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_clustered_barchart(dataset_files, dataset_names, output_path=None, dpi=600):
    """
    Create a clustered bar chart for comparing model performance across datasets.

    Args:
        dataset_files (list): List of paths to CSV files containing model scores
        dataset_names (list): List of dataset identifiers
        output_path (str, optional): Path to save the figure. If None, displays the plot
        dpi (int): DPI for saved figure
    """
    # Set figure style
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 20,
            "axes.labelsize": 48,
            "xtick.labelsize": 48,
            "ytick.labelsize": 48,
            "legend.fontsize": 36,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 2.0,
        }
    )

    # Load all datasets
    dataframes = []
    for file in dataset_files:
        df = pd.read_csv(file)
        # Rename EMForecaster to EMForecaster in the model column
        df["model"] = df["model"].replace("EMForecaster", "EMForecaster")
        df["model"] = df["model"].replace("PatchTST", "PatchTST")
        dataframes.append(df)

    # Get all unique models and calculate their average scores for sorting
    all_scores = {}
    for df in dataframes:
        for _, row in df.iterrows():
            model = row["model"]
            score = row["TOS"]
            if model not in all_scores:
                all_scores[model] = []
            all_scores[model].append(score)

    # Calculate average score for each model
    model_averages = {model: np.mean(scores) for model, scores in all_scores.items()}

    # Sort models by their average score, but always put EMForecaster first
    sorted_models = sorted(
        model_averages.keys(), key=lambda x: model_averages[x], reverse=True
    )
    if "EMForecaster" in sorted_models:
        sorted_models.remove("EMForecaster")
        sorted_models.insert(0, "EMForecaster")

    # Setup plot dimensions
    n_models = len(sorted_models)
    n_datasets = len(dataset_names)
    bar_width = 0.18

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))

    # Define colors (matching your reference)
    colors = [
        "#00395E",  # Dark blue
        "#00A499",  # Teal
        "#8B1F5C",  # Burgundy
        "#FF8B3D",  # Orange
        "#4B0082",  # Indigo
        "#2E8B57",  # Sea Green
    ]

    # Dataset display names
    # dataset_labels = {
    #     "rf_emf_det": "Department of Electronic Engineering 2022 (Rome, Italy)",
    #     "rf_emf_det2": "Department of Electronic Engineering 2023 (Rome, Italy)",
    #     "rf_emf_ptv": "University Hospital 2022 (Rome, Italy)",
    #     "rf_emf_ptv2": "University Hospital 2023 (Rome, Italy)",
    #     "rf_emf_tur": "Polytechnic of Turin (Turin, Italy)",
    #     "rf_emf": "Altinordu District (Ordu City, Turkey)"
    # }
    dataset_labels = {
        "rf_emf_det": "DEE 22'",
        "rf_emf_det2": "DEE 23'",
        "rf_emf_ptv": "UH 22'",
        "rf_emf_ptv2": "UH 23'",
        "rf_emf_tur": "PT",
        "rf_emf": "Turkey",
    }

    # Plot bars for each dataset
    for idx, (df, dataset) in enumerate(zip(dataframes, dataset_names)):
        # Create dictionary for quick TOS score lookup
        scores = dict(zip(df["model"], df["TOS"]))

        # Calculate x positions for this dataset's bars
        x_positions = (
            np.arange(n_models) * (n_datasets * bar_width * 2) + idx * bar_width
        )

        # Get TOS scores for all models (0 if model not in dataset)
        tos_scores = [scores.get(model, 0) for model in sorted_models]

        # Create bars
        ax.bar(
            x_positions,
            tos_scores,
            bar_width,
            label=dataset_labels[dataset],
            color=colors[idx % len(colors)],
            alpha=1.0,
            zorder=3,
        )

    # Customize axes with rotated labels
    ax.set_xticks(
        np.arange(n_models) * (n_datasets * bar_width * 2)
        + (bar_width * (n_datasets - 1) / 2)
    )
    ax.set_xticklabels(
        [model.replace("RF_EMF_", "") for model in sorted_models],
        rotation=30,
        ha="center",
        va="top",
    )

    # Add more bottom padding to accommodate rotated labels
    plt.tight_layout(
        rect=[0, 0.15, 0.8, 1]
    )  # Increased bottom padding from 0.05 to 0.15

    # Customize axes
    ax.set_ylabel(r"\textbf{Tradeoff Score}", labelpad=15)

    # Make sure all borders are visible and properly styled
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_linewidth(2.0)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_visible(True)

    # Add grid (moved before border settings to ensure grid stays behind)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Set background colors
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Add legend with more spacing
    ax.legend(
        loc="upper center",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="gray",
        bbox_to_anchor=(0.5, 1.15),  # Reduced spacing between plot and legend
        ncol=len(dataset_names),
    )

    # Adjust layout to accommodate top legend
    plt.tight_layout()

    # Add more extra space on top for the legend
    plt.subplots_adjust(top=0.85)  # Increased to reduce space between plot and legend

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser(
        description="Create a clustered bar chart of model performance scores"
    )
    parser.add_argument("--pred_len", type=int, help="Prediction length of forecast")
    parser.add_argument("--alpha", type=float, help="Significance level")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help="List of datasets (e.g., rf_emf_det rf_emf_ptv)",
    )
    args = parser.parse_args()

    # Generate file paths
    input_paths = [
        f"../neptune/results/conformal/tos_{dataset}_pred_len{args.pred_len}_alpha{args.alpha}.csv"
        for dataset in args.datasets
    ]

    # Create output directory if it doesn't exist
    output_dir = f"../../../figures/general/conformal"
    os.makedirs(output_dir, exist_ok=True)

    # Generate output path
    output_path = os.path.join(
        output_dir, f"tos_comparison_pred_len{args.pred_len}_alpha{args.alpha}.eps"
    )

    # Create the plot
    create_clustered_barchart(input_paths, args.datasets, output_path)
