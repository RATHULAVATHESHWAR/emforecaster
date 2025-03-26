import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def lambda_analysis(dataset_file, output_path=None, dpi=600):
    """
    Create a continuous plot showing how Tradeoff Score varies with lambda for each model.

    Args:
        dataset_file (str): Path to CSV file containing model scores
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

    # Load dataset
    df = pd.read_csv(dataset_file)

    # Model display names
    model_labels = {
        "EMForecaster": "EMForecaster",
        "PatchTST": "PatchTST",
        "RF_EMF_Transformer": "Transformer",
        "RF_EMF_LSTM": "LSTM",
        "RF_EMF_GRU": "GRU",
        "RF_EMF_CNN": "CNN",
        "RF_EMF_MLP": "MLP",
        "TSMixer": "TSMixer",
        "RF_EMF_DLinear": "DLinear",
    }

    # Rename models using the mapping
    df["model"] = df["model"].replace(model_labels)

    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(20, 16))

    # Define colors
    colors = [
        "#00395E",  # Dark blue
        "#00A499",  # Teal
        "#8B1F5C",  # Burgundy
        "#FF8B3D",  # Orange
        "#4B0082",  # Indigo
        "#2E8B57",  # Sea Green
        "#DC143C",  # Crimson
        "#DAA520",  # Goldenrod
    ]

    # Create lambda values for x-axis
    lambda_values = np.linspace(0, 1, 100)

    # Define desired legend order
    legend_order = [
        "EMForecaster",
        "DLinear",
        "PatchTST",
        "TSMixer",
        "LSTM",
        "CNN",
        "MLP",
        "Transformer",
    ]

    # Sort DataFrame according to legend order
    df["legend_order"] = df["model"].map(
        {name: idx for idx, name in enumerate(legend_order)}
    )
    df = df.sort_values("legend_order")

    # Plot curve for each model (now in the desired order)
    for idx, (_, data) in enumerate(df.iterrows()):
        wac = data["WAC"]
        miw_zscore = data["MIW_zscore"]
        model_name = data["model"]

        print(f"Model: {model_name}, WAC: {wac}, MIW_zscore: {miw_zscore}")

        scores = [
            lambda_val * wac / 100 + (1 - lambda_val) * (1 / (1 + np.exp(miw_zscore)))
            for lambda_val in lambda_values
        ]

        ax.plot(
            lambda_values,
            scores,
            label=model_name,
            color=colors[idx % len(colors)],
            linewidth=6,
            zorder=3,
            alpha=0.8,
        )

    # Extract dataset name from output_path if provided, otherwise from input path
    dataset_name = None
    if output_path:
        for key, label in dataset_labels.items():
            if label in output_path:
                dataset_name = label
                break
    if dataset_name is None and dataset_file:
        for key, label in dataset_labels.items():
            if key in dataset_file:
                dataset_name = label
                break

    # Customize axes
    ax.set_xlabel(r"$\lambda$", labelpad=15)
    ax.set_ylabel(
        (
            r"\textbf{Tradeoff Score (" + dataset_name + ")}"
            if dataset_name
            else r"\textbf{Tradeoff Score}"
        ),
        labelpad=15,
    )

    # Make sure all borders are visible and properly styled
    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_linewidth(2.0)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_visible(True)

    # Add grid (moved before border settings to ensure grid stays behind)
    ax.grid(True, linestyle="--", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Set background colors
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Add legend - with multiple columns
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=36,
        handletextpad=0.5,
        ncol=2,  # Split into 2 columns
        columnspacing=1.0,  # Adjust spacing between columns
        bbox_to_anchor=(0.98, 0.98),
    )  # Fine-tune position (x, y)

    # Keep tight layout
    plt.tight_layout()

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

    parser = ArgumentParser(description="Create lambda analysis plots for each dataset")
    parser.add_argument("pred_len", type=int, help="Prediction length of forecast")
    parser.add_argument("alpha", type=float, help="Significance level")
    parser.add_argument(
        "datasets",
        nargs="+",
        type=str,
        help="List of datasets (e.g., rf_emf_det rf_emf_ptv)",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = f"../../../figures/general/conformal"
    os.makedirs(output_dir, exist_ok=True)

    # Dataset display names
    dataset_labels = {
        "rf_emf_det": "DEE 22'",
        "rf_emf_ptv": "UH 22'",
        "rf_emf_ptv2": "UH 23'",
        "rf_emf_tur": "PT",
        "rf_emf": "Turkey",
    }

    # Handle 'all' argument by expanding it to all datasets
    if args.datasets[0] == "all":  # Changed from if args.datasets == "all"
        args.datasets = [
            "rf_emf_det",
            "rf_emf_ptv",
            "rf_emf_ptv2",
            "rf_emf_tur",
            "rf_emf",
        ]

    # Create a plot for each dataset
    for dataset in args.datasets:
        # Generate input path for this dataset
        input_path = f"../neptune/results/conformal/tos_{dataset}_pred_len{args.pred_len}_alpha{args.alpha}.csv"

        # Generate output path for this dataset
        output_path = os.path.join(
            output_dir,
            f"lambda_analysis_{dataset_labels[dataset]}_pred_len{args.pred_len}_alpha{args.alpha}.eps",
        )

        # Create the plot
        lambda_analysis(input_path, output_path)
