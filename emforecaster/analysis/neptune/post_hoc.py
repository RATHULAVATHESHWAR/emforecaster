import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore")

# Dictionary for mapping parameter names to their display names
PARAM_DISPLAY_NAMES = {
    # To be filled in with mappings like:
    "parameters/sl/patch_embed_dim": "Patch Embedding Dimension",
    "parameters/data/patch_dim": "Patch Dimension",
    "parameters/tsmixer/d_model": "STB Hidden Dimension",
    "parameters/tsmixer/num_enc_layers": "Number of STB Blocks",
}


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def analyze_hyperparameter(df, param_name, model_name, metric, save_path):
    """
    Analyze the effect of a hyperparameter on a specified metric.
    """
    if param_name not in df.columns:
        print(f"Warning: {param_name} not found in DataFrame")
        return None

    # Set up plotting style with LaTeX
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "figure.figsize": (12, 8),
            "axes.titlesize": 30,
            "axes.labelsize": 27,
            "xtick.labelsize": 27,
            "ytick.labelsize": 24,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.5,
            "axes.grid": False,
        }
    )

    fig, ax = plt.subplots()
    box_plot = sns.boxplot(
        x=param_name, y=metric, data=df, ax=ax, palette="viridis", width=0.6
    )

    # Map metric name if it's sl_test_loss
    y_label = "MSE" if metric == "sl_test/loss" else metric

    # Map parameter name if it exists in dictionary, otherwise use original
    x_label = param_name.split("/")[-1]  # Default to last part of parameter path
    if param_name in PARAM_DISPLAY_NAMES:
        x_label = PARAM_DISPLAY_NAMES[param_name]

    # Plot customization
    ax.set_xlabel(rf"\textrm{{{x_label}}}", fontsize=27)
    ax.set_ylabel(rf"\textrm{{{y_label}}}", fontsize=27)

    # Set y-axis limits with padding
    y_min, y_max = df[metric].min(), df[metric].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

    # Add mean values above each boxplot
    for i, (name, group) in enumerate(df.groupby(param_name)):
        mean_val = group[metric].mean()

        # Get the maximum y value for this group (including outliers)
        max_y = group[metric].max()

        # Add small padding above the maximum value
        padding = 0.02 * y_range
        ax.text(
            i,
            max_y + padding,
            f"{mean_val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#555555",
            fontsize=24,
        )

    # Set x-ticks without rotation
    plt.xticks(rotation=0)
    ax.tick_params(axis="both", which="major", labelsize=24)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return df.groupby(param_name)[metric].describe()


def main():
    parser = ArgumentParser(
        description="Analyze hyperparameter effects on model performance"
    )
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument("model", type=str, help="Name of the model")
    args = parser.parse_args()

    # Load data
    results_path = f"results/{args.dataset}/{args.model}.csv"
    config_path = f"ablations/{args.dataset}/{args.model}.yaml"

    df = pd.read_csv(results_path)
    config = load_yaml(config_path)

    metric = config["deciding_metric"]

    # Only analyze parameters that are in PARAM_DISPLAY_NAMES
    results = {}
    for param in PARAM_DISPLAY_NAMES.keys():
        output_dir = f"../../../figures/post_hoc/{args.model}"
        os.makedirs(output_dir, exist_ok=True)
        save_path = f'{output_dir}/{param.replace("/", "_")}_boxplot.eps'
        result = analyze_hyperparameter(df, param, args.model, metric, save_path)
        if result is not None:
            results[param] = result
            print(f"\nAnalysis for {param}:")
            print(result)

    # Find best/worst configurations
    n_configs = 5
    best_configs = df.nsmallest(n_configs, metric)[
        list(PARAM_DISPLAY_NAMES.keys()) + [metric]
    ]
    worst_configs = df.nlargest(n_configs, metric)[
        list(PARAM_DISPLAY_NAMES.keys()) + [metric]
    ]

    print(f"\nTop {n_configs} Best Configurations:")
    print(best_configs)
    print(f"\nTop {n_configs} Worst Configurations:")
    print(worst_configs)


if __name__ == "__main__":
    main()
