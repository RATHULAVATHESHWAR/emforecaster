import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)


def plot_amplitude_distributions(
    data_list: list[np.ndarray],
    names: list[str] = None,
    y_axis: str = "Density",
    x_axis: str = "EMF Exposure (V/m)",
    title: str = "EMF Amplitude Distribution",
    dpi: int = 600,
    colors: list[str] = None,
    alpha: float = 0.6,
    output_format: str = "eps",
    legend: bool = True,
    dataset_name: str = None,
) -> None:
    """
    Create a comparative histogram of multiple time series amplitudes.

    Args:
        data_list (list[np.ndarray]): List of 1D numpy arrays to analyze
        names (list[str], optional): List of names for each array. If None, will use indices
        y_axis (str): Label for y-axis. Default is "Density"
        title (str): Plot title
        dpi (int): DPI for saving plots
        colors (list[str], optional): List of colors for each series. If None, uses default color cycle
        alpha (float): Transparency for histograms. Default is 0.6
    """
    # Create output directory
    output_dir = os.path.abspath(
        os.path.join(os.getcwd(), "..", "..", "..", "figures", "general")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Use indices if names are not provided
    if names is None:
        names = [f"Series {i+1}" for i in range(len(data_list))]

    # Default colors if none provided (expanded colorblind-friendly palette)
    if colors is None:

        # colors = [
        #     '#4477AA',  # Blue
        #     '#EE6677',  # Red
        #     '#228833',  # Green
        #     '#CCBB44',  # Yellow
        #     '#66CCEE',  # Cyan
        #     '#AA3377',  # Purple
        # ]

        base_colors = [
            "#1f77b4",  # Steel Blue
            "#ff7f0e",  # Safety Orange
            "#2ca02c",  # Forest Green
            "#d62728",  # Brick Red
            "#9467bd",  # Medium Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf",  # Cyan
            "#aec7e8",  # Light Blue
            "#ffbb78",  # Light Orange
            "#98df8a",  # Light Green
            "#ff9896",  # Light Red
            "#c5b0d5",  # Light Purple
            "#c49c94",  # Light Brown
            "#f7b6d2",  # Light Pink
            "#c7c7c7",  # Light Gray
            "#dbdb8d",  # Light Olive
            "#9edae5",  # Light Cyan
            "#393b79",  # Dark Blue
            "#7b4173",  # Dark Purple
            "#a55194",  # Medium Purple-Pink
        ]
        colors = base_colors[: len(data_list)]

    # Ensure we have enough colors for all datasets
    if len(colors) < len(data_list):
        raise ValueError(
            f"Not enough colors provided. Need {len(data_list)} colors but only {len(colors)} were provided."
        )

    # Calculate global min and max for consistent bins
    global_min = min(np.min(series) for series in data_list)
    global_max = max(np.max(series) for series in data_list)

    # Create figure with wider ratio
    plt.figure(figsize=(8, 6))

    # Plot histograms
    for i, (data, name, color) in enumerate(zip(data_list, names, colors)):
        plt.hist(
            data,
            bins=30,
            density=True,  # Normalize to probability density
            alpha=alpha,
            color=color,
            label=name,
            edgecolor="black",
            linewidth=0.5,
        )

    # Customize plot
    plt.title(rf"\textbf{{{title} ({dataset_name})}}", fontsize=20, pad=20)
    plt.xlabel(rf"\textrm{{{x_axis}}}", fontsize=18)
    plt.ylabel(rf"\textrm{{{y_axis}}}", fontsize=18)

    # Position legend inside the plot on the right
    if legend:
        plt.legend(
            loc="upper right",
            fontsize=12,
            framealpha=0.9,
            edgecolor="black",
            fancybox=False,
            ncol=1,
        )

    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "amplitude_distributions." + output_format),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


# Example usage:
if __name__ == "__main__":
    from emforecaster.utils.dataloading import load_forecasting
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("dataset", type=str, default="italy")
    args = parser.parse_args()

    map = {
        "title": "EMF Amplitude Distribution",
        "x-axis": "EMF Exposure (V/m)",
        "y-axis": "Density",
        "rf_emf_det": {
            "name": "Department of Electronic Engineering 2022 (Rome, Italy)",
            "clip": True,
            "thresh": 0.7,
        },
        "rf_emf_det2": {
            "name": "Department of Electronic Engineering 2023 (Rome, Italy)",
            "clip": False,
            "thresh": 0.7,
        },
        "rf_emf_ptv": {
            "name": "Unversity Hospital 2022 (Rome, Italy)",
            "clip": True,
            "thresh": 0.7,
        },
        "rf_emf_ptv2": {
            "name": "Unversity Hospital 2023 (Rome, Italy)",
            "clip": False,
            "thresh": 0.7,
        },
        "rf_emf_tur": {
            "name": "Polytechnic of Turin (Turin, Italy)",
            "clip": False,
            "thresh": 1.0,
        },
        "rf_emf_tur2": {
            "name": "Train Station (Turin, Italy)",
            "clip": False,
            "thresh": 1.0,
        },
    }

    datasets = [
        "rf_emf_det",
        "rf_emf_det2",
        "rf_emf_ptv",
        "rf_emf_ptv2",
        "rf_emf_tur",
        "rf_emf_tur2",
    ]
    data_list = []
    names = []

    if args.dataset == "italy":

        for dataset in datasets:
            info = map[dataset]
            data = load_forecasting(
                dataset_name=dataset,
                clip=info["clip"],
                thresh=info["thresh"],
                date=False,
                full_channels=True,
            )
            data_list.append(data.squeeze())
            names.append(info["name"])
        legend = True
        dataset_name = "Italy"
    elif args.dataset == "turkey":
        # Turkey data
        turkey_data = load_forecasting(
            dataset_name="rf_emf", clip=False, date=False, full_channels=True
        )

        locations = [0, 3, 6, 9, 12, 15]

        for i in range(len(locations)):
            data_list.append(turkey_data[locations[i], :].squeeze())
            names.append(f"L{locations[i]+1} (Ordu City, Turkey)")
        legend = True
        dataset_name = "Turkey"

    plot_amplitude_distributions(
        data_list=data_list,
        names=names,
        title=map["title"],
        x_axis=map["x-axis"],
        y_axis=map["y-axis"],
        dpi=600,
        output_format="pdf",
        legend=legend,
        dataset_name=dataset_name,
    )
