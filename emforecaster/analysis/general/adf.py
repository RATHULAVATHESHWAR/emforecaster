import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import os

# Set up LaTeX style formatting
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


def plot_adf_results(
    data,
    channel_names,
    save_path="../../../figures/general/",
    filename="adf_results.eps",
    dpi=600,
    dataset_name="turkey",
):
    """
    Create and save ADF test results visualization with LaTeX formatting

    Parameters:
    -----------
    data : Union[numpy.ndarray, List[numpy.ndarray]]
        Either:
        - Array of shape (num_emforecaster, seq_len) containing the time series data
        - List of 1D numpy arrays, each containing a time series of possibly different length
    channel_names : list
        List of strings containing names for each time series
    save_path : str
        Directory path where the figure will be saved
    filename : str
        Name of the output file
    dpi : int
        Resolution of the output figure (default: 300)
    """
    # Validate inputs
    if isinstance(data, np.ndarray):
        if len(data.shape) != 2:
            raise ValueError(
                "If data is a numpy array, it must be 2D with shape (num_emforecaster, seq_len)"
            )
        num_series = data.shape[0]
    elif isinstance(data, list):
        if not all(isinstance(x, np.ndarray) for x in data):
            raise ValueError("If data is a list, all elements must be numpy arrays")
        if not all(len(x.shape) == 1 for x in data):
            raise ValueError("If data is a list, all arrays must be 1D")
        num_series = len(data)
    else:
        raise ValueError(
            "data must be either a 2D numpy array or a list of 1D numpy arrays"
        )

    if len(channel_names) != num_series:
        raise ValueError(
            f"Number of channel names ({len(channel_names)}) must match number of time series ({num_series})"
        )

    # Perform ADF test for each time series
    results = {
        "Series": channel_names,
        "ADF_Stat": [],
        "p_value": [],
        "Length": [],  # Add sequence length information
    }

    for i in range(num_series):
        # Extract time series based on input type
        if isinstance(data, np.ndarray):
            series = data[i]
        else:
            series = data[i]

        # Store sequence length
        results["Length"].append(len(series))

        # Perform ADF test
        try:
            adf_test = adfuller(series)
            results["ADF_Stat"].append(adf_test[0])  # ADF Statistic
            results["p_value"].append(adf_test[1])  # p-value
        except Exception as e:
            print(f"Warning: ADF test failed for series {channel_names[i]}: {str(e)}")
            results["ADF_Stat"].append(np.nan)
            results["p_value"].append(np.nan)

    # Create the DataFrame without sorting
    df = pd.DataFrame(results)

    # Add stationarity categories
    def get_stationarity_category(p_value):
        if pd.isna(p_value):
            return "Test Failed"
        if p_value < 0.01:
            return "Strongly Stationary"
        elif p_value < 0.05:
            return "Moderately Stationary"
        elif p_value < 0.1:
            return "Weakly Stationary"
        else:
            return "Non-Stationary"

    df["Stationarity"] = df["p_value"].apply(get_stationarity_category)

    # Create color mapping
    color_map = {
        "Strongly Stationary": "#2ecc71",
        "Moderately Stationary": "#3498db",
        "Weakly Stationary": "#f1c40f",
        "Non-Stationary": "#e74c3c",
    }

    # Remove sorting - keep original order
    # Create the visualization with more space for legend
    fig, ax = plt.subplots(figsize=(12, 8))  # Increased width for legend

    # Plot points (reversed order)
    for i in range(len(df)):
        if df["p_value"].iloc[i] < 0.05:
            marker = "*"
            size = 200
        else:
            marker = "o"
            size = 100

        # Use (len(df) - 1 - i) to reverse the order
        ax.scatter(
            df["ADF_Stat"].iloc[i],
            len(df) - 1 - i,  # This reverses the order
            c=color_map[df["Stationarity"].iloc[i]],
            marker=marker,
            s=size,
            zorder=5,
        )

    # Customize the plot
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_yticks(range(len(df)))

    # Reverse the y-tick labels
    yticklabels = [f"\\textit{{{series}}}" for series in reversed(df["Series"])]
    ax.set_yticklabels(yticklabels, fontsize=16)

    ax.set_xlabel("ADF Test Statistic", fontsize=16)
    # Create legend elements with larger font
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            label=cat,
            markersize=15,
        )  # Increased markersize
        for cat, color in color_map.items()
    ]
    # Add markers for significance
    legend_elements.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gray",
                label="$p < 0.05$",
                markersize=20,
                markeredgecolor="gray",
            ),  # Increased markersize
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                label="Original",
                markersize=15,
            ),  # Increased markersize
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                label="Differenced",
                markersize=15,
                markeredgecolor="black",
                linewidth=1,
            ),  # Increased markersize
        ]
    )

    # Place legend to the right of the plot with larger font
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),  # Move legend outside to the right
        fontsize=14,  # Increased font size
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="gray",
    )

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the figure
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to: {full_path}")

    # Show plot
    plt.show()

    return df  # Return the DataFrame with results


def plot_adf_comparison(
    data,
    channel_names,
    save_path="../../../figures/general/",
    filename="adf_comparison_results.eps",
    dpi=600,
    dataset_name="turkey",
):
    """
    Create and save ADF test results visualization comparing original and differenced series
    """
    # Calculate differenced data
    if isinstance(data, np.ndarray):
        diff_data = np.diff(data, axis=1)
    else:  # List of arrays
        diff_data = [np.diff(series) for series in data]

    # Combine original and differenced data for plotting
    combined_data = []
    combined_names = []
    for i in range(len(channel_names)):
        # Add original series
        if isinstance(data, np.ndarray):
            combined_data.append(data[i])
        else:
            combined_data.append(data[i])
        combined_names.append(channel_names[i])  # Simplified label

        # Add differenced series
        if isinstance(data, np.ndarray):
            combined_data.append(diff_data[i])
        else:
            combined_data.append(diff_data[i])
        combined_names.append(channel_names[i])  # Same label for differenced series

    # Perform ADF test for each time series
    results = {
        "Series": combined_names,
        "ADF_Stat": [],
        "p_value": [],
        "Length": [],
        "Is_Differenced": [],
    }

    for i, series in enumerate(combined_data):
        results["Length"].append(len(series))
        results["Is_Differenced"].append("Differenced" in combined_names[i])

        try:
            adf_test = adfuller(series)
            results["ADF_Stat"].append(adf_test[0])
            results["p_value"].append(adf_test[1])
        except Exception as e:
            print(f"Warning: ADF test failed for series {combined_names[i]}: {str(e)}")
            results["ADF_Stat"].append(np.nan)
            results["p_value"].append(np.nan)

    df = pd.DataFrame(results)

    # Add stationarity categories
    def get_stationarity_category(p_value):
        if pd.isna(p_value):
            return "Test Failed"
        if p_value < 0.01:
            return "Strongly Stationary"
        elif p_value < 0.05:
            return "Moderately Stationary"
        elif p_value < 0.1:
            return "Weakly Stationary"
        else:
            return "Non-Stationary"

    df["Stationarity"] = df["p_value"].apply(get_stationarity_category)

    # Color mapping
    color_map = {
        "Strongly Stationary": "#2ecc71",
        "Moderately Stationary": "#3498db",
        "Weakly Stationary": "#f1c40f",
        "Non-Stationary": "#e74c3c",
    }

    # Remove sorting by ADF statistic, keep original order
    df["Sort_Key"] = range(len(df))  # Preserve original order

    # Create the visualization with square figure
    fig, ax = plt.subplots(figsize=(12, 12))  # Changed to square

    # Plot points with different markers for original vs differenced
    num_unique_series = len(channel_names)
    for i in range(num_unique_series):
        # Get original and differenced data for the same series
        orig_idx = i * 2
        diff_idx = i * 2 + 1

        # Plot original point
        is_significant = df["p_value"].iloc[orig_idx] < 0.05
        marker = "*" if is_significant else "o"
        size = 300 if is_significant else 150  # Increased sizes

        ax.scatter(
            df["ADF_Stat"].iloc[orig_idx],
            num_unique_series - 1 - i,  # Reverse order
            c=color_map[df["Stationarity"].iloc[orig_idx]],
            marker=marker,
            s=size,
            zorder=5,
        )

        # Plot differenced point
        is_significant = df["p_value"].iloc[diff_idx] < 0.05
        marker = "*" if is_significant else "o"
        size = 300 if is_significant else 150  # Increased sizes

        ax.scatter(
            df["ADF_Stat"].iloc[diff_idx],
            num_unique_series - 1 - i,  # Same reversed i
            c=color_map[df["Stationarity"].iloc[diff_idx]],
            marker=marker,
            s=size,
            edgecolors="black",
            linewidth=1,
            zorder=5,
        )

    # Customize the plot
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_yticks(range(num_unique_series))

    # Format y-tick labels - maintain original order but reversed
    yticklabels = [f"\\textit{{{name}}}" for name in reversed(channel_names)]
    ax.set_yticklabels(yticklabels, fontsize=24)  # Changed from 32 to 24 (1.5x of 16)

    ax.set_xlabel("ADF Test Statistic", fontsize=24)  # Changed from 32 to 24

    # Increase tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=24)  # Changed from 32 to 24
    ax.tick_params(axis="both", which="minor", labelsize=24)  # Changed from 32 to 24

    # Create legend elements with larger font
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            label=cat,
            markersize=22,
        )
        for cat, color in color_map.items()
    ]
    # Add markers for significance and original/differenced
    legend_elements.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gray",
                label="$p < 0.05$",
                markersize=30,
                markeredgecolor="gray",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                label="Original",
                markersize=22,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gray",
                label="Differenced",
                markersize=22,
                markeredgecolor="black",
                linewidth=1,
            ),
        ]
    )

    # Place legend inside the plot in the lower left
    ax.legend(
        handles=legend_elements,
        loc="lower left",  # Changed from 'upper right'
        fontsize=21,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="gray",
        bbox_to_anchor=(0.02, 0.02),
    )  # Fine-tune position within plot

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to: {full_path}")
    plt.show()

    return df


if __name__ == "__main__":
    from argparse import ArgumentParser
    from emforecaster.utils.dataloading import load_forecasting

    parser = ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to load")
    parser.add_argument(
        "--diff", action="store_true", help="Whether to apply differencing"
    )
    args = parser.parse_args()

    map = {
        "italy": {
            "title": "Rome, Italy Locations (DEE & UH)",
            "channel_names": [
                "DEE 22'",
                # "DEE 23'",
                "UH 22'",
                "UH 23'",
                "PT",
                "TS",
            ],
            # "dataset_ids": ["rf_emf_det", "rf_emf_det2", "rf_emf_ptv", "rf_emf_ptv2", "rf_emf_tur", "rf_emf_tur2"],
            "dataset_ids": [
                "rf_emf_det",
                "rf_emf_ptv",
                "rf_emf_ptv2",
                "rf_emf_tur",
                "rf_emf_tur2",
            ],
            "clip": True,
            "thresh": 0.8,
            "date": False,
            "differencing": 1 if args.diff else 0,
        },
        "turkey": {
            "title": "Altinordu District (Ordu City, Turkey)",
            "channel_names": ["L" + str(i + 1) for i in range(17)],
            "differencing": 1 if args.diff else 0,
        },
    }

    dataset_map = {
        "rf_emf_det": {
            "title": "Department of Electronic Engineering 2022 (Rome, Italy)",
            "codename": "DEE 22'",
            "y-axis": "EMF Intensity (V/m)",
            "x-axis": "Date",
            "clip": True,
            "thresh": 0.7,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        # "rf_emf_det2":
        #         {
        #             "title": "Department of Electronic Engineering 2023 (Rome, Italy)",
        #             "codename": "DEE 23'",
        #             "y-axis": "EMF Intensity (V/m)",
        #             "x-axis": "Date",
        #             "clip": False,
        #             "thresh": 1.0,
        #             "date": True,
        #             "sampling_frequency": 1/(60*6), # 6min
        #             "multivariate": False,
        #         },
        "rf_emf_ptv": {
            "title": "Unversity Hospital 2022 (Rome, Italy)",
            "codename": "UH 22'",
            "y-axis": "EMF Intensity (V/m)",
            "x-axis": "Date",
            "clip": True,
            "thresh": 0.7,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_ptv2": {
            "title": "Unversity Hospital 2023 (Rome, Italy)",
            "codename": "UH 23'",
            "y-axis": "EMF Intensity (V/m)",
            "x-axis": "Date",
            "clip": False,
            "thresh": 0.7,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_tur": {
            "title": "Polytechnic of Turin (Turin, Italy)",
            "codename": "POT",
            "y-axis": "EMF Intensity (V/m)",
            "x-axis": "Date",
            "clip": False,
            "thresh": 1.0,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_tur2": {
            "title": "Train Station (Turin, Italy)",
            "codename": "TS",
            "y-axis": "EMF Intensity (V/m)",
            "x-axis": "Date",
            "clip": False,
            "thresh": 1.0,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf": {
            "title": "Altinordu District (Ordu City, Turkey)",
            "codename": "AD",
            "y-axis": "EMF Intensity (V/m)",
            "x-axis": "Time (6min)",
            "clip": False,
            "thresh": 1.0,
            "date": False,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": True,
        },
    }

    data = []
    if args.dataset_name == "italy" or args.dataset_name == "both":
        info = map["italy"]
        data = []
        for dataset_id in info["dataset_ids"]:
            data.append(
                load_forecasting(
                    dataset_name=dataset_id,
                    clip=dataset_map[dataset_id]["clip"],
                    thresh=dataset_map[dataset_id]["thresh"],
                    date=False,
                    full_channels=True,
                ).squeeze()
            )

        if args.dataset_name == "both":
            italy_data = data

    if args.dataset_name == "turkey" or args.dataset_name == "both":
        info = map["turkey"]
        data = load_forecasting(
            dataset_name="rf_emf",
            clip=False,
            thresh=False,
            date=False,
            full_channels=True,
            differencing=0,  # Always load original data
        )
        if args.dataset_name == "both":
            turkey_data = data
    else:
        raise ValueError("Invalid dataset name")

    if args.dataset_name == "both":
        # Combine channel names in order
        italy_info = map["italy"]
        turkey_info = map["turkey"]
        combined_names = italy_info["channel_names"] + turkey_info["channel_names"]

        # Combine data in same order
        combined_data = []
        for d in italy_data:  # italy_data is already a list of arrays
            combined_data.append(d)
        for i in range(turkey_data.shape[0]):  # turkey_data is a 2D array
            combined_data.append(turkey_data[i])

        # Plot combined data
        if args.diff:
            plot_adf_comparison(
                data=combined_data,
                channel_names=combined_names,
                filename=f"combined_adf_comparison_results.eps",
                dpi=600,
                dataset_name="both",
            )
        else:
            plot_adf_results(
                data=combined_data,
                channel_names=combined_names,
                filename=f"combined_adf_results.eps",
                dpi=600,
                dataset_name="both",
            )
