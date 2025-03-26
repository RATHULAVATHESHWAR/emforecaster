import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import signal
from statsmodels.tsa.stattools import adfuller
from typing import Optional, List, Dict, Literal
from scipy.fft import fft
import pywt
import os
from datetime import datetime


def set_theme(theme: Literal["light", "dark"] = "light"):
    """Set the plotting theme parameters."""
    base_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.titlesize": 20,
        "axes.labelsize": 18,  # Increased from 16
        "xtick.labelsize": 18,
        "ytick.labelsize": 16,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.5,
        "axes.grid": False,
    }

    if theme == "dark":
        theme_params = {
            **base_params,
            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        }
        plot_colors = {
            "face": "black",
            "text": "white",
            "line": "white",
            "grid": "white",
            "spine": "white",
        }
    else:  # light theme
        theme_params = {
            **base_params,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
        plot_colors = {
            "face": "white",
            "text": "black",
            "line": "black",
            "grid": "gray",
            "spine": "black",
        }

    plt.rcParams.update(theme_params)
    return plot_colors


def plot_correlation_heatmap(
    data_list: List[np.ndarray],
    info_list: List[Dict],
    output_dir: str,
    dpi: int = 600,
    output_format: str = "eps",
    theme: Literal["light", "dark"] = "light",
    cmap: Literal[
        "RdYlBu_r", "coolwarm", "viridis", "plasma", "inferno", "magma", "Blues"
    ] = "RdYlBu_r",
) -> None:
    """
    Create and save a correlation heatmap for multiple time series.

    Args:
        data_list (List[np.ndarray]): List of time series datasets
        info_list (List[Dict]): List of dictionaries containing information for each dataset
        output_dir (str): Directory to save the output plot
        dpi (int): DPI for saving plots. Default is 600
        output_format (str): Output format for saving plots. Default is "eps"
        theme (str): Plot theme, either 'light' or 'dark'. Default is 'light'
    """
    # Get colors from theme
    colors = set_theme(theme)

    # Calculate correlation matrix
    n_series = len(data_list)
    corr_matrix = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(n_series):
            # Ensure data is properly shaped for correlation calculation
            series1 = data_list[i].astype(float)
            series2 = data_list[j].astype(float)

            # Calculate correlation
            corr = np.corrcoef(series1, series2)[0, 1]
            corr_matrix[i, j] = corr

    # Create figure with larger size for better readability
    fig = plt.figure(figsize=(12, 10), facecolor=colors["face"])
    ax = fig.add_subplot(111)

    # Create heatmap with blue-white gradient if specified
    if cmap == "Blues":
        # Create custom colormap from white to blue
        colors_blue = plt.cm.Blues(np.linspace(0, 1, 256))
        cmap = LinearSegmentedColormap.from_list("Custom", [(1, 1, 1), colors_blue[-1]])

    im = ax.imshow(corr_matrix, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

    # Add colorbar with larger font size
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(
        r"\textrm{Correlation Coefficient}", fontsize=20, color=colors["text"]
    )

    # Set title
    # plt.title(r"\textbf{Time Series Correlation Matrix}", fontsize=20, pad=20, color=colors['text'])

    # Get location names for labels
    if mode == "turkey":
        locations = [f"L{i+1}" for i in range(n_series)]  # L1, L2, ..., L17
    else:
        locations = [info["codename"].split(" (")[0] for info in info_list]

    # Set ticks and labels
    ax.set_xticks(np.arange(n_series))
    ax.set_yticks(np.arange(n_series))
    ax.set_xticklabels(locations, fontsize=24)
    ax.set_yticklabels(locations, fontsize=24)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values in each cell
    for i in range(n_series):
        for j in range(n_series):
            # Set font size based on mode
            coef_fontsize = 24 if mode == "italy" else 16  # 1.5x bigger for Italy

            # Set text color based on colormap and correlation value
            if cmap == "viridis":
                color = "black"  # Always black for viridis
            else:
                # Choose text color based on background brightness for other colormaps
                color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"

            text = ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=coef_fontsize,
            )

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(
        os.path.join(output_dir, f"correlation_matrix.{output_format}"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor=colors["face"],
        edgecolor="none",
    )
    plt.close()


def plot_fft(
    channel_data, sampling_frequency, output_path, output_format="eps", dpi=600
):
    """
    Plot FFT with appropriate period scaling based on data length.
    Parameters:
    -----------
    channel_data : array-like
        Time series data to analyze
    sampling_frequency : float
        Sampling frequency in Hz
    output_path : str
        Path to save the output plot
    output_format : str
        Format to save the plot (default: "eps")
    dpi : int
        DPI for saving plot (default: 600)
    """
    # Calculate key time parameters
    sampling_period = 1 / (sampling_frequency * 60)  # Convert Hz to per minute
    data_length_minutes = len(channel_data) * sampling_period

    # Print diagnostic information
    print(
        f"Data length: {data_length_minutes/60:.2f} hours ({data_length_minutes/1440:.2f} days)"
    )
    print(f"Sampling period: {sampling_period:.3f} minutes")

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Compute FFT
    fft_result = fft(channel_data)
    n = len(channel_data)
    freqs = np.fft.fftfreq(n, d=1 / sampling_frequency)
    periods = 1 / freqs / 60  # Convert to periods in minutes

    # Only consider positive frequencies up to Nyquist
    positive_freq_idxs = np.where((freqs > 0) & (freqs <= sampling_frequency / 2))

    # Get magnitudes for positive frequencies
    magnitudes = np.abs(fft_result)[positive_freq_idxs]
    periods_pos = periods[positive_freq_idxs]

    # Decimate the data (take every Nth point)
    decimation_factor = 10  # Adjust this value to control point density
    periods_pos = periods_pos[::decimation_factor]
    magnitudes = magnitudes[::decimation_factor]

    # Plot the main FFT curve in black
    ax.semilogy(periods_pos, magnitudes, "k-", linewidth=2)  # 'k-' for solid black line

    # Set title and labels with LaTeX formatting
    ax.set_xlabel(r"\textrm{Period (1/Hz)}", fontsize=36)
    ax.set_ylabel(r"\textrm{Amplitude Spectrum (log scale)}", fontsize=36)

    # Set proper x-axis limits
    min_period = 2 * sampling_period  # Nyquist limit
    max_period = 50 * 60  # 50 hours (just a bit after 48h)
    ax.set_xlim(min_period, max_period)
    ax.set_xscale("log")

    # Remove default grid
    ax.grid(False)

    # Define required hour markers (in minutes)
    required_hours = [1, 2, 4, 8, 12, 24, 48]
    tick_locs = [h * 60 for h in required_hours]  # Convert hours to minutes

    # Add vertical lines at each hour marker
    for tick in tick_locs:
        if tick == 24 * 60:  # 24 hours in minutes
            ax.axvline(x=tick, color="red", linestyle="--", alpha=0.3)
        else:
            ax.axvline(x=tick, color="gray", linestyle="--", alpha=0.3)

    # Set these specific ticks without rotation
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"{h}h" for h in required_hours], fontsize=36)

    # Set font sizes for both axes
    ax.tick_params(axis="both", which="major", labelsize=36)
    ax.tick_params(axis="both", which="minor", labelsize=36)

    # Remove tick label rotation
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Save plot
    plt.savefig(
        f"{output_path}.{output_format}",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def analyze_emforecaster(
    data_list: List[np.ndarray],
    info_list: List[Dict],
    dates_list: Optional[List[np.ndarray]] = None,
    dpi: int = 600,
    output_format: str = "eps",
    theme: Literal["light", "dark"] = "light",
    cmap: Literal[
        "RdYlBu_r", "coolwarm", "viridis", "plasma", "inferno", "magma"
    ] = "RdYlBu_r",
    correlation: bool = False,
    fft: bool = False,
    turkey_dates: bool = False,
) -> None:
    """
    Analyze and visualize time series data with improved LaTeX formatting and larger font sizes.

    Args:
        data_list (List[np.ndarray]): List of time series datasets.
        info_list (List[Dict]): List of dictionaries containing information for each time series datasets.
        dates_list (List[np.ndarray]): List of date strings numpy arrays for each dataset. Default is None.
        dpi (int): DPI for saving plots. Default is 600.
        output_format (str): Output format for saving plots. Default is "eps".
        theme (str): Plot theme, either 'light' or 'dark'. Default is 'light'.
    """
    # Set theme and get colors
    colors = set_theme(theme)

    # Create output directory
    output_dir = os.path.abspath(
        os.path.join(os.getcwd(), "..", "..", "..", "figures", "general")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate correlation heatmap if requested
    if correlation and len(data_list) > 1:
        plot_correlation_heatmap(
            data_list=data_list,
            info_list=info_list,
            output_dir=output_dir,
            dpi=dpi,
            output_format=output_format,
            theme=theme,
            cmap=cmap,
        )

    for i in range(len(data_list)):
        # Calculate global min and max for amplitude axis
        global_min = np.min(data_list[i])
        global_max = np.max(data_list[i])

        # Add some padding to the limits
        y_range = global_max - global_min
        global_min -= y_range * 0.05
        global_max += y_range * 0.05

        # Individual channel analysis
        data = data_list[i].squeeze()

        # Get title and y-axis label from info_list
        title = info_list[i]["title"]
        y_axis = info_list[i]["y-axis"]

        if turkey_dates:
            # Create figure
            fig = plt.figure(figsize=(24, 5), facecolor=colors["face"])
            ax = fig.add_subplot(111)

            # Plot full data range
            ax.plot(data, color=colors["line"], linewidth=1.2)

            # Force x-axis to show hours
            ax.set_xlabel("Time (hours)", fontsize=26, color="black")

            # Calculate positions for 4-hour intervals
            samples_per_4hours = int(
                (4 * 3600) / 15
            )  # 4 hours in samples (at 15s sampling)
            tick_positions = list(range(0, len(data), samples_per_4hours))
            tick_labels = [f"{4*i:02d}:00" for i in range(len(tick_positions))]

            # Set the ticks explicitly
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0, fontsize=26)

            # Title and labels
            ax.set_title(
                rf"\textbf{{{title}}}", fontsize=32, pad=20, color=colors["text"]
            )
            ax.set_ylabel(r"\textrm{" + y_axis + r"}", fontsize=26, color="black")
            ax.set_ylim(global_min, global_max)

            # Set tick size for y-axis
            plt.yticks(fontsize=26)

        elif dates_list is not None:
            dates = dates_list[i]

            # Create figure
            fig = plt.figure(figsize=(24, 5), facecolor=colors["face"])
            ax = fig.add_subplot(111)

            # Original plotting code
            x_indices = list(range(len(dates)))
            ax.plot(x_indices, data, color=colors["line"], linewidth=1.2)

            # Title and labels
            ax.set_title(
                rf"\textbf{{{title}}}", fontsize=32, pad=20, color=colors["text"]
            )
            ax.set_ylabel(r"\textrm{" + y_axis + r"}", fontsize=26, color="black")
            ax.set_ylim(global_min, global_max)

            # Set tick size for y-axis
            plt.yticks(fontsize=26)

            # Set up ticks
            num_ticks = 5
            step = max(len(x_indices) // num_ticks, 1)
            selected_indices = x_indices[::step]
            selected_dates = [dates[i] for i in selected_indices]
            ax.set_xticks(selected_indices)
            ax.set_xticklabels(selected_dates, fontsize=26, color="black")

        # Common styling for both cases
        # Center align the date labels
        for tick in ax.get_xticklabels():
            tick.set_ha("center")

        # Style the spines
        for spine in ax.spines.values():
            spine.set_color(colors["spine"])
            spine.set_linewidth(1.5)

        # Add subtle grid
        ax.grid(True, axis="y", linestyle="--", alpha=0.2, color=colors["grid"])

        # Adjust layout
        plt.subplots_adjust(bottom=0.2)

        # Save with tight layout
        file_name = info_list[i]["file_codename"]
        plt.savefig(
            os.path.join(output_dir, f"{file_name}.{output_format}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor=colors["face"],
            edgecolor="none",
        )
        plt.close()

        if fft:
            plot_fft(
                data,
                info_list[i]["sampling_frequency"],
                os.path.join(output_dir, f"{info_list[i]['file_codename']}_fft"),
                output_format,
                dpi,
            )


def convert_date_format(date_string, include_time=True):
    """
    Convert date string to formatted output.
    Handles both 'YYYY/MM/DD HH:MM:SS' and 'DD/MM/YYYY HH:MM:SS' formats.
    """
    try:
        # First try DD/MM/YYYY format
        date_obj = datetime.strptime(date_string, "%d/%m/%Y %H:%M:%S")
    except ValueError:
        try:
            # Then try YYYY/MM/DD format
            date_obj = datetime.strptime(date_string, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            raise ValueError(
                f"Date string '{date_string}' must be in either 'DD/MM/YYYY HH:MM:SS' or 'YYYY/MM/DD HH:MM:SS' format"
            )

    # Format the date part
    date_part = date_obj.strftime("%B %d").replace(
        " 0", " "
    )  # Remove leading zero in day
    date_part += (
        "th"
        if 11 <= date_obj.day <= 13
        else {1: "st", 2: "nd", 3: "rd"}.get(date_obj.day % 10, "th")
    )
    date_part += f", {date_obj.year}"

    # Add time if requested
    if include_time:
        time_part = date_obj.strftime("%I:%M%p").lstrip("0").lower()
        return f"{date_part} - {time_part}"
    else:
        return date_part


# Example usage
if __name__ == "__main__":
    from emforecaster.utils.dataloading import load_forecasting
    import argparse

    parser = argparse.ArgumentParser(description="Time Series Analysis")
    parser.add_argument(
        "mode", type=str, default="italy", help="Mode of operation: 'italy' or 'turkey'"
    )
    parser.add_argument(
        "proportion",
        type=float,
        default=1,
        help="Proportion of data to analyze, .e.g, 0.1 is the first 10% of data. If >1 then takes absolute time indices.",
    )
    parser.add_argument(
        "output_format",
        type=str,
        default="eps",
        help="Output format for saving plots. Default is 'eps'.",
    )
    parser.add_argument(
        "cmap",
        type=str,
        default="RdYlBu_r",
        help="Colormap for correlation heatmap. Default is 'RdYlBu_r'.",
    )
    parser.add_argument(
        "--correlation", action="store_true", help="Plot correlation heatmap."
    )
    parser.add_argument("--fft", action="store_true", help="Plot FFT.")
    args = parser.parse_args()

    map = {
        "rf_emf_det": {
            "title": "Department of Electronic Engineering 2022 (Rome, Italy)",
            "codename": "DEE 22'",
            "file_codename": "DEE_22",
            "y-axis": "EMF Exposure (V/m)",
            "x-axis": "Date",
            "clip": True,
            "thresh": 0.7,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_det2": {
            "title": "Department of Electronic Engineering 2023 (Rome, Italy)",
            "codename": "DEE 23'",
            "file_codename": "DEE_23",
            "y-axis": "EMF Exposure (V/m)",
            "x-axis": "Date",
            "clip": False,
            "thresh": 1.0,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_ptv": {
            "title": "Unversity Hospital 2022 (Rome, Italy)",
            "codename": "UH 22'",
            "file_codename": "UH_22",
            "y-axis": "EMF Exposure (V/m)",
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
            "file_codename": "UH_23",
            "y-axis": "EMF Exposure (V/m)",
            "x-axis": "Date",
            "clip": False,
            "thresh": 0.7,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_tur": {
            "title": "Polytechnic of Turin (Turin, Italy)",
            "codename": "PT",
            "file_codename": "PT",
            "y-axis": "EMF Exposure (V/m)",
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
            "file_codename": "TS",
            "y-axis": "EMF Exposure (V/m)",
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
            "file_codename": "AD",
            "y-axis": "EMF Exposure (V/m)",
            "x-axis": "Time (6min)",
            "clip": False,
            "thresh": 1.0,
            "date": False,
            "sampling_frequency": 1 / 15,  # 15s
            "multivariate": True,
        },
    }

    mode = args.mode
    if mode == "italy" or mode == "both":
        datasets = [
            "rf_emf_det",
            "rf_emf_det2",
            "rf_emf_ptv",
            "rf_emf_ptv2",
            "rf_emf_tur",
            "rf_emf_tur2",
        ]
    elif mode == "turkey" or mode == "both":
        # channels = [2, 13] # Locations 3 and 14
        channels = list(range(17))  # Locations 1-17
    else:
        raise ValueError("Invalid mode")

    data_list = []
    dates_list = [] if mode == "italy" or mode == "both" else None
    info_list = []

    if mode == "italy" or mode == "both":
        for dataset in datasets:
            info = map[dataset]
            data = load_forecasting(
                dataset_name=dataset,
                clip=info["clip"],
                thresh=info["thresh"],
                date=info["date"],
                full_channels=True,
            )

            if info["date"]:
                if dataset in {"rf_emf_tur", "rf_emf_tur2"}:
                    dates = [date_string.replace("-", "/") for date_string in data[0]]
                else:
                    dates = data[0]

                convert_date_format_vec = np.vectorize(convert_date_format)
                dates = convert_date_format_vec(dates, include_time=False)
                data = data[1:].T
            else:
                dates = None

            if args.proportion < 1:
                data = data[: int(args.proportion * len(data))]
                if info["date"]:
                    dates = dates[: int(args.proportion * len(dates))]
            elif args.proportion > 1:
                data = data[: int(args.proportion)]
                if info["date"]:
                    dates = dates[: int(args.proportion)]

            data_list.append(data.squeeze())
            info_list.append(info)
            if info["date"]:
                dates_list.append(dates)
    elif mode == "turkey" or mode == "both":
        for channel in channels:
            # Create a new copy of the info dictionary for each channel
            channel_info = map["rf_emf"].copy()
            channel_info["title"] = f"L{channel+1} - {channel_info['title']}"

            data = load_forecasting(
                dataset_name="rf_emf",
                clip=False,
                date=False,
                full_channels=True,
            )

            data = data[channel]

            data_list.append(data)
            info_list.append(channel_info)

    analyze_emforecaster(
        data_list=data_list,
        info_list=info_list,
        dpi=600,
        dates_list=dates_list,
        output_format=args.output_format,
        correlation=args.correlation,
        fft=args.fft,
        turkey_dates=True if mode == "turkey" else False,
        cmap=args.cmap,
    )
