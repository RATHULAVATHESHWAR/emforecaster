import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from statsmodels.tsa.stattools import adfuller
from typing import Optional
from scipy.fft import fft
import pywt
import os
from datetime import datetime

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


def analyze_emforecaster(
    data: np.ndarray,
    sampling_frequency: float,
    multivariate: bool = True,
    corr_plot: bool = False,
    cross_corr: bool = False,
    x_axis: str = "Time",
    y_axis: str = "Amplitude",
    title: str = "Time Series",
    dpi: int = 300,
    dates: Optional[np.ndarray] = None,
) -> None:
    """
    Analyze and visualize time series data with improved LaTeX formatting and larger font sizes.

    Args:
        data (np.ndarray): Input time series data. Shape (num_channels, seq_len) for multivariate,
                           or (seq_len,) for univariate.
        multivariate (bool): Whether the input is multivariate (True) or univariate (False).
        corr_plot (bool): If True, generate correlation heatmap for multivariate data.
        cross_corr (bool): If True, generate cross-correlation plots for multivariate data.
        x_axis (str): Label for x-axis. Default is "Time".
        y_axis (str): Label for y-axis. Default is "Amplitude".
        dpi (int): DPI for saving plots. Default is 300.

    Returns:
        None
    """
    if multivariate:
        num_channels, seq_len = data.shape
    else:
        num_channels, seq_len = 1, len(data)
        data = data.reshape(1, -1)

    # Create output directory
    output_dir = os.path.abspath(
        os.path.join(os.getcwd(), "..", "..", "..", "figures", "general")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Correlation plot for multivariate data
    if corr_plot and multivariate:
        print(f"Making correlation heatmap.")
        plt.figure(figsize=(10, 8))
        correlation_matrix = np.corrcoef(data)
        sns.heatmap(correlation_matrix, annot=True, cmap="inferno", vmin=-1, vmax=1)
        plt.title(r"\textbf{Correlation Heatmap}", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=dpi)
        plt.close()

    # Cross-correlation for multivariate data
    if cross_corr and multivariate and num_channels > 1:
        print(f"Computing cross-correlation between different channels.")
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                cross_corr = signal.correlate(data[i], data[j], mode="full")
                lags = signal.correlation_lags(len(data[i]), len(data[j]), mode="full")

                plt.figure(figsize=(12, 6))
                plt.plot(lags, cross_corr)
                plt.title(
                    r"\textbf{Cross-correlation: Channel "
                    + f"{i+1}"
                    + r" vs Channel "
                    + f"{j+1}"
                    + r"}",
                    fontsize=20,
                )
                plt.xlabel(r"\textrm{Lag}", fontsize=16)
                plt.ylabel(r"\textrm{Correlation}", fontsize=16)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"cross_corr_ch{i+1}_ch{j+1}.png"), dpi=dpi
                )
                plt.close()

    # Calculate global min and max for amplitude axis
    global_min = np.min(data)
    global_max = np.max(data)

    # Individual channel analysis
    for i in range(num_channels):
        print(f"Performing individual channel analysis for channel {i+1}.")
        channel_data = data[i]

        # # 2x2 plot layout
        # fig, axs = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1]})
        # fig.suptitle(fr"\textbf{{{title}}}", fontsize=20)

        # # Time series plot (top left)
        # if dates is not None:
        #     x_indices = list(range(len(dates)))

        #     # Plot using indices for x-axis
        #     axs[0, 0].plot(x_indices, channel_data, color='black')

        #     # Set up ticks
        #     num_ticks = 5  # Adjust this number to control how many ticks you want
        #     step = max(len(x_indices) // num_ticks, 1)

        #     # Set ticks and labels
        #     axs[0, 0].set_xticks(x_indices[::step])
        #     axs[0, 0].set_xticklabels(dates[::step], rotation=45, ha='right')

        #     # Rest of your code remains the same
        #     axs[0, 0].set_title(fr"\textbf{{{title}}}", fontsize=18)
        #     axs[0, 0].set_xlabel(r"\textrm{" + x_axis + r"}", fontsize=16)
        #     # axs[0, 0].set_ylabel(r"\textrm{" + y_axis + r"}", fontsize=16)
        #     axs[0, 0].set_ylim(global_min, global_max)
        #     axs[0, 0].tick_params(axis='x', rotation=45)

        #     # Adjust layout to prevent cut-off labels
        #     plt.tight_layout()
        # else:
        #     axs[0, 0].plot(channel_data, color='black')
        #     axs[0, 0].set_title(fr"\textbf{{{title}}}", fontsize=18)
        #     axs[0, 0].set_xlabel(r"\textrm{" + x_axis + r"}", fontsize=16)
        #     axs[0, 0].set_ylabel(r"\textrm{" + y_axis + r"}", fontsize=16)
        #     axs[0, 0].set_ylim(global_min, global_max)

        # # Amplitude distribution (top right)
        # print(f"Making amplitude distribution plot for channel {i+1}.")
        # axs[0, 1].hist(channel_data, bins=30, color='black', alpha=0.7)
        # axs[0, 1].set_title(r"\textbf{Amplitude Distribution}", fontsize=18)
        # axs[0, 1].set_xlabel(r"\textrm{Amplitude}", fontsize=16)
        # axs[0, 1].set_ylabel(r"\textrm{Frequency}", fontsize=16)
        # axs[0, 1].set_xlim(global_min, global_max)

        # # FFT (bottom left)
        # print(f"Making FFT plot for channel {i+1}.")
        # plot_fft(axs, channel_data, sampling_frequency)

        # # Continuous Wavelet Transform (bottom right)
        # print(f"Making CWT plot for channel {i+1}.")
        # widths = np.arange(1, 31)
        # cwtmatr, freqs = pywt.cwt(channel_data, widths, 'morl')
        # im = axs[1, 1].imshow(np.abs(cwtmatr), extent=[0, len(channel_data), 1, 31], cmap='inferno',
        #                       aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        # axs[1, 1].set_title(r"\textbf{Continuous Wavelet Transform}", fontsize=18)
        # axs[1, 1].set_xlabel(r"\textrm{Time}", fontsize=16)
        # axs[1, 1].set_ylabel(r"\textrm{Scale}", fontsize=16)
        # cbar = plt.colorbar(im, ax=axs[1, 1])
        # cbar.set_label(r'\textrm{Magnitude}', fontsize=16)

        # # Additional analysis: Augmented Dickey-Fuller test for stationarity
        # print(f"Performing Augmented Dickey-Fuller test for channel {i+1}.")
        # result = adfuller(channel_data)
        # adf_statistic, p_value = result[0], result[1]
        # axs[0, 0].text(0.05, 0.95, r"\textbf{ADF Statistic}: " + f"{adf_statistic:.2f}" + r"\\\textbf{p-value}: " + f"{p_value:.4f}",
        #                transform=axs[0, 0].transAxes, verticalalignment='top', fontsize=12,
        #                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, f"ch_{i+1}_analysis.png"), dpi=dpi)
        # plt.close()

        print(f"Making wide channel figure for channel {i+1}.")
        if dates is not None:
            x_indices = list(range(len(dates)))
            plt.figure(figsize=(20, 5))
            plt.plot(x_indices, channel_data, color="black")
            # plt.title(r"\textbf{Channel " + f"{i+1}" + r" - (Wide)}", fontsize=20)
            plt.title(rf"\textbf{{{title}}}", fontsize=22)
            # plt.xlabel(r"\textrm{" + x_axis + r"}", fontsize=18)
            plt.ylabel(r"\textrm{" + y_axis + r"}", fontsize=18)
            plt.ylim(global_min, global_max)

            # Set up ticks
            num_ticks = 5  # Adjust this number to control how many ticks you want
            step = max(len(x_indices) // num_ticks, 1)

            # Set ticks and labels
            plt.xticks(
                x_indices[::step],
                [dates[i] for i in x_indices[::step]],
                rotation=45,
                ha="right",
                fontsize=16,
            )
            plt.yticks(fontsize=16)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"ch_{i+1}_wide.png"), dpi=dpi)
            plt.close()

        else:
            plt.figure(figsize=(20, 5))
            plt.plot(channel_data, color="black")
            plt.title(r"\textbf{Channel " + f"{i+1}" + r" - (Wide)}", fontsize=20)
            plt.xlabel(r"\textrm{" + x_axis + r"}", fontsize=16)
            plt.ylabel(r"\textrm{" + y_axis + r"}", fontsize=16)
            plt.ylim(global_min, global_max)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"ch_{i+1}_wide.png"), dpi=dpi)
            plt.close()

    print(
        "Analysis complete. Plots have been saved in the 'figures/general' directory."
    )


def plot_fft(axs, channel_data, sampling_frequency):
    """
    Plot FFT with appropriate period scaling based on data length.
    Parameters:
    -----------
    axs : matplotlib.axes.Axes
        The subplot axes array where the FFT will be plotted
    channel_data : array-like
        Time series data to analyze
    sampling_frequency : float
        Sampling frequency in Hz
    """
    # Calculate key time parameters
    sampling_period = 1 / (sampling_frequency * 60)  # Convert Hz to per minute
    data_length_minutes = len(channel_data) * sampling_period

    # Print diagnostic information
    print(
        f"Data length: {data_length_minutes/60:.2f} hours ({data_length_minutes/1440:.2f} days)"
    )
    print(f"Sampling period: {sampling_period:.3f} minutes")

    # Compute FFT
    fft_result = fft(channel_data)
    n = len(channel_data)
    freqs = np.fft.fftfreq(n, d=1 / sampling_frequency)
    periods = 1 / freqs / 60  # Convert to periods in minutes

    # Only consider positive frequencies up to Nyquist
    positive_freq_idxs = np.where((freqs > 0) & (freqs <= sampling_frequency / 2))

    # Plot the FFT
    axs[1, 0].semilogy(
        periods[positive_freq_idxs], np.abs(fft_result)[positive_freq_idxs]
    )

    # Set title and labels with LaTeX formatting
    axs[1, 0].set_title(r"\textbf{Fast Fourier Transform}", fontsize=18)
    axs[1, 0].set_xlabel(r"\textrm{Period}", fontsize=16)
    axs[1, 0].set_ylabel(r"\textrm{Magnitude (log scale)}", fontsize=16)

    # Set proper x-axis limits based on data length
    min_period = 2 * sampling_period  # Nyquist limit
    max_period = data_length_minutes  # Can't detect periods longer than data length
    axs[1, 0].set_xlim(min_period, max_period)
    axs[1, 0].set_xscale("log")

    # Add gridlines
    axs[1, 0].grid(True, which="both", ls="-", alpha=0.5)

    def generate_period_sequence(min_val, max_val):
        """Generate sequence of periods based on data length"""
        ticks = []

        if max_val <= 24 * 60:  # If data is 24 hours or less
            # Minute sequence: 1, 15, 30 minutes
            minute_sequence = [1, 15, 30]
            ticks.extend([m for m in minute_sequence if min_val <= m <= max_val])

            # Hour sequence for short duration: 1, 2, 4, 8, 16, 24
            hour_sequence = [1, 2, 4, 8, 16, 24]
            minute_markers = [h * 60 for h in hour_sequence]
            ticks.extend([m for m in minute_markers if min_val <= m <= max_val])
        else:
            # Standard hour sequence for longer duration
            hour_sequence = [1, 2, 4, 8, 16, 24]
            minute_markers = [h * 60 for h in hour_sequence]
            ticks.extend([m for m in minute_markers if min_val <= m <= max_val])

            # Day sequence for longer duration
            day_sequence = [2, 4, 8, 16, 32, 64]
            day_minutes = [d * 24 * 60 for d in day_sequence]
            ticks.extend([d for d in day_minutes if min_val <= d <= max_val])

        return sorted(ticks)

    # Generate and set ticks
    tick_locations = generate_period_sequence(min_period, max_period)
    axs[1, 0].set_xticks(tick_locations)

    # Format tick labels
    def format_period(p):
        """Format period values into human-readable labels"""
        if p < 60:
            return f"{int(p)}m"
        elif p < 24 * 60:
            return f"{int(p//60)}h"
        else:
            return f"{int(p//(24*60))}d"

    tick_labels = [format_period(p) for p in tick_locations]
    axs[1, 0].set_xticklabels(tick_labels)

    # Rotate tick labels for better readability
    plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha="right")

    # Add text box with data length information
    info_text = f"Data length: {data_length_minutes/60:.1f}h"
    axs[1, 0].text(
        0.02,
        0.98,
        info_text,
        transform=axs[1, 0].transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    return axs


def pairwise_cross_correlation(
    data_list: list[np.ndarray],
    names: list[str] = None,
    x_axis: str = "Lag",
    y_axis: str = "Correlation",
    dpi: int = 300,
) -> None:
    """
    Perform pairwise cross-correlation analysis on a list of 1D numpy arrays.

    Args:
        data_list (list[np.ndarray]): List of 1D numpy arrays to analyze.
        names (list[str], optional): List of names for each array. If None, will use indices.
        x_axis (str): Label for x-axis. Default is "Lag".
        y_axis (str): Label for y-axis. Default is "Correlation".
        dpi (int): DPI for saving plots. Default is 300.

    Returns:
        None
    """
    # Create output directory
    output_dir = os.path.abspath(
        os.path.join(os.getcwd(), "..", "..", "..", "figures", "cross_correlation")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Use indices if names are not provided
    if names is None:
        names = [f"Series {i+1}" for i in range(len(data_list))]

    # Perform pairwise cross-correlation
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            # Compute cross-correlation
            cross_corr = signal.correlate(data_list[i], data_list[j], mode="full")
            lags = signal.correlation_lags(
                len(data_list[i]), len(data_list[j]), mode="full"
            )

            # Normalize cross-correlation
            norm_factor = np.sqrt(np.sum(data_list[i] ** 2) * np.sum(data_list[j] ** 2))
            cross_corr_normalized = cross_corr / norm_factor

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(lags, cross_corr_normalized)
            plt.title(
                r"\textbf{Cross-correlation: "
                + f"{names[i]}"
                + r" vs "
                + f"{names[j]}"
                + r"}",
                fontsize=20,
            )
            plt.xlabel(r"\textrm{" + x_axis + r"}", fontsize=16)
            plt.ylabel(r"\textrm{" + y_axis + r"}", fontsize=16)

            # Add vertical line at lag 0
            plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)

            # Add text with max correlation and its lag
            max_corr_idx = np.argmax(np.abs(cross_corr_normalized))
            max_corr = cross_corr_normalized[max_corr_idx]
            max_lag = lags[max_corr_idx]
            plt.text(
                0.05,
                0.95,
                f"Max correlation: {max_corr:.2f}\nAt lag: {max_lag}",
                transform=plt.gca().transAxes,
                verticalalignment="top",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"cross_corr_{names[i]}_{names[j]}.png"),
                dpi=dpi,
            )
            plt.close()

    print(
        "Pairwise cross-correlation analysis complete. Plots have been saved in the 'figures/cross_correlation' directory."
    )


# def convert_date_format(date_string, include_time=True):
#     # Parse the input date string
#     date_obj = datetime.strptime(date_string, '%d/%m/%Y %H:%M:%S')

#     # Format the date part
#     date_part = date_obj.strftime("%B %d").replace(' 0', ' ')  # Remove leading zero in day
#     date_part += "th" if 11 <= date_obj.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(date_obj.day % 10, "th")
#     date_part += f", {date_obj.year}"

#     # Add time if requested
#     if include_time:
#         time_part = date_obj.strftime("%I:%M%p").lstrip('0').lower()
#         return f"{date_part} - {time_part}"
#     else:
#         return date_part


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
        "dataset_name", type=str, default="rf_emf_det", help="Name of the dataset"
    )
    parser.add_argument(
        "proportion",
        type=float,
        default=1,
        help="Proportion of data to analyze, .e.g, 0.1 is the first 10% of data. If >1 then takes absolute time indices.",
    )
    args = parser.parse_args()

    map = {
        "rf_emf_det": {
            "title": "Department of Electronic Engineering 2022 (Rome, Italy)",
            "y-axis": "Electric Field Intensity (V/m)",
            "x-axis": "Date",
            "clip": True,
            "thresh": 0.7,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_det2": {
            "title": "Department of Electronic Engineering 2023 (Rome, Italy)",
            "y-axis": "Electric Field Intensity (V/m)",
            "x-axis": "Date",
            "clip": False,
            "thresh": 1.0,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_ptv": {
            "title": "Unversity Hospital 2022 (Rome, Italy)",
            "y-axis": "Electric Field Intensity (V/m)",
            "x-axis": "Date",
            "clip": True,
            "thresh": 0.7,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_ptv2": {
            "title": "Unversity Hospital 2023 (Rome, Italy)",
            "y-axis": "Electric Field Intensity (V/m)",
            "x-axis": "Date",
            "clip": False,
            "thresh": 0.7,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf_tur": {
            "title": "Polytechnic of Turin (Turin, Italy)",
            "y-axis": "Electric Field Intensity (V/m)",
            "x-axis": "Date",
            "clip": False,
            "thresh": 1.0,
            "date": True,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": False,
        },
        "rf_emf": {
            "title": "Altinordu District (Ordu City, Turkey)",
            "y-axis": "Electric Field Intensity (V/m)",
            "x-axis": "Time (6min)",
            "clip": False,
            "thresh": 1.0,
            "date": False,
            "sampling_frequency": 1 / (60 * 6),  # 6min
            "multivariate": True,
        },
    }

    info = map[args.dataset_name]

    data = load_forecasting(
        dataset_name=args.dataset_name,
        clip=info["clip"],
        thresh=info["thresh"],
        date=info["date"],
        full_channels=True,
    )

    if info["date"]:

        if args.dataset_name == "rf_emf_tur":
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

    print(f"Data shape: {data.shape}")

    analyze_emforecaster(
        data=data,
        sampling_frequency=info["sampling_frequency"],
        multivariate=info["multivariate"],
        corr_plot=False,
        cross_corr=False,
        x_axis=info["x-axis"],
        y_axis=info["y-axis"],
        title=info["title"],
        dpi=600,
        dates=dates,
    )
