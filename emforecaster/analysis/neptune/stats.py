import neptune
import os
import pandas as pd
import csv
from rich.console import Console


def process_exp_id(runs_table_df, exp_id, metrics, parameters):
    filtered_runs = runs_table_df[runs_table_df["parameters/exp/id"] == exp_id]

    if len(filtered_runs) != 5:
        print(f"Skipping exp_id {exp_id}: {len(filtered_runs)} runs found (expected 5)")
        return None

    results = {"exp_id": exp_id}

    # Calculate metrics
    for metric in metrics:
        mean = filtered_runs[metric].mean()
        std = filtered_runs[metric].std()
        results[f"{metric}_mean"] = mean
        results[f"{metric}_std"] = std

    # Store parameters
    for param in parameters:
        results[param] = filtered_runs[param].iloc[
            0
        ]  # Assuming all runs have the same parameter value

    return results


def main():
    console = Console()

    exp_ids = ["6VdYMGhSyA", "mbaGp9dfrU", "MqHKYk3IeW"]  # Add your exp_ids here
    metrics = ["sl_test/loss", "sl_test/mae"]
    project = "rf-emf-forecasting/RF-EMF"
    parameters = ["parameters/exp/model_id"]

    api_token = os.getenv("NEPTUNE_API_TOKEN")
    project = neptune.init_project(project=project, api_token=api_token)
    # Fetch the runs table only once
    console.log("Fetching runs table...")
    runs_table_df = project.fetch_runs_table().to_pandas()

    filters = {}

    # Filter the runs_table_df
    filtered_df = runs_table_df[
        (runs_table_df["parameters/exp/model_id"] == "RF_EMF_CNN")
        & (runs_table_df["parameters/data/dataset"] == "rf_emf")
        & (runs_table_df["parameters/data/single_channel"] == True)
    ]

    # Print the number of rows in the filtered DataFrame
    console.log(f"Number of rows after filtering: {len(filtered_df)}")

    # Compute the mean for "sl_test/loss" and "sl_test/mae"
    mean_loss = filtered_df["sl_test/loss"].mean()
    mean_mae = filtered_df["sl_test/mae"].mean()

    console.log(f"Mean sl_test/loss: {mean_loss:.4f}")
    console.log(f"Mean sl_test/mae: {mean_mae:.4f}")

    # results = []
    # for exp_id in exp_ids:
    #     result = process_exp_id(runs_table_df, exp_id, metrics, parameters)
    #     if result:
    #         results.append(result)

    # # Save results to CSV
    # if results:
    #     csv_file = "experiment_results.csv"
    #     with open(csv_file, 'w', newline='') as file:
    #         writer = csv.DictWriter(file, fieldnames=results[0].keys())
    #         writer.writeheader()
    #         for row in results:
    #             writer.writerow(row)
    #     console.log(f"Results saved to {csv_file}")
    # else:
    #     console.log("No valid results to save")


if __name__ == "__main__":
    main()
