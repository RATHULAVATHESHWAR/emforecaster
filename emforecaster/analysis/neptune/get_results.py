from genericpath import exists
import os
import yaml
import neptune
import numpy as np
import pandas as pd
from itertools import product
from rich.console import Console
import argparse


def load_yaml(path="args.yaml"):
    args = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return args


def load_project(args):
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    project = neptune.init_project(
        project=args["neptune"]["project"],
        api_token=api_token,
        # mode="read-only"  # Use read-only mode when just fetching data
    )
    return project


def get_query(filter):
    type_map = {
        bool: "bool",
        str: "string",
        int: "int",
        float: "float",
        list: "list",
        dict: "dict",
    }

    query_parts = []

    for key, value in filter.items():
        for type_class, type_name in type_map.items():
            if isinstance(value, type_class):
                if isinstance(value, bool):
                    value_str = str(value).lower()
                elif isinstance(value, str):
                    value_str = f'"{value}"'
                else:
                    value_str = str(value)

                query_part = f"({key}:{type_name} = {value_str})"
                query_parts.append(query_part)
                break
        else:
            raise ValueError(f"Unsupported type for key '{key}': {type(value)}")

    return " AND ".join(query_parts) + " AND sys/failed:bool = False"


def get_runs_table(project, query, columns, metrics, console):
    system_info = ["parameters/exp/id"]
    total_columns = columns + system_info + metrics

    try:
        # Fetch runs table with multiple conditions in the query
        df = project.fetch_runs_table(
            query=query, columns=total_columns, sort_by="sys/creation_time"
        ).to_pandas()

        if df.empty:
            console.log("No runs found matching the query.")
    except Exception as e:
        console.log(f"Error fetching runs table: {str(e)}")
        return None

    return df


def get_unique_combinations(df):
    # Get unique values for each column
    unique_values = {col: df[col].unique() for col in df.columns}

    # Create a list of (column, value) tuples for each column
    column_value_pairs = [
        [(col, val) for val in unique_values[col]] for col in df.columns
    ]

    # Use itertools.product to generate all combinations
    unique_combinations = product(*column_value_pairs)

    return unique_combinations


def filter_duplicate_runs(df, seeds, console, mode="recent"):
    """
    Args:
        df: Contains: params + metrics (from get_runs_table), list of columns for a single combination.

    Returns:
        df: Filtered DataFrame with no duplicate runs for the particular combo.
    """

    assert mode == "recent", NotImplementedError(
        "Only 'recent' mode is supported for now."
    )

    # Calculate total expected rows
    total_expected_rows = len(seeds)

    # TODO: Check whether there is at least one row with value seed, for each seed in seeds
    # Check if total rows do not match

    if df.empty:
        console.log("No rows found for the given combination.")
        return df

    if total_expected_rows != len(df):
        console.log(
            f"Warning: Expected {total_expected_rows} rows, but found {len(df)} rows. Checking for duplicates..."
        )

        # Step 1: Find duplicate rows
        duplicates = df[df.duplicated(keep=False)]

        # Step 2: Get the unique values from the 'parameters/exp/id' column for these duplicate rows
        unique_exp_ids = set(duplicates["parameters/exp/id"])

        # Step 3: Convert sys/creation_time to datetime if it's not already df['sys/creation_time'] = pd.to_datetime(df['sys/creation_time'])
        # Step 4: Find the row with the maximum sys/creation_time for each exp/id
        df_max_per_exp_id = df.loc[
            df.groupby("parameters/exp/id")["sys/creation_time"].idxmax()
        ]

        # Step 5: Find the exp/id with the overall maximum sys/creation_time
        max_exp_id = df_max_per_exp_id.loc[
            df_max_per_exp_id["sys/creation_time"].idxmax(), "parameters/exp/id"
        ]

        # Step 6: Filter the original dataframe to keep only rows with the max_exp_id
        df_cleaned = df[df["parameters/exp/id"] == max_exp_id]

        # console.log results
        console.log(f"exp/id with most recent sys/creation_time: {max_exp_id}")

        return df_cleaned
    else:
        return df


def compute_averages(df, metrics):
    averages = df[metrics].mean()
    return pd.DataFrame([averages], columns=metrics)


def df_to_csv(df, filename):
    # Convert DataFrame to a standard CSV format
    df.to_csv(filename, index=False)
    print(f"DataFrame saved to {filename} in standard CSV format")


def single_exp_analysis(df, args, combos, console, save_path="results.csv"):

    results_df = pd.DataFrame(columns=args["params"] + args["metrics"])
    for combo in combos:
        filtered_df = df.copy()
        for col, val in combo:
            filtered_df = filtered_df[filtered_df[col] == val]

        # Remove duplicate runs
        filtered_df = filter_duplicate_runs(filtered_df, args["seeds"], console)

        # Compute averages
        averages = compute_averages(filtered_df, args["metrics"])

        # Create a df with combo and averaged metrics
        combo_vals = [val for col, val in combo]
        combo_df = pd.DataFrame([combo_vals], columns=args["params"])
        combo_results = pd.concat([combo_df, averages], axis=1)
        results_df = pd.concat([results_df, combo_results])

    console.log(f"Saving results_df to {save_path}...")

    if args["ranking"] == "min":
        results_df = results_df.sort_values(by=args["deciding_metric"], ascending=True)
    elif args["ranking"] == "max":
        results_df = results_df.sort_values(by=args["deciding_metric"], ascending=False)
    else:
        raise ValueError(f"Ranking mode not supported. Use 'max' or 'min'.")

    df_to_csv(results_df, save_path)


def analysis(args, filter, project, console, save_path="results.csv"):
    # Query
    query = get_query(filter)

    # Fetch runs table based on query
    df = get_runs_table(project, query, args["params"], args["metrics"], console)

    # Get unique combinations
    combos = get_unique_combinations(df[args["params"]])
    single_exp_analysis(df, args, combos, console, save_path=save_path)


def main(dataset, model_id):
    # Rich console
    console = Console()

    # Arguments
    yaml_path = os.path.join("ablations", dataset, model_id + ".yaml")
    args = load_yaml(yaml_path)

    # Load project
    project = load_project(args)

    # Check
    if "multi_filter" not in args.keys():
        console.log("Running single-filter version.")
        save_dir = os.path.join("results", dataset)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, model_id + ".csv")
        analysis(args, args["single_filter"], project, console, save_path=save_path)

    # Multi-config version
    else:
        console.log("Running multi-filter version.")
        # Create an iterable of all combinations
        filter_keys = list(args["multi_filter"].keys())
        filter_iterator = product(*list(args["multi_filter"].values()))

        save_dir = os.path.join("results", dataset)
        os.makedirs(save_dir, exist_ok=True)

        for i, filter_vals in enumerate(filter_iterator):
            # Query
            filter = dict(zip(filter_keys, filter_vals))
            save_path = os.path.join(save_dir, model_id, f"{i}.csv")
            analysis(args, filter, project, console, save_path=save_path)

    project.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on Neptune experiments.")
    parser.add_argument(
        "dataset", type=str, default="rf_emf", help="Path to the arguments file."
    )
    parser.add_argument(
        "model_id",
        type=str,
        default="emforecaster",
        help="Path to the arguments file.",
    )
    args = parser.parse_args()
    main(args.dataset, args.model_id)
