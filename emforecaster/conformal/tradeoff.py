import pandas as pd
import numpy as np
import os


def compute_and_save_tos(df, save_path, beta=2 / 3, lambda_param=0.5):
    """
    Compute Trade-off Score (TOS) for each model and save results to specified path.

    Parameters:
    df: pandas DataFrame containing columns:
        'parameters/exp/model_id'
        'sl_test/jc'
        'sl_test/ic'
        'sl_test/interval_width_mean'
    save_path: str, path where to save the resulting CSV
    beta: weight parameter for WAC calculation (default 0.6)
    lambda_param: weight parameter for TOS calculation (default 0.7)

    Returns:
    DataFrame with processed columns plus WAC and TOS scores, and saves it to CSV
    """
    # Create a copy with renamed columns for processing
    working_df = df.copy()
    column_mapping = {
        "parameters/exp/model_id": "model",
        "sl_test/jc": "JC",
        "sl_test/ic": "IC",
        "sl_test/interval_width_mean": "MIW",
    }
    working_df = working_df.rename(columns=column_mapping)

    # Check required columns exist
    required_columns = ["model", "JC", "IC", "MIW"]
    missing_columns = [col for col in required_columns if col not in working_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check if there are duplicate model entries
    if working_df.groupby("model").size().max() > 1:
        raise ValueError(
            "Found duplicate model entries. Each model should appear exactly once."
        )

    # Calculate Weighted Average Coverage (WAC)
    working_df["WAC"] = (beta * working_df["JC"] + (1 - beta) * working_df["IC"]) / 2

    # Compute z-scores for MIW
    working_df["MIW_zscore"] = (
        working_df["MIW"] - working_df["MIW"].mean()
    ) / working_df["MIW"].std()

    # Calculate sigmoid of z-scores
    working_df["MIW_sigmoid"] = 1 / (1 + np.exp(working_df["MIW_zscore"]))

    # Compute final TOS score
    working_df["TOS"] = (
        lambda_param * (working_df["WAC"] / 100)
        + (1 - lambda_param) * working_df["MIW_sigmoid"]
    )

    # Prepare final dataframe with model column first
    result_df = working_df[
        ["model", "IC", "JC", "MIW", "WAC", "MIW_zscore", "MIW_sigmoid", "TOS"]
    ]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    result_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    return result_df


# Example usage
if __name__ == "__main__":
    # Example data with original column names
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name. Options: 'rf_emf_det', 'rf_emf_ptv', 'rf_emf_ptv2', 'rf_emf_tur'",
    )
    parser.add_argument("pred_len", type=int, help="Prediction length of forecast")
    parser.add_argument("alpha", type=float, help="Significance level")
    args = parser.parse_args()

    # Pathing + load data
    input_path = "../analysis/neptune/results/conformal/ctsf.csv"
    save_path = f"../analysis/neptune/results/conformal/tos_{args.dataset}_pred_len{args.pred_len}_alpha{args.alpha}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = pd.read_csv(input_path)

    # Filter for pred_len, alpha, and dataset
    data = data[
        (data["parameters/data/pred_len"] == args.pred_len)
        & (data["parameters/conf/alpha"] == args.alpha)
        & (data["parameters/data/dataset"] == args.dataset)
    ]

    # print(data.head())

    # Assert there is only one row per model_id
    assert (
        data.groupby("parameters/exp/model_id").size().max() == 1
    ), "Duplicate rows, check ctsf.csv"

    df = pd.DataFrame(data)
    result = compute_and_save_tos(df, save_path)
