import pandas as pd
import os
import sys


def transfer_rows(dataset, source_file, n_rows, target_file="ctsf.csv"):
    """
    Transfer the first n rows from source CSV to target CSV in results/{dataset}/ directory.
    Only transfers columns that exist in the target CSV.

    Parameters:
    dataset (str): Name of the dataset (determines directory)
    source_file (str): Name of source CSV file
    n_rows (int): Number of rows to transfer
    target_file (str): Name of target CSV file (default: 'ctsf.csv')
    """
    # Construct full paths
    directory = os.path.join("results", dataset)
    source_path = os.path.join(directory, source_file)
    target_path = os.path.join(directory, target_file)

    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Read first n rows from source file
        df_source = pd.read_csv(source_path, nrows=n_rows)

        if os.path.exists(target_path):
            # If target exists, read it
            df_target = pd.read_csv(target_path)

            # Filter source dataframe to only include columns from target
            common_columns = [
                col for col in df_target.columns if col in df_source.columns
            ]
            df_source_filtered = df_source[common_columns]

            # Concatenate the dataframes
            df_combined = pd.concat([df_target, df_source_filtered], ignore_index=True)

            # Save combined dataframe
            df_combined.to_csv(target_path, index=False)
            print(f"Added {n_rows} rows to existing file {target_path}")

            # Print information about skipped columns
            skipped_cols = set(df_source.columns) - set(df_target.columns)
            if skipped_cols:
                print(f"Skipped columns not in target: {skipped_cols}")
        else:
            # If target doesn't exist, create it with the new rows
            df_source.to_csv(target_path, index=False)
            print(f"Created new file {target_path} with {n_rows} rows")

    except FileNotFoundError:
        print(f"Error: Source file '{source_path}' not found")
    except pd.errors.EmptyDataError:
        print(f"Error: Source file '{source_path}' is empty")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <dataset> <source_file> <n_rows> [target_file]")
        sys.exit(1)

    dataset = sys.argv[1]
    source_file = sys.argv[2]
    n_rows = int(sys.argv[3])
    target_file = sys.argv[4] if len(sys.argv) > 4 else "ctsf.csv"

    transfer_rows(dataset, source_file, n_rows, target_file)
