from emforecaster.utils.dataloading import load_forecasting
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


def evaluate_arima_sliding(
    train, val, test, window_size=336, pred_lengths=[96, 192, 336, 512], order=(1, 1, 1)
):
    """
    Evaluate ARIMA using sliding windows on test set, fitting model only once on training data.

    Args:
        train: Training sequence of shape (1, num_time_steps)
        val: Validation sequence of shape (1, num_time_steps)
        test: Test sequence of shape (1, num_time_steps)
        window_size: Size of input window (default: 336)
        pred_lengths: List of prediction lengths to evaluate
        order: ARIMA order (p,d,q)

    Returns:
        dict: Results for each prediction length
    """
    # Flatten sequences
    train_seq = train.ravel()
    test_seq = test.ravel()

    # Fit ARIMA once on full training sequence
    print("Fitting ARIMA model on training data...")
    model = ARIMA(train_seq, order=order)
    fitted_model = model.fit()

    results = {}

    # Evaluate for each prediction length
    for pred_len in pred_lengths:
        print(f"\nEvaluating {pred_len}-step forecasts...")

        all_predictions = []
        all_labels = []

        # Sliding window evaluation
        for i in range(len(test_seq) - window_size - pred_len + 1):
            # Get input window
            input_window = test_seq[i : i + window_size]

            # Get true values for this window
            true_future = test_seq[i + window_size : i + window_size + pred_len]

            try:
                # Update model with current window without refitting
                model_window = fitted_model.apply(input_window)

                # Generate forecast
                predictions = model_window.forecast(steps=pred_len)

                all_predictions.append(predictions)
                all_labels.append(true_future)
            except:
                print(f"Warning: Failed prediction at index {i}")
                continue

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate metrics
        mse = mean_squared_error(all_labels, all_predictions)
        mae = mean_absolute_error(all_labels, all_predictions)
        rmse = np.sqrt(mse)

        results[pred_len] = {
            "predictions": all_predictions,
            "labels": all_labels,
            "metrics": {"mse": mse, "mae": mae, "rmse": rmse},
        }

        print(f"Results for {pred_len}-step forecast:")
        print(f"Number of evaluation windows: {len(all_predictions)}")
        print(f"MSE:  {mse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # Optional: Print shape of predictions array
        print(f"Predictions shape: {all_predictions.shape}")
        print(f"Labels shape: {all_labels.shape}")

    return results


# Example usage
train, val, test = load_forecasting(
    dataset_name="rf_emf_ptv2",
    seq_len=336,
    pred_len=96,
    window_stride=1,
    scale=True,
    train_split=0.7,
    val_split=0.1,
    univariate=False,
    resizing_mode="None",
    single_channel=False,
    target_channel=0,
    clip=True,
    thresh=0.7,
    date=False,
    full_channels=False,
    differencing=0,
)

# Run evaluation with sliding windows
results = evaluate_arima_sliding(
    train=train,
    val=val,
    test=test,
    window_size=336,
    pred_lengths=[96, 192, 336, 512],
    order=(1, 0, 1),
)


# Dataloader
#
# train, val, test = load_forecasting(
#     dataset_name="rf_emf_det",
#     seq_len=336,
#     pred_len=96,
#     window_stride=1,
#     scale=True,
#     train_split=0.7,
#     val_split=0.1,
#     univariate=False,
#     resizing_mode="None",
#     single_channel=False,
#     target_channel=0,
#     clip=False,
#     thresh=1.0,
#     date=False,
#     full_channels=False,
#     differencing=0)
