import warnings

warnings.filterwarnings("ignore")

import os
import yaml
import time
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Set, Tuple, List, Optional, Dict, Union

import random
import string


def generate_random_id(length=10):
    # Seed the random number generator with current time and os-specific random data
    random.seed(int(time.time() * 1000) ^ os.getpid())

    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


class Experiment(BaseModel):
    model_id: str = Field(
        default="PatchTST",
        description="Model ID.",
    )
    backbone_id: str = Field(
        default="LSTM",
        description="Backbone for the RecurrentModel class. Options: 'LSTM', 'RNN', 'GRU', 'Mamba'.",
    )
    patch_model_id: str = Field(
        default="DLinear",
        description="Model used for patch embedding. Options: 'DLinear'.",
    )
    seed_list: List[int] = Field(
        default=[2024],
        description="List of random seeds to run a single experiment on.",
    )
    seed: int = Field(default=2024, description="Random seed")
    learning_type: str = Field(
        default="sl", description="Type of learning: 'sl', 'ssl'"
    )
    id: str = Field(
        default_factory=generate_random_id,
        description="Experiment ID, randomly generated 10-character string",
    )
    neptune: bool = Field(
        default=False,
        description="Whether to use Neptune for logging. If False, offline logging (JSON) will be used.",
    )
    api_token: str = Field(
        default=os.environ.get("NEPTUNE_API_TOKEN", ""), description="Neptune API token"
    )
    project_name: str = Field(
        default="xmootoo/soz-localization", description="Neptune project name"
    )
    time: str = Field(
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        description="Neptune run ID",
    )
    early_stopping: bool = Field(
        default=True, description="Whether to use early stopping"
    )
    acc: bool = Field(
        default=False, description="Evaluate accuracy or not for classification tasks"
    )
    ch_acc: bool = Field(
        default=False,
        description="Evaluate whole channel accuracy or not for classification tasks",
    )
    thresh: float = Field(
        default=0.5, description="The threshold for binary classification"
    )
    mae: bool = Field(
        default=False, description="Evaluate mean absolute error or not for forecasting"
    )
    other_metrics: bool = Field(
        default=False,
        description="Whether to use other metrics for classification evaluation (e.g., F1-score",
    )
    batchwise_metrics: bool = Field(
        default=False,
        description="Used for Stochastic Sparse Sampling (SSS). Computes the metrics scores using only windows in a given batch -> channel predictions, rather than aggregating all windows across all batches -> channel predictions.",
    )
    best_model_metric: str = Field(
        default="loss",
        description="Metric to use for model saving and early stopping. Options: 'loss', 'acc', 'ch_acc'",
    )
    tuning_metric: str = Field(
        default="loss",
        description="Metric to use for hyperparameter tuning. Options: 'ch_acc', 'ch_f1', 'ch_auroc', etc. See get_logger_mapping() under utils/classification.py for a full list of keys.",
    )
    mps: bool = Field(
        default=False,
        description="Whether to use MPS for Apple silicon hardware acceleration",
    )
    rank_verbose: bool = Field(
        default=False,
        description="Verbose logging in the console for each rank, e.g., prints the loss for each rank before reduction.",
    )
    sklearn: bool = Field(
        default=False,
        description="Whether to use scikit-learn for model training and evaluation",
    )
    sklearn_n_jobs: int = Field(
        default=-1, description="Number of CPU cores to use (-1 for all cores)"
    )
    sklearn_verbose: int = Field(default=1, description="Verbosity level")
    grid_search: bool = Field(
        default=False,
        description="Whether to use grid search for hyperparameter tuning",
    )
    task: str = Field(
        default="classification",
        description="Task type. Options: 'forecasting', 'classification'",
    )
    gpu_id: int = Field(
        default=0, description="GPU ID to use for for single device training"
    )
    u_weight: bool = Field(
        default=False,
        description="Whether to use uncertainty/confidence based weighting in the ChannelLoss for SSS",
    )
    ablation_id: int = Field(
        default=1, description="Ablation ID for the base experiment"
    )
    calibrate: bool = Field(
        default=False,
        description="Whether to calibrate the model for its window or channel probabilities",
    )
    calibration_model: str = Field(
        default="isotonic_regression",
        description="Calibration model. Options: 'isotonic_regression', 'platt_scaling', 'ensemble', 'beta_calibration'.",
    )
    calibration_type: str = Field(
        default="none",
        description="Calibration type. Options: 'none', 'window', 'channel'.",
    )
    va_inductive: bool = Field(
        default=True,
        description=" True to run the Inductive (IVAP) or False for Cross (CVAP) Venn-ABERS calibtration",
    )
    va_splits: int = Field(
        default=5,
        description="Number of splits for Cross (CVAP) Venn-ABERS calibration.",
    )


class Data(BaseModel):
    dataset: str = Field(
        default="electricity",
        description="Name of the dataset. Options: 'electricity', 'traffic', 'weather', 'exchange_rate', 'illness', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'open_neuro', 'LongTerm17'.",
    )
    dtype: str = Field(
        default="float32", description="Type of data. Options: 'float32', 'float64'"
    )
    seq_len: int = Field(default=512, description="Sequence length of the input.")
    window_stride: int = Field(
        default=1, description="Window stride for generating windows."
    )
    pred_len: int = Field(
        default=96, description="Prediction length of the forecast window."
    )
    num_classes: int = Field(
        default=2, description="Number of classes for classification tasks."
    )
    num_channels: int = Field(default=321, description="Number of time series channels")
    drop_last: bool = Field(
        default=False, description="Whether to drop the last batch."
    )
    scale: bool = Field(default=True, description="Normalize data along each channel.")
    balance: bool = Field(
        default=True, description="Balance classes within dataset for classification."
    )
    train_split: float = Field(
        default=0.6, description="Portion of data to use for training"
    )
    val_split: float = Field(
        default=0.2, description="Portion of data used for validation"
    )
    num_workers: int = Field(
        default=4, description="Number of workers for the dataloader"
    )
    pin_memory: bool = Field(
        default=True, description="Whether to pin memory for the dataloader"
    )
    prefetch_factor: int = Field(
        default=2, description="Prefetch factor for the dataloader"
    )
    shuffle_test: bool = Field(
        default=False, description="Whether to shuffle the test set"
    )
    patching: bool = Field(
        default=False, description="Whether to use patching for the dataset (LSTM only)"
    )
    patch_dim: int = Field(default=16, description="Patch dimension or patch length.")
    patch_stride: int = Field(
        default=8, description="Patch stride for generating patches.}"
    )
    univariate: bool = Field(
        default=False,
        description="Whether to process a multivariate time series as univariate data (mixed together but separated by channel)",
    )
    seq_load: bool = Field(
        default=False,
        description="Whether to use sequential dataloading. Loads train datasets first, and then test set at test time.",
    )
    rank_seq_load: bool = Field(
        default=False,
        description="Whether to use sequential dataloading for each rank for large datasets, only for DDP.",
    )
    pad_to_max: bool = Field(
        default=False,
        description="Whether to pad sequences to the maximum length in the dataset.",
    )
    dataset_only: bool = Field(
        default=False,
        description="Whether to only load the dataset without training the model (for sickit-learn).",
    )
    tslearn: bool = Field(
        default=False,
        description="Whether to use tslearn time series data for processing multiple time series.",
    )
    numpy_data: bool = Field(
        default=False,
        description="Whether to use numpy data in dataloading (will not be converted to torch tensors).",
    )
    rocket_transform: bool = Field(
        default=False,
        description="Whether to use the ROCKET transform for time series data.",
    )
    full_channels: bool = Field(
        default=False,
        description="Whether to use the full channels for the OpenNeuro dataset or not.",
    )
    resizing_mode: str = Field(
        default="None",
        description="Mode for resizing the time series data. Options: 'pad_trunc' or 'resizing' for linear interpolation.",
    )
    median_seq_len: bool = Field(
        default=False,
        description="Whether to use the median sequence length from the training set as the context/window size.",
    )
    single_channel: bool = Field(
        default=False,
        description="Whether to use a single model for each channel. Use target_channel to specify channel ID.",
    )
    target_channel: int = Field(
        default=-1, description="Target channel for univariate modelling."
    )
    time_indices: bool = Field(
        default=False,
        description="Whether to use relative time indices for sorting windows within the channel.",
    )
    clip: bool = Field(
        default=False,
        description="Whether to clip the time series data for noisy peaks. It will apply linear interpolation between nearest neighbour points.",
    )
    clip_thresh: float = Field(
        default=0.8, description="Threshold for clipping the time series data."
    )
    datetime: bool = Field(
        default=False,
        description="Whether to use datetime data for the time series analysis.",
    )
    datetime_features: List[str] = Field(
        default=["year", "month", "day", "hour", "minute", "second"],
        description="Datetime features to use for the time series analysis. Choose a subset of the default argument.",
    )
    cyclical_encoding: bool = Field(
        default=False,
        description="Whether to use cyclical encoding for datetime features.",
    )
    average_italy: bool = Field(
        default=False,
        description="Whether to average the time series data for the Italy dataset.",
    )
    difference_input: bool = Field(
        default=False,
        description="Whether to use first-order differencing of the input time series for the model.",
    )


class Conformal(BaseModel):
    conf: bool = Field(
        default=False,
        description="Whether to use conformal prediction. This will split .",
    )
    alpha: float = Field(
        default=0.1,
        description="Significance level (i.e. error rate) for the conformal prediction.",
    )
    corrected: bool = Field(
        default=False,
        description="Whether to use Bonferonni correction for conformal time series forecasting. This will use the corrected_critical_scores.",
    )
    intervals: bool = Field(
        default=True,
        description="Whether to report interval width mean and standard deviation for evaluation metrics.",
    )
    validation_eval: bool = Field(
        default=False,
        description="Whether evaluate the model for coverage on the validation set at each epoch. If False, it only evaluates on the test set.",
    )


class SL(BaseModel):
    optimizer: str = Field(
        default="adam",
        description="Optimizer for supervised learning: 'adam' or 'adamw'",
    )
    criterion: str = Field(
        default="MSE",
        description="Criterion for supervised learning: 'MSE', 'SmoothL1', 'CrossEntropy', 'BCE', 'ChannelLossBCE', 'ChannelLossCE'",
    )
    num_enc_layers: int = Field(
        default=3, description="Number of encoder layers in the model"
    )
    d_model: int = Field(default=128, description="Dimension of the model")
    d_ff: int = Field(
        default=256, description="Dimension of the feedforward network model"
    )
    num_heads: int = Field(
        default=16, description="Number of heads in each MultiheadAttention block"
    )
    dropout: float = Field(
        default=0.05, description="Dropout for some of the linears layers in PatchTSTOG"
    )
    attn_dropout: float = Field(default=0.2, description="Dropout value for attention")
    ff_dropout: float = Field(default=0.2, description="Dropout value for feed forward")
    pred_dropout: float = Field(
        default=0.1, description="Dropout value for prediction"
    )  # CyclicalEMForecaster (usable)
    batch_first: bool = Field(
        default=True, description="Whether the first dimension is batch"
    )
    norm_mode: str = Field(
        default="batch1d",
        description="Normalization mode: 'batch1d', 'batch2d', or 'layer'",
    )
    batch_size: int = Field(default=64, description="Batch size")
    revin: bool = Field(
        default=True, description="Whether to use instance normalization with RevIN."
    )
    revout: bool = Field(
        default=True, description="Whether to use add mean and std back after forecast."
    )
    revin_affine: bool = Field(
        default=True,
        description="Whether to use learnable affine parameters for RevIN.",
    )
    eps_revin: float = Field(
        default=1e-5, description="Epsilon value for reversible input"
    )
    lr: float = Field(default=1e-4, description="Learning rate")
    epochs: int = Field(default=100, description="Number of epochs to train")
    scheduler: str = Field(
        default=None, description="Scheduler to use for learning rate annealing"
    )
    weight_decay: float = Field(
        default=1e-6, description="Weight decay for the optimizer"
    )
    dataset_class: str = Field(
        default="forecasting",
        description="Task type: 'forecasting''",
    )
    early_stopping: bool = Field(
        default=False, description="Early stopping for supervised learning."
    )
    head_type: str = Field(
        default="linear",
        description="Head type for supervised learning: 'linear' or 'mlp'",
    )  # CyclicalEMForecaster (usable)
    num_kernels: int = Field(default=32, description="Number of convolutional kernels.")
    return_head: bool = Field(
        default=False, description="Whether to return the head of the model."
    )

    # CyclicalEMForecaster
    datetime_backbone: str = Field(
        default="EMForecaster",
        description="Backbone for datetime model: 'EMForecaster', 'LinearPatcher'",
    )
    num_head_layers: int = Field(
        default=2,
        description="Number of layers in the TSMixer head of the CyclicalEMForecaster",
    )
    head_d_model: int = Field(
        default=128,
        description="Hidden dimension for TSMixer head in the CyclicalEMForecaster model",
    )

    # Recurrent Model
    bidirectional: bool = Field(
        default=False,
        description="Whether to use bidirectional LSTM (typically for classification).",
    )
    last_state: bool = Field(
        default=True,
        description="Whether to use the last state hidden of the LSTM (typically for classification).",
    )
    avg_state: bool = Field(
        default=False,
        description="Whether to use the average the hidden states of the LSTM.",
    )

    # ROCKET
    max_dilation: int = Field(
        default=32,
        description="Value is proportional to the max dilation size for the ROCKET model's kernels.",
    )

    # DTW + k-NN
    num_neighbours: int = Field(default=5, description="Number of neighbors to use")
    knn_weights: str = Field(
        default="uniform",
        description="Weight function used in prediction. Options: 'uniform', 'distance'",
    )
    knn_metric: str = Field(
        default="dtw",
        description="Distance metric to use. Options: 'dtw', 'dtw_sakoe_chiba', 'softdtw'",
    )
    knn_metric_params: Dict = Field(
        default={}, description="Additional keyword arguments for the metric function"
    )

    # EMForecaster
    patch_norm: str = Field(
        default="layer",
        description="Normalization mode for the EMForecaster model",
    )
    patch_act: str = Field(
        default="gelu",
        description="Activation function for the EMForecaster model",
    )
    patch_embed_dim: int = Field(
        default=128, description="Embedding dimension for the EMForecaster model"
    )
    independent_patching: bool = Field(
        default=False,
        description="Whether to use independent patching for the EMForecaster model",
    )
    pos_enc: str = Field(
        default="learnable",
        description="Positional encoding for the EMForecaster model",
    )


class MLPMixer(BaseModel):
    tok_mixer_dim: int = Field(default=128, description="Token mixer hidden dimension")
    cha_mixer_dim: int = Field(
        default=128, description="Channel mixer hidden dimension"
    )
    pos_enc_type: str = Field(
        default="1d_sincos",
        description="Position encoding type, Options: 'learnable', '1d_sincos' or 'None'",
    )
    dropout: float = Field(
        default=0.0,
        description="Dropout value for Token-Mixer MLP and Channel-Mixer MLP",
    )


class DDP(BaseModel):
    ddp: bool = Field(
        default=False, description="Running distributive process on multiple nodes"
    )
    master_addr: str = Field(
        default="localhost", description="Address of the master node"
    )
    master_port: str = Field(default="12355", description="Port of the master node")
    shuffle: bool = Field(
        default=True, description="Shuffle the data for distributed training"
    )
    slurm: bool = Field(default=False, description="Running on SLURM cluster")
    find_unused_parameters: bool = Field(
        default=False, description="Find unused parameters for distributed training"
    )


class EarlyStopping(BaseModel):
    patience: int = Field(default=10, description="Patience for early stopping.")
    verbose: bool = Field(
        default=True, description="Verbose print for early stopping class."
    )
    delta: float = Field(
        default=0.0, description="Delta additive to the best scores for early stopping."
    )


class EMA(BaseModel):
    beta_0: float = Field(default=0.996, description="Starting EMA decay rate")
    beta_1: float = Field(default=1.0, description="Ending EMA decay rate")
    ema_momentum_scheduler: bool = Field(
        default=True, description="Scheduling for the EMA decay rate"
    )


class Scheduler(BaseModel):
    warmup_steps: int = Field(
        default=15,
        description="Number of warmup epochs for the scheduler for cosine_warmup",
    )
    start_lr: float = Field(
        default=1e-4,
        description="Starting learning rate for the scheduler for warmup, for cosine_warmup",
    )
    ref_lr: float = Field(
        default=1e-3,
        description="End learning rate for the scheduler after warmp, for cosine_warmup",
    )
    final_lr: float = Field(
        default=1e-6,
        description="Final learning rate by the end of the schedule (starting from ref_lr) for cosine_warmup",
    )
    T_max: int = Field(
        default=100,
        description="Maximum number of epochs for the scheduler for CosineAnnealingLR or cosine_warmup",
    )
    last_epoch: int = Field(
        default=-1,
        description="Last epoch for the scheduler or CosineAnnealingLR or cosine_warmup",
    )
    eta_min: float = Field(
        default=1e-6, description="Minimum learning rate for CosineAnnealingLR"
    )
    pct_start: float = Field(
        default=0.3,
        description="Percentage of the cycle (in number of steps) spent increasing the learning rate for OneCycleLR",
    )
    lradj: str = Field(
        default="type3",
        description="Learning rate adjustment type (ontop of scheduling). Options: 'type3', 'TST'",
    )


class OpenNeuro(BaseModel):
    patient_cluster: str = Field(
        default="jh",
        description="Patient cluster for the OpenNeuro dataset. Options: 'jh', 'pt', 'umf', 'ummc'",
    )
    kernel_size: int = Field(
        default=150, description="Kernel size for OpenNeuro dataset."
    )
    kernel_stride: int = Field(
        default=75, description="Kernel stride for OpenNeuro dataset."
    )
    pool_type: str = Field(
        default="avg", description="Pooling type for OpenNeuro dataset."
    )
    alpha: float = Field(
        default=1.0, description="Weight for the Normal Loss in the OpenNeuro dataset."
    )
    beta: float = Field(
        default=1.0, description="Weight for the Channel Loss in the OpenNeuro dataset."
    )
    gamma: float = Field(
        default=0.0,
        description="Weight for the variance of the channcel predictions in ChannelLOss.",
    )
    chi: float = Field(
        default=0.0,
        description="Weight for the entropy loss of the uncertainty/confidence coefficients to encourage diversity (not focused on a single window).",
    )
    ch_loss: bool = Field(
        default=False,
        description="Use channel-wise loss or not for classification tasks",
    )
    ch_loss_type: str = Field(
        default="BCE",
        description="Channel loss type for the OpenNeuro dataset. Options: 'BCE', 'CE'",
    )
    ch_var_loss: bool = Field(
        default=False,
        description="Whether to include the variance of the channel predictions in the loss for ChannelLoss.",
    )
    task: str = Field(
        default="binary",
        description="Task type for the OpenNeuro dataset. Options: 'binary', 'multi'. Where 'multi' includes surgical outcomes. 'multi' is only valid for 'pt' and 'ummc' clusters.",
    )
    ch_aggr: bool = Field(
        default=False,
        description="Whether to aggregate the channel latent representations before prediction.",
    )
    ch_reduction: str = Field(
        default="mean",
        description="Channel reduction type for the OpenNeuro dataset. Options: 'mean', 'max', 'sum'",
    )
    all_clusters: bool = Field(
        default=False,
        description="Whether to use all clusters for training + evaluation or not.",
    )
    loocv: bool = Field(
        default=False,
        description="Leave-one-out cross-validation for the OpenNeuro dataset on the patient clusters.",
    )
    train_clusters: List[str] = Field(
        default=["jh", "umf", "pt"],
        description="The patient clusters in Leave-one-out cross-validation used for training. Options, any subset of {'jh', 'pt', 'umf', 'ummc'}",
    )
    test_clusters: List[str] = Field(
        default=["umf"],
        description="The patient clusters in Leave-one-out cross-validation used for testing. Options, any subset of {'jh', 'pt', 'umf', 'ummc'}",
    )


class PatchTST(BaseModel):
    num_enc_layers: int = Field(
        default=3, description="Number of encoder layers for the PatchTST model."
    )
    d_model: int = Field(
        default=16, description="Model dimension for the PatchTST model."
    )
    d_ff: int = Field(
        default=128, description="FeedForward dimension for the PatchTST model."
    )
    num_heads: int = Field(
        default=4, description="Number of heads for the PatchTST model."
    )
    attn_dropout: float = Field(
        default=0.3,
        description="Dropout rate for attention mechanism in the PatchTST model.",
    )
    ff_dropout: float = Field(
        default=0.3,
        description="Dropout rate for feedforward mechanism in the PatchTST model.",
    )
    norm_mode: str = Field(
        default="batch1d", description="Normalization mode for the PatchTST model."
    )


class DLinear(BaseModel):
    moving_avg: int = Field(
        default=25, description="Moving average window for the DLinear model."
    )
    individual: bool = Field(
        default=False,
        description="Whether to use model channels together or separately.",
    )

    # EMForecaster
    final_moving_avg: int = Field(
        default=25,
        description="Moving average window for the final forecast in the DLinear model.",
    )


class TSMixer(BaseModel):
    num_enc_layers: int = Field(
        default=2, description="Number of encoder layers for the TSMixer model."
    )
    dropout: float = Field(
        default=0.3, description="Dropout rate for the TSMixer model."
    )
    d_model: int = Field(
        default=16,
        description="Hidden dimension of the Temporal and Channel MLPs for the TSMixer model.",
    )


class TimesNet(BaseModel):
    num_enc_layers: int = Field(
        default=2, description="Number of encoder layers for the TimesNet model."
    )
    d_model: int = Field(
        default=16, description="Model dimension for the TimesNet model."
    )
    d_ff: int = Field(
        default=128, description="FeedForward dimension for the TimesNet model."
    )
    num_kernels: int = Field(
        default=6, description="Number of kernels for the TimesNet model."
    )
    c_out: int = Field(
        default=1, description="Output channels for the TimesNet model for forecasting."
    )
    top_k: int = Field(
        default=3,
        description="Top k amplitudes used for the periodic slicing block in TimesNet.",
    )
    dropout: float = Field(
        default=0.3, description="Dropout rate for the TimesNet model."
    )


class ModernTCN(BaseModel):
    num_enc_layers: List[int] = Field(
        default=[2],
        description="Choose from {1, 2, 3} and can make it a list (in str format a,b,c,...) for multistaging with 5 possible stages [a,b,c,d,e] with each element from {1, 2, 3}. For example [1, 1] or [2, 2, 3].",
    )
    d_model: List[int] = Field(
        default=[16],
        description="The model dimension (i.e. Conv1D channel dimension) for each stage. Choose from {32, 64, 128, 256, 512}. Make a list (in str format a,b,c,...)  for multistaging, length equal to number of stages.",
    )
    ffn_ratio: int = Field(
        default=1,
        description="The expansion factor for the feed-forward networks in each block, d_ffn = d_model*ffn_ratio. Choose from {1, 2, 4, 8}",
    )
    dropout: float = Field(
        default=0.0, description="Dropout rate for the ModernTCN model."
    )
    class_dropout: float = Field(
        default=0.0, description="Dropout rate for the classification head."
    )
    large_size: List[int] = Field(
        default=[9],
        description="Size of the large kernel. Choose from {13, 31, 51, 71}. Make a list (in str format a,b,c,...)  for multistaging, length equal to number of stages.",
    )
    small_size: List[int] = Field(
        default=[5],
        description="Size of the small kernel Set to 5 for all experiments. Make a list (in str format a,b,c,...) for multistaging, length equal to number of stages.",
    )
    dw_dims: List[int] = Field(
        default=[256],
        description="Depthwise dimension for each stage. Set to 256 for all stages. Make a list (in str format a,b,c,...) for multistaging, length equal to number of stages.",
    )


class Global(BaseModel):
    exp: Experiment = Experiment()
    data: Data = Data()

    sl: SL = SL()
    ddp: DDP = DDP()
    early_stopping: EarlyStopping = EarlyStopping()
    ema: EMA = EMA()
    scheduler: Scheduler = Scheduler()
    open_neuro: OpenNeuro = OpenNeuro()
    mlp_mixer: MLPMixer = MLPMixer()
    patchtst: PatchTST = PatchTST()
    dlinear: DLinear = DLinear()
    tsmixer: TSMixer = TSMixer()
    timesnet: TimesNet = TimesNet()
    moderntcn: ModernTCN = ModernTCN()
    conf: Conformal = Conformal()


def load_config(file_path: str) -> Global:
    print(f"Received file_path in load_config: {file_path}")
    print(f"Absolute file_path in load_config: {os.path.abspath(file_path)}")
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return Global(**config_data)
