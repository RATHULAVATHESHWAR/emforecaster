import warnings

warnings.filterwarnings("ignore", message="h5py not installed")
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Supervised Models
from emforecaster.models.patchtst import PatchTST
from emforecaster.models.recurrent import RecurrentModel
from emforecaster.models.rf_emf_baselines import (
    MLP as RF_EMF_MLP,
    CNN as RF_EMF_CNN,
    LSTM as RF_EMF_LSTM,
    Transformer as RF_EMF_Transformer,
)
from emforecaster.models.linear import Linear
from emforecaster.models.dlinear import DLinear
from emforecaster.models.modern_tcn import ModernTCN
from emforecaster.models.timesnet import TimesNet
from emforecaster.models.tsmixer import TSMixer
from emforecaster.models.emforecaster import EMForecaster

# Unsupervised models
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

# Grid Search for scikit-learn models
from sklearn.model_selection import GridSearchCV

# Layers
from emforecaster.layers.patchtst.revin import RevIN

# Optimizers and Schedulers
from torch import optim
from emforecaster.utils.schedulers import WarmupCosineSchedule, PatchTSTSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


def get_model(args, generator=torch.Generator()):
    if args.exp.model_id == "PatchTST":
        model = PatchTST(
            num_enc_layers=args.sl.num_enc_layers,
            d_model=args.sl.d_model,
            d_ff=args.sl.d_ff,
            num_heads=args.sl.num_heads,
            num_channels=args.data.num_channels,
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            attn_dropout=args.sl.attn_dropout,
            ff_dropout=args.sl.ff_dropout,
            pred_dropout=args.sl.pred_dropout,
            batch_first=args.sl.batch_first,
            norm_mode=args.sl.norm_mode,
            revin=args.sl.revin,
            revout=args.sl.revout,
            revin_affine=args.sl.revin_affine,
            eps_revin=args.sl.eps_revin,
            patch_dim=args.data.patch_dim,
            stride=args.data.patch_stride,
            head_type=args.sl.head_type,
        )
    elif args.exp.model_id == "RecurrentModel":
        model = RecurrentModel(
            d_model=args.sl.d_model,
            backbone_id=args.exp.backbone_id,
            num_enc_layers=args.sl.num_enc_layers,
            pred_len=args.data.pred_len,
            bidirectional=args.sl.bidirectional,
            dropout=args.sl.dropout,
            seq_len=args.data.seq_len,
            patching=args.data.patching,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            num_channels=args.data.num_channels,
            head_type=args.sl.head_type,
            norm_mode=args.sl.norm_mode,
            revin=args.sl.revin,
            revout=args.sl.revout,
            revin_affine=args.sl.revin_affine,
            eps_revin=args.sl.eps_revin,
            last_state=args.sl.last_state,
            avg_state=args.sl.avg_state,
        )
    elif args.exp.model_id == "Linear":
        model = Linear(
            in_features=args.data.seq_len,
            out_features=args.data.pred_len,
            norm_mode=args.sl.norm_mode,
        )
    elif args.exp.model_id == "DLinear":
        model = DLinear(
            task=args.exp.task,
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            num_classes=args.data.pred_len,
            moving_avg=args.dlinear.moving_avg,
            individual=args.dlinear.individual,
            return_head=args.sl.return_head,
        )
    elif args.exp.model_id == "EMForecaster":
        model = EMForecaster(
            args=args,
            seed=args.exp.seed,
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            revin=args.sl.revin,
            revout=args.sl.revout,
            revin_affine=args.sl.revin_affine,
            eps_revin=args.sl.eps_revin,
            patch_model_id=args.exp.patch_model_id,
            backbone_id=args.exp.backbone_id,
            patch_norm=args.sl.patch_norm,
            patch_act=args.sl.patch_act,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            patch_embed_dim=args.sl.patch_embed_dim,
            independent_patching=args.sl.independent_patching,
            pos_enc=args.sl.pos_enc,
        )

    elif args.exp.model_id == "ModernTCN":
        model = ModernTCN(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            num_classes=args.data.pred_len,
            num_channels=args.data.num_channels,
            task=args.exp.task,
            return_head=args.sl.return_head,
            dropout=args.sl.dropout,
            class_dropout=args.moderntcn.class_dropout,
            ffn_ratio=args.moderntcn.ffn_ratio,
            num_enc_layers=args.moderntcn.num_enc_layers,
            large_size=args.moderntcn.large_size,
            d_model=args.moderntcn.d_model,
            revin=args.sl.revin,
            affine=args.sl.revin_affine,
            small_size=args.moderntcn.small_size,
            dw_dims=args.moderntcn.dw_dims,
        )

    elif args.exp.model_id == "TimesNet":
        model = TimesNet(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            d_model=args.timesnet.d_model,
            d_ff=args.timesnet.d_ff,
            num_enc_layers=args.sl.num_enc_layers,
            num_kernels=args.timesnet.num_kernels,
            c_out=args.timesnet.c_out,
            top_k=args.timesnet.top_k,
            dropout=args.sl.dropout,
            task=args.exp.task,
            revin=args.sl.revin,
            revin_affine=args.sl.revin_affine,
            revout=args.sl.revout,
            eps_revin=args.sl.eps_revin,
            return_head=args.sl.return_head,
        )

    elif args.exp.model_id == "KNN":
        if args.exp.grid_search:
            model = GridSearchCV(
                KNeighborsTimeSeriesClassifier(),
                param_grid=args.sl.knn_param_grid,
                n_jobs=args.exp.sklearn_n_jobs,
                cv=args.sl.cv,
                verbose=args.exp.sklearn_verbose,
            )
        else:
            model = KNeighborsTimeSeriesClassifier(
                n_neighbors=args.sl.num_neighbours,
                weights=args.sl.knn_weights,
                metric=args.sl.knn_metric,
                metric_params=args.sl.knn_metric_params,
                n_jobs=args.exp.sklearn_n_jobs,
                verbose=args.exp.sklearn_verbose,
            )
    elif args.exp.model_id == "TSMixer":
        model = TSMixer(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_enc_layers=args.sl.num_enc_layers,
            d_model=args.sl.d_model,
            num_channels=args.data.num_channels,
            dropout=args.sl.dropout,
            revin=args.sl.revin,
            revin_affine=args.sl.revin_affine,
            revout=args.sl.revout,
            eps_revin=args.sl.eps_revin,
        )
    
    elif args.exp.model_id == "RF_EMF_MLP":
        model = RF_EMF_MLP(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            d_model=args.sl.d_model,
        )
    elif args.exp.model_id == "RF_EMF_CNN":
        model = RF_EMF_CNN(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_kernels=args.sl.num_kernels,
        )
    elif args.exp.model_id == "RF_EMF_LSTM":
        model = RF_EMF_LSTM(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            d_model=args.sl.d_model,
        )
    elif args.exp.model_id == "RF_EMF_Transformer":
        model = RF_EMF_Transformer(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            d_model=args.sl.d_model,
            num_heads=args.sl.num_heads,
            num_enc_layers=args.sl.num_enc_layers,
        )
    else:
        raise ValueError("Please select a valid model_id.")
    return model


def get_optim(args, model, optimizer_type="adamw", flag="sl"):
    if args.exp.sklearn:
        return None

    optimizer_classes = {"adam": optim.Adam, "adamw": optim.AdamW}
    if optimizer_type not in optimizer_classes:
        raise ValueError("Please select a valid optimizer.")
    optimizer_class = optimizer_classes[optimizer_type]

    param_groups = exclude_weight_decay(
        model, args, flag
    )  # Exclude bias and normalization parameters from weight decay

    optimizer = optimizer_class(param_groups)  # Set optimizer

    return optimizer


def exclude_weight_decay(model, args, flag="sl"):
    # Separate parameters into those that will use weight decay and those that won't
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or isinstance(
                param, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, RevIN)
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {"params": decay_params, "weight_decay": eval(f"args.{flag}.weight_decay")},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return param_groups


def get_scheduler(args, scheduler_type, training_mode, optimizer, num_batches=0):
    if args.exp.sklearn:
        return None

    if scheduler_type == "cosine_warmup" and training_mode == "pretrain":
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=args.scheduler.warmup_steps,
            start_lr=args.scheduler.start_lr,
            ref_lr=args.scheduler.ref_lr,
            T_max=args.scheduler.T_max,
            final_lr=args.scheduler.final_lr,
        )
    elif scheduler_type == "cosine":
        if training_mode == "downstream":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.downstream.epochs,
                eta_min=args.downstream.lr * 1e-2,
                last_epoch=args.downstream.last_epoch,
            )
        elif training_mode == "supervised":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.sl.epochs,
                eta_min=args.sl.lr * 1e-2,
                last_epoch=args.scheduler.last_epoch,
            )
    elif scheduler_type == "patchtst" and training_mode == "supervised":
        scheduler = PatchTSTSchedule(optimizer, args, num_batches)
    elif scheduler_type == "onecycle" and training_mode == "supervised":
        scheduler = OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=num_batches,
            pct_start=args.scheduler.pct_start,
            epochs=args.sl.epochs,
            max_lr=args.sl.lr,
        )
    elif scheduler_type is None:
        return None
    else:
        raise ValueError("Please select a valid scheduler_type.")
    return scheduler


def get_criterion(args, criterion_type):
    if criterion_type == "MSE":
        return nn.MSELoss()
    elif criterion_type == "SmoothL1":
        return nn.SmoothL1Loss()
    elif criterion_type == "BCE":
        return nn.BCEWithLogitsLoss()
    elif criterion_type == "BCE_normal":
        return nn.BCELoss()
    elif criterion_type == "CE":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Please select a valid criterion_type.")


def forward_pass(args, model, batch, model_id, device):

    if model_id in {
        "PatchTST",
        "PatchTSTOG",
        "TSMixer",
        "RecurrentModel",
        "Linear",
        "DLinear",
        "ModernTCN",
        "TimesNet",
        "RF_EMF_CNN",
        "RF_EMF_MLP",
        "RF_EMF_LSTM",
        "RF_EMF_Transformer",
        "EMForecaster",
        "CyclicalEMForecaster",
    }:

        x = batch[0]
        x = x.to(device)

        if args.data.difference_input:
            x = x.diff(dim=1)

            # Pad the last dimension with the last value of the sequence
            x = torch.cat([x, x[:, -1].unsqueeze(1)], dim=1)

        output = model(x)
        # output = output.view(-1)[0].view(1) if args.sl.batch_size==1 and args.open_neuro.task else output
    else:
        raise ValueError("Please select a valid model_id.")

    return output


def prepare_output_and_target(output, target, args, device):
    out = output.to(
        device
    ).squeeze()  # (batch_size, 1, 1) -> (batch_size,) for binary task
    target = target.to(device)

    if args.open_neuro.task == "binary":
        if out.dim() == 0:  # scalar output for batch_size == 1
            out = out.unsqueeze(0)  # (,) -> (1,)
    elif args.open_neuro.task == "multi":
        if out.dim() == 1:  # (num_classes,) for batch_size == 1
            out = out.unsqueeze(0)  # (num_classes,) -> (1, num_classes)

    # Ensure out and target have the same size
    if args.open_neuro.task == "binary":
        assert (
            out.size() == target.size()
        ), f"Size mismatch: out {out.size()}, target {target.size()}"
    elif args.open_neuro.task == "multi":
        assert out.size(0) == target.size(
            0
        ), f"Size mismatch: out {out.size()}, target {target.size()}"

    target = target.long() if args.open_neuro.task == "multi" else target.float()

    return out, target


def compute_loss(output, batch, criterion, model_id, args, device):
    if model_id in {
        "PatchTST",
        "PatchTSTOG",
        "TSMixer",
        "RecurrentModel",
        "Linear",
        "DLinear",
        "ModernTCN",
        "TimesNet",
        "RF_EMF_CNN",
        "RF_EMF_MLP",
        "RF_EMF_LSTM",
        "RF_EMF_Transformer",
        "EMForecaster",
        "CyclicalEMForecaster",
    }:
        out, target = prepare_output_and_target(output, batch[1], args, device)
        # output = output.squeeze()
        # target = batch[1].to(device)
        # target = target.squeeze() if args.sl.batch_size==1 else target
        loss = criterion(out, target)
    else:
        raise ValueError("Please select a valid model_id.")

    return loss


def model_update(model, loss, optimizer, model_id, alpha=0.6):
    if model_id in {
        "PatchTST",
        "TSMixer",
        "RecurrentModel",
        "Linear",
        "DLinear",
        "ModernTCN",
        "TimesNet",
        "RF_EMF_CNN",
        "RF_EMF_MLP",
        "RF_EMF_LSTM",
        "RF_EMF_Transformer",
        "EMForecaster",
        "CyclicalEMForecaster",
    }:
        loss.backward()
        # check_gradients(model)
        optimizer.step()
    else:
        raise ValueError("Please select a valid model_id.")


def check_gradients(model, threshold_low=1e-5, threshold_high=1e2):
    vanishing = []
    exploding = []
    normal = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < threshold_low:
                vanishing.append((name, grad_norm))
            elif grad_norm > threshold_high:
                exploding.append((name, grad_norm))
            else:
                normal.append((name, grad_norm))

    print(f"Gradient statistics:")
    print(
        f"  Total parameters with gradients: {len(vanishing) + len(exploding) + len(normal)}"
    )
    print(
        f"  Vanishing gradients: {len(vanishing)} ({len(vanishing) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)"
    )
    print(
        f"  Exploding gradients: {len(exploding)} ({len(exploding) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)"
    )
    print(
        f"  Normal gradients: {len(normal)} ({len(normal) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)"
    )

    if vanishing:
        print("\nVanishing gradients:")
        for name, grad_norm in vanishing[:10]:  # Print first 10
            print(f"  {name}: {grad_norm}")
        if len(vanishing) > 10:
            print(f"  ... and {len(vanishing) - 10} more")

    if exploding:
        print("\nExploding gradients:")
        for name, grad_norm in exploding[:10]:  # Print first 10
            print(f"  {name}: {grad_norm}")
        if len(exploding) > 10:
            print(f"  ... and {len(exploding) - 10} more")

    # Compute and print gradient statistics
    all_grads = [
        param.grad.norm().item()
        for name, param in model.named_parameters()
        if param.grad is not None
    ]
    if all_grads:
        print("\nGradient norm statistics:")
        print(f"  Mean: {np.mean(all_grads):.6f}")
        print(f"  Median: {np.median(all_grads):.6f}")
        print(f"  Std: {np.std(all_grads):.6f}")
        print(f"  Min: {np.min(all_grads):.6f}")
        print(f"  Max: {np.max(all_grads):.6f}")
