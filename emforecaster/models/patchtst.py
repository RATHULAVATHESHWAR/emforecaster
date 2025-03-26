import torch
from torch import Tensor
import torch.nn as nn
from emforecaster.layers.patchtst.revin import RevIN
from emforecaster.layers.patchtst.pos_enc import PositionalEncoding
from emforecaster.layers.patcher import Patcher
from emforecaster.layers.patchtst.backbone import PatchTSTBackbone
from emforecaster.utils.weight_init import xavier_init
from typing import Optional


class PatchTST(nn.Module):
    def __init__(
        self,
        num_enc_layers,
        d_model,
        d_ff,
        num_heads,
        num_channels,
        seq_len,
        pred_len,
        attn_dropout=0.0,
        ff_dropout=0.0,
        pred_dropout=0.0,
        batch_first=True,
        norm_mode="batch1d",
        revin=True,
        revout=True,
        revin_affine=True,
        eps_revin=1e-5,
        patch_dim=16,
        stride=1,
        return_head=True,
        head_type="linear",
        channel_independent=False,  # Head only
        target_channels: Optional[list] = None,  # Head only
    ) -> None:
        super(PatchTST, self).__init__()

        # Parameters
        self.num_patches = int((seq_len - patch_dim) / stride) + 2
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.target_channels = target_channels
        self.revout = revout

        # Initialize layers
        if revin:
            self._init_revin()
        else:
            self._revin = None
            self.revout = None

        self.patcher = Patcher(patch_dim, stride)
        self.pos_enc = PositionalEncoding(patch_dim, d_model, self.num_patches)
        self.backbone = PatchTSTBackbone(
            num_enc_layers=num_enc_layers,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_channels=num_channels,
            num_patches=self.num_patches,
            pred_len=pred_len,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            pred_dropout=pred_dropout,
            batch_first=batch_first,
            norm_mode=norm_mode,
            return_head=return_head,
            head_type=head_type,
            channel_independent=channel_independent,
            target_channels=target_channels,
        )

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self):
        self._revin = True
        self.revin = RevIN(
            num_channels=self.num_channels,
            eps=self.eps_revin,
            affine=self.revin_affine,
            target_channels=self.target_channels,
        )

    def forward(self, x, y=None, ch_ids=None):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        # Patcher
        x = self.patcher(x)

        # Project + Positional Encoding
        x = self.pos_enc(x)

        # Transformer + Linear Head
        x = self.backbone(x, y, ch_ids)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x