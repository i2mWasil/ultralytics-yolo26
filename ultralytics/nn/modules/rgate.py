"""Reflection/Glare Suppression Gate for YOLO26 (rainy-night wet-road reflections).

Place this file at: ultralytics/nn/modules/rgate.py

Two designs: rgate/yolo26-rgate-p3.yaml (one gate on P3: layer 16, stride 8) and rgate/yolo26-rgate-p3p4.yaml
(gates on P3 and on P4: layer 19, stride 16). Each gate has its own weights and its own gamma. The gates feed only
the Detect head; P5 is never gated.

Data flow of one gate (YOLO26n, 640x640 input; P4 numbers in brackets):
    img (B,3,640,640, values 0..1) --ReflectionPrior, max-pool 8 [16]--> R (B,1,80,80) [(B,1,40,40)]
    F = neck output, layer 16 (B,64,80,80) [layer 19 (B,128,40,40)]
    concat(F,R) -> Conv3x3+BN+SiLU (hidden = max(16, C/4): 16 [32]) -> Conv3x3 (B,1,h,w) -> sigmoid = S
    F_out = F * (1 - tanh(g * gamma) * S)   same shape as F   gamma: learnable scalar, starts at 0; g = gamma_gain = 10
    tanh keeps the factor between 0 and 2: at most full suppression, at most double strength, never a sign flip.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


class ReflectionPrior(nn.Module):
    """Fixed (no learned weights) 'bright and colourless' score, pooled down to the gated feature map's grid.

    R = clip((V - v0) / (1 - v0), 0, 1) * (1 - S), then max-pooled by `stride`.
    V = max(r,g,b) (brightness), S = (max - min) / max (colourfulness).
    The v0 threshold stops dim grey pixels (e.g. letterbox padding, V=0.447, S=0) from scoring.
    Max-pool (not average) so a tiny 2-3 pixel glint still shows up in its cell (8x8 at P3, 16x16 at P4).
    Uses only basic tensor ops so it exports to ONNX / TensorRT.
    """

    def __init__(self, v0: float = 0.6, stride: int = 8):
        super().__init__()
        self.v0, self.stride = v0, stride

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        mx = img.amax(1, keepdim=True)
        mn = img.amin(1, keepdim=True)
        # clamp at 1e-3, not "+ 1e-6": 1e-6 is too small for FP16 (TensorRT may round it to 0 -> 0/0 = NaN on
        # black pixels). Pixels this dark have brightness score 0 anyway, so the result is unchanged.
        sat = (mx - mn) / mx.clamp(min=1e-3)
        bright = ((mx - self.v0) / (1.0 - self.v0)).clamp(0.0, 1.0)
        r = bright * (1.0 - sat)
        return F.max_pool2d(r, self.stride, self.stride)


class ReflectionGate(nn.Module):
    """Learned spatial suppression mask applied to one feature map (P3 or P4).

    Input:  f   (B, C, H/s, W/s)  feature map from the neck, s = `stride` (8 for P3, 16 for P4)
            img (B, 3, H, W)      the network input image (supplied by the patched _predict_once)
    Output: (B, C, H/s, W/s)      same shape as f, so the Detect head is unchanged.

    S close to 1 = "this cell looks like a reflection, turn it down".
    gamma starts at 0, so on step 0 the model is exactly the original YOLO26.
    """

    needs_image = True  # flag read by the patched BaseModel._predict_once

    def __init__(self, c1: int, hidden: int | None = None, v0: float = 0.6, stride: int = 8, gamma_gain: float = 10.0,
                 use_prior: bool = True):
        super().__init__()
        self.use_prior = use_prior  # False = control arm: same gate, prior replaced by zeros
        self.gamma_gain = gamma_gain  # strength = tanh(gain * gamma); a single scalar gets a tiny gradient, see CLAUDE.md
        hidden = hidden or max(16, c1 // 4)
        self.prior = ReflectionPrior(v0, stride)
        self.fuse = Conv(c1 + 1, hidden, 3, 1)  # Conv3x3 + BN + SiLU
        self.mask = nn.Conv2d(hidden, 1, 3, 1, 1)
        nn.init.constant_(self.mask.bias, -2.0)  # S starts low (~0.12) everywhere
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_r = self.last_s = self.last_logit = None  # kept for the auxiliary loss and for visualisation

    def __getstate__(self):  # never pickle the cached maps into checkpoints
        state = self.__dict__.copy()
        state["last_r"] = state["last_s"] = state["last_logit"] = None
        return state

    @property
    def level(self) -> str:
        """Feature level this gate works on, from its prior stride: 8 -> 'P3', 16 -> 'P4'."""
        return f"P{int(round(math.log2(self.prior.stride)))}"

    def forward(self, f: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        r = self.prior(img)
        if r.shape[-2:] != f.shape[-2:]:
            raise RuntimeError(f"ReflectionGate: prior grid {tuple(r.shape[-2:])} does not match the feature map "
                               f"{tuple(f.shape[-2:])}; the gate's prior stride ({self.prior.stride}) must equal the "
                               "stride of the layer it gates (8 for P3, 16 for P4)")
        if not self.use_prior:
            r = torch.zeros_like(r)
        logit = self.mask(self.fuse(torch.cat((f, r), 1)))
        s = logit.sigmoid()
        # the loss uses the logit (BCE-with-logits is AMP-safe; plain BCE on s raises under autocast)
        if self.training:
            self.last_r, self.last_s, self.last_logit = r, s, logit
        else:
            self.last_r, self.last_s, self.last_logit = r.detach(), s.detach(), logit.detach()
        return f * (1.0 - self.strength() * s)

    def strength(self) -> torch.Tensor:
        """Effective gate strength tanh(gain * gamma), in (-1, 1). Positive = suppress, negative = boost.

        Starts at 0 (gamma = 0) with slope 1, so training starts exactly as before the tanh was added.
        """
        return torch.tanh(self.gamma_gain * self.gamma)


def boxes_to_grid_mask(batch: dict, h: int, w: int, device) -> torch.Tensor:
    """(B,1,h,w) mask, 1 where a grid cell overlaps any ground-truth box of that image.

    Overlap, not "cell centre inside box": a box narrower than one cell (8 px at P3) can miss every cell centre,
    and the aux loss would then tell the gate to suppress that small object (e.g. a distant sign or headlight).
    """
    bs = batch["img"].shape[0]
    boxes = batch["bboxes"].to(device).float()  # (N,4) normalised xywh
    if boxes.numel() == 0:
        return torch.zeros(bs, 1, h, w, device=device)
    idx = batch["batch_idx"].to(device).long().view(-1)
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx, gy = gx.reshape(1, -1), gy.reshape(1, -1)
    x1, y1 = boxes[:, 0:1] - boxes[:, 2:3] / 2, boxes[:, 1:2] - boxes[:, 3:4] / 2
    x2, y2 = boxes[:, 0:1] + boxes[:, 2:3] / 2, boxes[:, 1:2] + boxes[:, 3:4] / 2
    hx, hy = 0.5 / w, 0.5 / h  # half a cell: cell spans [centre - half, centre + half]
    inside = ((gx - hx < x2) & (gx + hx > x1) & (gy - hy < y2) & (gy + hy > y1)).float()  # (N, h*w), overlap
    owner = F.one_hot(idx, bs).float().t()  # (bs, N): which image each box belongs to
    mask = (owner @ inside).clamp(max=1.0)  # matmul works on CPU, CUDA and Apple MPS
    return mask.view(bs, 1, h, w)


class RGateE2ELoss(E2ELoss):
    """YOLO26 end-to-end loss + optional two-sided gate loss.

    Target for S: high where the prior says 'reflection' AND no object box; low inside object boxes.
    Cells that are neither (dark background) are ignored, so the gate is not pushed anywhere there.
    With several gates (P3 and P4) the gate loss is the MEAN over the gates, each on its own grid,
    so lambda means the same total aux weight whether one or two gates are used.
    """

    def __init__(self, model, loss_fn=v8DetectionLoss):
        super().__init__(model, loss_fn)
        self.gates = [m for m in model.modules() if isinstance(m, ReflectionGate)]
        cfg = getattr(model, "yaml", {}) or {}
        self.lam = float(cfg.get("rgate_lambda", 0.0))  # 0 = gate learns from detection loss only
        self.tau = float(cfg.get("rgate_tau", 0.3))

    def gate_loss(self, g: ReflectionGate, batch: dict) -> torch.Tensor | None:
        """Weighted BCE between one gate's S (as logits) and its target, on that gate's grid. None if not usable."""
        if g.last_logit is None:
            return None
        z, r = g.last_logit.float(), g.last_r.detach().float()
        if z.shape[0] != batch["img"].shape[0]:
            return None
        b = boxes_to_grid_mask(batch, z.shape[2], z.shape[3], z.device)
        target = r * (1.0 - b)
        weight = ((b > 0) | (r > self.tau)).float()
        bce = F.binary_cross_entropy_with_logits(z, target, reduction="none")
        return (bce * weight).sum() / weight.sum().clamp(min=1.0)

    def __call__(self, preds, batch):
        loss, items = super().__call__(preds, batch)
        if not self.gates or self.lam <= 0:
            return loss, items
        parts = [lg for lg in (self.gate_loss(g, batch) for g in self.gates) if lg is not None]
        if len(parts) != len(self.gates):  # a gate missed this batch: skip the aux loss rather than mix levels
            return loss, items
        l_gate = torch.stack(parts).mean()
        loss = loss + self.lam * l_gate * batch["img"].shape[0]  # detection loss is also scaled by batch size
        items = dict(items)
        items["gate"] = l_gate.detach()
        return loss, items
