"""Jaume-Actor prototype: perception -> single forward -> discrete choice.

One deliberate design constraint drives everything here: **no autoregressive
decode at inference**. The planning report measured 154 ms per token on a 350M
model (T=1) — ten tokens of action text is already 1.5 s, which is unplayable.
So the actor never emits a string. It scores a fixed candidate set (UI elements)
or a fixed action set (maze), takes an argmax, and stops. That is one forward
pass per tick, and the tick cost is known in advance.

The second constraint is the loss. The planning report §4.7 showed that
per-pair BCE collapses to "predict negative" and learns an action prior while
ignoring state. Here the loss is **softmax CE across the candidate set inside
one state**, so an action-only prior is provably useless.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from envs import N_WORDS


def cnn(in_ch: int, out: int, width: int = 32, depth: int = 3, in_size: int = 32) -> nn.Sequential:
    layers, c, size = [], in_ch, in_size
    for i in range(depth):
        co = width * (2**min(i, 2))
        layers += [
            nn.Conv2d(c, co, 3, stride=2 if i == 0 else 2, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        ]
        c = co
        size //= 2
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class SpatialCNN(nn.Module):
    """CNN that KEEPS the spatial layout.

    The pooled variant (cnn) is right for classifying a whole crop, but it is
    fatal for control: after a global average pool the network cannot tell
    *where* the agent or the goal is, only what the frame contains on
    average. This version downsamples but never pools away position, and
    concatenates normalised (x, y) coordinate channels so the geometry of the
    grid is explicit.
    """

    def __init__(self, in_ch: int = 3, width: int = 32, depth: int = 3, in_size: int = 128, out_dim: int = 256):
        super().__init__()
        layers, c = [], in_ch + 2  # +2 coordinate channels
        for i in range(depth):
            co = width * (2**i)
            layers += [
                nn.Conv2d(c, co, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            ]
            c = co
        self.body = nn.Sequential(*layers)
        hw = in_size // (2**depth)
        self.proj = nn.Sequential(nn.Linear(c * hw * hw, out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        f = self.body(torch.cat([x, ys, xs], dim=1))
        return self.proj(f.flatten(1))


def in_size_div(size: int, depth: int) -> int:
    return size // (2**depth)


class UIActor(nn.Module):
    """Scores candidate elements for one instruction. One forward = one decision."""

    def __init__(self, d: int = 128, use_screen: bool = True, use_crop: bool = True, use_box: bool = True):
        super().__init__()
        self.use_screen, self.use_crop, self.use_box = use_screen, use_crop, use_box
        self.crop = cnn(3, d, width=24, depth=3, in_size=32) if use_crop else None
        self.screen = cnn(3, 64, width=24, depth=4, in_size=128) if use_screen else None
        self.text = nn.EmbeddingBag(N_WORDS, 64, mode="mean")
        self.ctx = nn.Sequential(nn.Linear(64 + (64 if use_screen else 0), d), nn.ReLU(inplace=True))
        self.score = nn.Sequential(
            nn.Linear(d + (d if use_crop else 0) + (4 if use_box else 0), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, crops, boxes, screen, text, mask):
        B, N = boxes.shape[:2]
        t = self.text(text)  # (B,64)
        if self.use_screen:
            t = torch.cat([t, self.screen(screen)], dim=-1)
        ctx = self.ctx(t)  # (B,d)
        parts = [ctx.unsqueeze(1).expand(B, N, -1)]
        if self.use_crop:
            parts.append(self.crop(crops.reshape(B * N, *crops.shape[2:])).reshape(B, N, -1))
        if self.use_box:
            parts.append(boxes)
        x = torch.cat(parts, dim=-1)
        logits = self.score(x).squeeze(-1)  # (B,N)
        return logits.masked_fill(~mask, -1e4)


class MazeActor(nn.Module):
    """Frame -> 4 action logits. Closed-loop at a fixed tick, no history module.

    Uses SpatialCNN so that position survives to the head. A flat MLP on
    pooled features cannot solve the maze (measured: 0.0 closed-loop success),
    because "which way is the goal" is a spatial question.
    """

    def __init__(self, n_act: int = 4, d: int = 128, width: int = 32, depth: int = 3):
        super().__init__()
        self.trunk = SpatialCNN(in_ch=3, width=width, depth=depth)
        self.head = nn.Sequential(nn.Linear(256, d), nn.ReLU(inplace=True), nn.Linear(d, n_act))

    def forward(self, frame):
        return self.head(self.trunk(frame))
