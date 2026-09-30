# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).


# Preserve archived computational syntax for direct source comparison.
# ruff: noqa: UP008

from __future__ import annotations

import math

import torch.nn as nn

from .skeleton_DME import SkeletonConv, SkeletonPool


class SkeletonResidual(nn.Module):
    def __init__(
        self,
        topology,
        neighbour_list,
        joint_num,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        bias,
        extra_conv,
        pooling_mode,
        activation,
        last_pool,
    ):
        super(SkeletonResidual, self).__init__()

        kernel_even = False if kernel_size % 2 else True

        seq = []
        for _ in range(extra_conv):
            # (T, J, D) => (T, J, D)
            seq.append(
                SkeletonConv(
                    neighbour_list,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    joint_num=joint_num,
                    kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                    stride=1,
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=bias,
                )
            )
            seq.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        # (T, J, D) => (T/2, J, 2D)
        seq.append(
            SkeletonConv(
                neighbour_list,
                in_channels=in_channels,
                out_channels=out_channels,
                joint_num=joint_num,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias,
                add_offset=False,
            )
        )
        # G1 edit: EMAGE hard-coded GroupNorm(10,·) requires out_channels%10==0, which holds
        # for SMPL-X widths (240 etc.) but crashes for the G1 topology's widths. Use gcd so it
        # degrades to a valid group count (>=1) for any width; ==10 when 10 divides out_channels.
        seq.append(nn.GroupNorm(math.gcd(out_channels, 10) or 1, out_channels))
        self.residual = nn.Sequential(*seq)

        # (T, J, D) => (T/2, J, 2D)
        self.shortcut = SkeletonConv(
            neighbour_list,
            in_channels=in_channels,
            out_channels=out_channels,
            joint_num=joint_num,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=True,
            add_offset=False,
        )

        seq = []
        # (T/2, J, 2D) => (T/2, J', 2D)
        pool = SkeletonPool(
            edges=topology,
            pooling_mode=pooling_mode,
            channels_per_edge=out_channels // len(neighbour_list),
            last_pool=last_pool,
        )
        if len(pool.pooling_list) != pool.edge_num:
            seq.append(pool)
        seq.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        self.common = nn.Sequential(*seq)

    def forward(self, input):
        output = self.residual(input) + self.shortcut(input)

        return self.common(output)
