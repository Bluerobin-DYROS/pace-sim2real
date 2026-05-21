# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations
from dataclasses import MISSING

import torch

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg
from pace_sim2real.utils import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model.

    This class extends the base DCMotorCfg with Pace-specific parameters.
    """
    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: dict[str, float] | float | None = 0.0
    max_delay: int | None = 0


@configclass
class PaceActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for Pace per-joint LSTM actuator model.

    The network output is multiplied by ``torque_scale`` to recover Nm. The resulting torque
    is then delayed by ``max_delay`` simulation steps (per-env delay set via
    :meth:`pace_actuator.PaceActuatorNetLSTM.update_time_lags`).

    Stiffness and damping are inherited from DCMotorCfg but unused — the LSTM replaces the
    PD controller. ``saturation_effort`` is also inherited and defaults to a large value so
    DCMotor's BEMF clipping never fires (raw LSTM output is passed through).
    """

    class_type: type = pace_actuator.PaceActuatorNetLSTM

    # PD gains unused by the network; default to None so the actuator falls back to zeros.
    stiffness: dict[str, float] | float | None = None
    damping: dict[str, float] | float | None = None

    # DCMotor's saturation_effort is required by parent __init__; large default disables
    # the BEMF torque-speed envelope.
    saturation_effort: float = 1.0e9

    network_dir: str = MISSING
    """Directory containing per-joint TorchScript files."""

    network_prefix: str = ""
    """Filename prefix prepended to ``net_joint_names`` entries (e.g. ``"p73_lstm_"``)."""

    net_joint_names: list[str] = MISSING
    """Per-joint network filename tokens, in the same order as :attr:`joint_names_expr`."""

    torque_scale: float = 100.0
    """Multiplier applied to the network output (training scaled torque by 1/torque_scale)."""

    encoder_bias: dict[str, float] | list[float] | float | None = 0.0
    """Per-joint encoder bias subtracted from the measured joint position before the network."""

    max_delay: int = 0
    """Maximum action-delay buffer length, in simulation steps."""
