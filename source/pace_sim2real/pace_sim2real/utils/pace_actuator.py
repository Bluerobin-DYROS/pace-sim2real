# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import DCMotor
from isaaclab.utils import DelayBuffer
from isaaclab.utils.assets import read_file
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    # only for type checking
    from .pace_actuator_cfg import PaceActuatorNetLSTMCfg, PaceDCMotorCfg


class PaceDCMotor(DCMotor):
    """Pace DC Motor actuator model with encoder bias and action delay.

    The actuator models a DC motor whose controller receives joint positions in the encoder
    frame by adding a per-joint encoder bias to the true joint positions. In other words,
    the controller operates on biased (encoder) positions rather than the true joint positions.

    The torque command computed by the PD controller is applied after a configurable delay
    (in simulation steps) to represent latency between command calculation and actuation.

    The software implementation is inspired by DelayedPDActuator.
    """

    cfg: PaceDCMotorCfg

    def __init__(self, cfg: PaceDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)

        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        self.torques_delay_buffer.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute actuator model with encoder bias added to joint positions (joint position in encoder frame, not simulation frame)
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(control_action_sim.joint_efforts)
        return control_action_sim


class PaceActuatorNetLSTM(DCMotor):
    """Pace actuator with per-joint LSTM networks, encoder bias, and torque-output delay.

    Inheritance: ``ActuatorBase → IdealPDActuator → DCMotor → PaceActuatorNetLSTM``.

    * Encoder bias subtracted from measured joint position before forming
      the LSTM input (``pos_err = target - (joint_pos - encoder_bias)``).

    """

    cfg: PaceActuatorNetLSTMCfg

    def __init__(self, cfg: PaceActuatorNetLSTMCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # ----- per-joint LSTM TorchScript networks -----
        if len(cfg.net_joint_names) != len(cfg.joint_names_expr):
            raise ValueError(
                f"net_joint_names length ({len(cfg.net_joint_names)}) must match "
                f"joint_names_expr length ({len(cfg.joint_names_expr)})."
            )
        joint_to_net = {jn: nn for jn, nn in zip(cfg.joint_names_expr, cfg.net_joint_names)}
        missing = [jn for jn in self.joint_names if jn not in joint_to_net]
        if missing:
            raise ValueError(
                f"PaceActuatorNetLSTM cfg is missing net_joint_names entries for joints: {missing}. "
                f"joint_names_expr must list every joint resolved by the articulation."
            )

        networks: list[torch.nn.Module] = []
        for jn in self.joint_names:
            pt_path = os.path.join(cfg.network_dir, f"{cfg.network_prefix}{joint_to_net[jn]}.pt")
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Actuator net not found: {pt_path}")
            file_bytes = read_file(pt_path)
            net = torch.jit.load(file_bytes, map_location=self._device).eval()
            networks.append(net)
        self.networks = torch.nn.ModuleList(networks)

        # Sanity-check that all networks share the same LSTM architecture.
        first_lstm_sd = networks[0].lstm.state_dict()
        num_layers = len(first_lstm_sd) // 4
        hidden_dim = first_lstm_sd["weight_hh_l0"].shape[1]
        for j, net in enumerate(networks):
            sd = net.lstm.state_dict()
            if len(sd) // 4 != num_layers or sd["weight_hh_l0"].shape[1] != hidden_dim:
                raise ValueError(
                    f"Network for joint '{self.joint_names[j]}' has mismatched LSTM "
                    f"architecture (expected layers={num_layers}, hidden={hidden_dim})."
                )

        # ----- persistent LSTM hidden/cell state (stateful inference) -----
        # Shape: (num_joints, num_layers, num_envs, hidden_dim)

        self._lstm_num_layers = num_layers
        self._lstm_hidden_dim = hidden_dim
        self.sea_hidden_state = torch.zeros(
            self.num_joints, num_layers, self._num_envs, hidden_dim, device=self._device,
        )
        self.sea_cell_state = torch.zeros(
            self.num_joints, num_layers, self._num_envs, hidden_dim, device=self._device,
        )

        # ----- encoder bias -----
        if isinstance(cfg.encoder_bias, (list, tuple)) and len(cfg.encoder_bias) != self.num_joints:
            raise ValueError(
                f"encoder_bias must have {self.num_joints} elements, got {len(cfg.encoder_bias)}."
            )
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)

        # ----- per-env action delay applied to the LSTM-output torque -----
        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

        self._torque_scale = float(cfg.torque_scale)

    def reset(self, env_ids: Sequence[int]):
        # Parent (DCMotor → IdealPDActuator → ActuatorBase) reset (currently no-op).
        super().reset(env_ids)
        self.torques_delay_buffer.reset(env_ids)
        self.sea_hidden_state[:, :, env_ids, :] = 0.0
        self.sea_cell_state[:, :, env_ids, :] = 0.0

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel

        # Position error in the encoder frame: target - (joint_pos - bias).
        pos_err = control_action.joint_positions - (joint_pos - self.encoder_bias)

        # Per-joint stateful LSTM inference. Input ``[pos_err, joint_vel]`` with seq_len=1.
        torques = torch.empty_like(pos_err)
        with torch.inference_mode():
            for j, net in enumerate(self.networks):
                x = torch.stack([pos_err[:, j], joint_vel[:, j]], dim=-1).unsqueeze(1)
                h_in = self.sea_hidden_state[j]
                c_in = self.sea_cell_state[j]
                out, (h_new, c_new) = net(x, (h_in, c_in))
                self.sea_hidden_state[j] = h_new
                self.sea_cell_state[j] = c_new
                torques[:, j] = out.reshape(out.shape[0])

        # Recover Nm from training-time scaled output, then apply action delay.
        torques = torques * self._torque_scale
        torques = self.torques_delay_buffer.compute(torques)

        self.computed_effort = torques
        self.applied_effort = torques  # no clipping — match actuator_net_comparison.py raw LSTM output

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
