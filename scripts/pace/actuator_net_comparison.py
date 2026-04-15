import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Actuator-net vs PD comparison for P73.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Pace-P73-Walker-v0")
parser.add_argument("--actuator_net_dir", type=str,
                    default="/home/bru24-server/IsaacSim/pace-sim2real/data/p73_lstm_",
                    help="Directory containing per-joint .pt files")
parser.add_argument("--min_frequency", type=float, default=0.1)
parser.add_argument("--max_frequency", type=float, default=0.5)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--torque_scale", type=float, default=100.0,
                    help="Multiply net output by this to recover Nm. Default 100.0 because "
                         "load_experiments() in training used torque_scaling=0.01, so "
                         "model predicts torque/100 and inference must multiply by 100.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from torch import pi

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root

from isaaclab.actuators import ActuatorNetLSTM, ActuatorNetLSTMCfg
from isaaclab.utils.types import ArticulationActions

# ── joint mapping ─────────────────────────────────────────────────────────────
LEG_JOINT_NAMES = [
    "L_HipRoll_Joint",    # 0
    "L_HipPitch_Joint",   # 1
    "L_HipYaw_Joint",     # 2
    "L_Knee_Joint",       # 3
    "L_AnklePitch_Joint", # 4
    "L_AnkleRoll_Joint",  # 5
    "R_HipRoll_Joint",    # 6
    "R_HipPitch_Joint",   # 7
    "R_HipYaw_Joint",     # 8
    "R_Knee_Joint",       # 9
    "R_AnklePitch_Joint", # 10
    "R_AnkleRoll_Joint",  # 11
]

NET_JOINT_NAMES = [
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
]

# PD gains
KP = torch.tensor([
    1536.0, 937.5, 625.0, 747.552, 490.644, 490.104,
    1536.0, 937.5, 625.0, 747.552, 490.644, 490.104,
])
KD = torch.tensor([
    76.8, 37.5, 12.5, 37.378, 16.355, 5.337,
    76.8, 37.5, 12.5, 37.378, 16.355, 5.337,
])

# Chirp trajectory parameters
TRAJ_DIRECTIONS = torch.tensor(
    [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)
TRAJ_BIAS  = torch.tensor([0.0] * 12)
TRAJ_SCALE = torch.tensor([0.1] * 12)
INIT_POS = torch.tensor([
    0.0, 0.18, 0.0, 0.35, -0.17, 0.0,
    0.0, -0.18, 0.0, -0.35, 0.17, 0.0,
])


# ── Isaac Lab ActuatorNetLSTM wrapper (per-joint, stateless) ──────────────────

class PerJointActuatorNetLSTM:
    def __init__(self, net_dir: str, num_envs: int, device: str, torque_scale: float):
        self.num_joints = 12
        self.num_envs = num_envs
        self.device = device
        self.torque_scale = torque_scale
        self.actuators: list[ActuatorNetLSTM] = []

        for i, net_name in enumerate(NET_JOINT_NAMES):
            pt_path = os.path.join(net_dir, f"p73_lstm_{net_name}.pt")
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Actuator net not found: {pt_path}")

            cfg = ActuatorNetLSTMCfg(
                joint_names_expr=[LEG_JOINT_NAMES[i]],
                network_file=pt_path,
                saturation_effort=1e9,  # large value → no clipping here
                effort_limit=1e9,
                velocity_limit=1e9,
            )


            act = ActuatorNetLSTM(
                cfg=cfg,
                joint_names=[LEG_JOINT_NAMES[i]],
                joint_ids=torch.tensor([i], device=device),  # local index
                num_envs=num_envs,
                device=device,
                stiffness=torch.zeros(num_envs, 1, device=device),
                damping=torch.zeros(num_envs, 1, device=device),
                armature=torch.zeros(num_envs, 1, device=device),
                friction=torch.zeros(num_envs, 1, device=device),
                effort_limit=torch.tensor([[1e9]], device=device).expand(num_envs, -1),
                velocity_limit=torch.tensor([[1e9]], device=device).expand(num_envs, -1),
            )
            self.actuators.append(act)

        print(f"[INFO]: Loaded {len(self.actuators)} Isaac Lab ActuatorNetLSTM instances")

    def reset_all(self):
        """Reset hidden/cell states for all joints and all envs."""
        env_ids = list(range(self.num_envs))
        for act in self.actuators:
            act.reset(env_ids)

    @torch.no_grad()
    def compute_torques(
        self,
        target_pos: torch.Tensor,   # (num_envs, 12) desired joint positions
        joint_pos: torch.Tensor,     # (num_envs, 12) current joint positions
        joint_vel: torch.Tensor,     # (num_envs, 12) current joint velocities
    ) -> torch.Tensor:
        # Reset hidden state every step → stateless inference
        self.reset_all()

        torques = torch.zeros(self.num_envs, self.num_joints, device=self.device)

        for j, act in enumerate(self.actuators):
            # Slice single joint: (num_envs, 1)
            tgt_j = target_pos[:, j:j+1]
            pos_j = joint_pos[:, j:j+1]
            vel_j = joint_vel[:, j:j+1]

            # Build ArticulationActions (Isaac Lab's input container)
            control_action = ArticulationActions(
                joint_positions=tgt_j,
                joint_velocities=None,
                joint_efforts=None,
            )

            # ActuatorNetLSTM.compute() returns modified ArticulationActions
            result = act.compute(control_action, pos_j, vel_j)
            torques[:, j] = result.joint_efforts.squeeze(-1)

        return torques * self.torque_scale


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )

    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 3.0)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    articulation = env.unwrapped.scene["robot"]

    joint_order = env_cfg.sim2real.joint_order  # 13 joints (includes WaistYaw)
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order], device=device
    )
    leg_local_ids = torch.tensor(
        [i for i, name in enumerate(joint_order) if name in LEG_JOINT_NAMES],
        device=device,
    )

    # ── physical parameter setup ───────────────────────────────────────────
    armature = torch.tensor([
        0.9600, 0.3750, 0.0625, 0.3563, 0.1289, 0.1288,
        0.9600, 0.3750, 0.0625, 0.3563, 0.1289, 0.1288,
        0.1600
    ], device=device).unsqueeze(0)

    damping = torch.tensor([
        6.9901, 6.9951, 1.7339, 6.8993, 3.1745, 6.9974,
        2.6404, 6.9879, 4.5515, 6.9904, 6.2856, 6.9987,
        6.5851
    ], device=device).unsqueeze(0)

    friction = torch.tensor([
        0.1735, 6.9898, 2.7360, 2.5399, 2.0902, 6.2475,
        1.9378, 4.6317, 0.0063, 1.6559, 1.0097, 6.3659,
        0.9821
    ], device=device).unsqueeze(0)

    bias_full = torch.tensor([
        0.1000,  0.0084, -0.0833, -0.1000,  0.1000,  0.0902,
        0.1000, -0.0717, -0.0944,  0.1000, -0.1000, -0.0742,
        -0.0106
    ], device=device).unsqueeze(0)

    time_lag = torch.tensor([[round(0.5598)]], dtype=torch.int, device=device)

    env.reset()
    print(f"  joint limits: {articulation.data.joint_pos_limits[0, joint_ids]}")

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature)))
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping)))
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = friction

    drive_types = articulation.actuators.keys()
    for drive_type in drive_types:
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias_full[:, drive_joint_idx])
        articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs))

    leg_art_ids = joint_ids[leg_local_ids]
    leg_bias = bias_full[0, leg_local_ids]

    # ── chirp trajectory ─────────────────────────────────────────────────────
    dt = env.unwrapped.sim.get_physics_dt()
    sample_rate = 1.0 / dt
    num_steps = int(args_cli.duration * sample_rate)
    t = torch.linspace(0, args_cli.duration, steps=num_steps, device=device)

    phase = 2 * pi * (args_cli.min_frequency * t +
                      ((args_cli.max_frequency - args_cli.min_frequency) / (2 * args_cli.duration)) * t ** 2)
    chirp = torch.sin(phase)

    traj_dir   = TRAJ_DIRECTIONS.to(device)
    traj_scale = TRAJ_SCALE.to(device)
    init_pos   = INIT_POS.to(device)

    leg_chirp = chirp.unsqueeze(-1).expand(-1, 12).clone()
    leg_chirp = init_pos.unsqueeze(0) + leg_chirp * traj_dir.unsqueeze(0) * traj_scale.unsqueeze(0)

    N_dof = len(joint_ids)
    trajectory = torch.zeros((num_steps, N_dof), device=device)
    trajectory[:, leg_art_ids] = leg_chirp

    # ── load actuator nets using Isaac Lab's ActuatorNetLSTM ──────────────────
    per_joint_net = PerJointActuatorNetLSTM(
        net_dir=args_cli.actuator_net_dir,
        num_envs=args_cli.num_envs,
        device=str(device),
        torque_scale=args_cli.torque_scale,
    )

    kp = KP.to(device)
    kd = KD.to(device)

    # ── teleport to start pose ────────────────────────────────────────────────
    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0) + bias_full)
    articulation.write_joint_velocity_to_sim(torch.zeros((1, N_dof), device=device))

    # ── buffers ───────────────────────────────────────────────────────────────
    dof_pos_buf        = torch.zeros((num_steps, N_dof), device=device)
    dof_target_pos_buf = torch.zeros((num_steps, N_dof), device=device)
    pd_torque_buf      = torch.zeros((num_steps, 12), device=device)
    net_torque_buf     = torch.zeros((num_steps, 12), device=device)
    pos_error_buf      = torch.zeros((num_steps, 12), device=device)
    vel_buf            = torch.zeros((num_steps, 12), device=device)
    actual_torque_buf  = torch.zeros((num_steps, 12), device=device)

    # ── simulation loop ───────────────────────────────────────────────────────
    for step in range(num_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():

            dof_pos_buf[step] = articulation.data.joint_pos[0, joint_ids] - bias_full[0]

            q_leg  = articulation.data.joint_pos[0, leg_art_ids]
            dq_leg = articulation.data.joint_vel[0, leg_art_ids]

            tgt_leg = leg_chirp[step]
            err_leg = tgt_leg - q_leg + leg_bias

            # PD torque (analytical)
            pd_tau = kp * err_leg - kd * dq_leg

            # Actuator net torque via Isaac Lab ActuatorNetLSTM (stateless)
            # Need (num_envs, 12) shape
            tgt_batch = tgt_leg.unsqueeze(0).expand(args_cli.num_envs, -1)
            pos_batch = (q_leg - leg_bias).unsqueeze(0).expand(args_cli.num_envs, -1)
            vel_batch = dq_leg.unsqueeze(0).expand(args_cli.num_envs, -1)

            net_tau = per_joint_net.compute_torques(tgt_batch, pos_batch, vel_batch)
            net_tau = net_tau[0]  # take env 0

            # Record
            pd_torque_buf[step]  = pd_tau
            net_torque_buf[step] = net_tau
            pos_error_buf[step]  = err_leg
            vel_buf[step]        = dq_leg

            # Step
            actions = trajectory[step, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            env.step(actions)
            actual_torque_buf[step] = articulation.data.applied_torque[0, leg_art_ids]

            dof_target_pos_buf[step] = articulation._data.joint_pos_target[0, joint_ids]

            if step < 5:
                act = articulation.actuators['walker_motors']
                print(f"computed: {act.computed_effort[0, :6]}")
                print(f"applied:  {act.applied_effort[0, :6]}")
                print(f"vel:      {dq_leg[:6]}")
                print(f"target:   {articulation._data.joint_pos_target[0, leg_art_ids[:6]]}")
                print(f"actual:   {q_leg[:6]}")

            if (step+1) % int(sample_rate) == 0:
                print(f"[INFO]: {step / sample_rate:.1f} / {args_cli.duration:.1f} s")

    env.close()

    from time import sleep
    sleep(1)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = project_root() / "data" / env_cfg.sim2real.robot_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "time":        t.cpu(),
        "dof_pos":     dof_pos_buf.cpu(),
        "des_dof_pos": dof_target_pos_buf.cpu(),
        "pos_error":   pos_error_buf.cpu(),
        "velocity":    vel_buf.cpu(),
        "pd_torque":   pd_torque_buf.cpu(),
        "net_torque":  net_torque_buf.cpu(),
        "joint_names": LEG_JOINT_NAMES,
        "actual_torque": actual_torque_buf.cpu(),
    }, out_dir / "chirp_data.pt")
    print(f"[INFO]: Saved data to {out_dir / 'chirp_data.pt'}")

    # ── plot: PD torque vs actuator-net torque ────────────────────────────────
    t_np   = t.cpu().numpy()
    pd_np  = pd_torque_buf.cpu().numpy()
    net_np = net_torque_buf.cpu().numpy()
    actual_np = actual_torque_buf.cpu().numpy()
    err_np = pos_error_buf.cpu().numpy()

    fig, axes = plt.subplots(6, 2, figsize=(16, 21))
    axes = axes.flatten()
    for j in range(12):
        ax = axes[j]
        ax.plot(t_np, pd_np[:, j],     label="PD (sim)",        color="tab:blue",   linewidth=1.2)
        ax.plot(t_np, net_np[:, j],    label="ActuatorNetLSTM", color="tab:orange", linewidth=0.8, alpha=0.85)
        ax.plot(t_np, actual_np[:, j], label="Actual torque",   color="tab:green",  linewidth=0.8, alpha=0.85)
        ax.set_title(LEG_JOINT_NAMES[j], fontsize=9)
        ax.set_ylabel("Torque [Nm]", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Time [s]", fontsize=8)
    axes[-1].set_xlabel("Time [s]", fontsize=8)
    fig.suptitle("PD torque vs Isaac Lab ActuatorNetLSTM (per-joint, stateless) — P73 chirp", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "torque_comparison_isaaclab.png", dpi=120)
    plt.close()
    print(f"[INFO]: Saved plot to {out_dir / 'torque_comparison_isaaclab.png'}")

    # ── plot: position tracking ───────────────────────────────────────────────
    dof_pos_np = dof_pos_buf.cpu().numpy()
    des_pos_np = dof_target_pos_buf.cpu().numpy()
    leg_ids_np = leg_local_ids.cpu().numpy()

    fig2, axes2 = plt.subplots(6, 2, figsize=(16, 21))
    axes2 = axes2.flatten()
    for j, li in enumerate(leg_ids_np):
        ax = axes2[j]
        ax.plot(t_np, des_pos_np[:, li], label="Target", color="grey",      linewidth=1.0, linestyle="--", alpha=0.7)
        ax.plot(t_np, dof_pos_np[:, li], label="Actual", color="tab:green", linewidth=1.2)
        ax.set_title(LEG_JOINT_NAMES[j], fontsize=9)
        ax.set_ylabel("Position [rad]", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes2[-2].set_xlabel("Time [s]", fontsize=8)
    axes2[-1].set_xlabel("Time [s]", fontsize=8)
    fig2.suptitle("Position tracking — P73 chirp", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "position_tracking_isaaclab.png", dpi=120)
    plt.close()
    print(f"[INFO]: Saved plot to {out_dir / 'position_tracking_isaaclab.png'}")

    # ── plot: position error ──────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(6, 2, figsize=(16, 21))
    axes3 = axes3.flatten()
    for j in range(12):
        ax = axes3[j]
        ax.plot(t_np, err_np[:, j], label="Position error", color="tab:red", linewidth=1.0)
        ax.set_title(LEG_JOINT_NAMES[j], fontsize=9)
        ax.set_ylabel("Position error [rad]", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes3[-2].set_xlabel("Time [s]", fontsize=8)
    axes3[-1].set_xlabel("Time [s]", fontsize=8)
    fig3.suptitle("Position error — P73 chirp", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "position_error_isaaclab.png", dpi=120)
    plt.close()
    print(f"[INFO]: Saved plot to {out_dir / 'position_error_isaaclab.png'}")


if __name__ == "__main__":
    main()
    simulation_app.close()

