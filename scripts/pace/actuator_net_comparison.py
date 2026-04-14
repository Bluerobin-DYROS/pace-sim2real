import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Actuator-net vs PD comparison for P73.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Pace-P73-Walker-v0")
parser.add_argument("--actuator_net_dir", type=str,
                    default="/home/bru24-server/IsaacSim/pace-sim2real/data/p73_lstm_",
                    help="Path prefix for actuator-net files: <prefix><joint_name>.pt")
parser.add_argument("--min_frequency", type=float, default=0.1)
parser.add_argument("--max_frequency", type=float, default=0.8)
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--torque_scale", type=float, default=100.0,
                    help="Multiply net output by this to recover Nm. Default 100.0 because "
                         "load_experiments() in training used torque_scaling=0.01, so "
                         "model predicts torque/100 and inference must multiply by 100.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything after sim launch ───────────────────────────────────────────────
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from torch import pi

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root

# ── joint mapping ─────────────────────────────────────────────────────────────
# 12 leg joints in the order used by the actuator-net .pt files.
# Index in this list == index used by JOINT_GROUPS in the training code.
LEG_JOINT_NAMES = [
    "L_HipRoll_Joint",    # 0  left_hip_roll
    "L_HipPitch_Joint",   # 1  left_hip_pitch
    "L_HipYaw_Joint",     # 2  left_hip_yaw
    "L_Knee_Joint",       # 3  left_knee_pitch
    "L_AnklePitch_Joint", # 4  left_ankle_pitch
    "L_AnkleRoll_Joint",  # 5  left_ankle_roll
    "R_HipRoll_Joint",    # 6  right_hip_roll
    "R_HipPitch_Joint",   # 7  right_hip_pitch
    "R_HipYaw_Joint",     # 8  right_hip_yaw
    "R_Knee_Joint",       # 9  right_knee_pitch
    "R_AnklePitch_Joint", # 10 right_ankle_pitch
    "R_AnkleRoll_Joint",  # 11 right_ankle_roll
]

NET_JOINT_NAMES = [
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
]

# PD gains from p73_walker.py (stiffness = Kp, damping = Kd)
KP = torch.tensor([
    1536.0, 937.5, 625.0, 747.552, 490.644, 490.104,   # left
    1536.0, 937.5, 625.0, 747.552, 490.644, 490.104,   # right
])
KD = torch.tensor([
    76.8, 37.5, 12.5, 37.378, 16.355, 5.337,           # left
    76.8, 37.5, 12.5, 37.378, 16.355, 5.337,           # right
])

# Chirp trajectory parameters (matching data_collection.py, 12-joint version)
TRAJ_DIRECTIONS = torch.tensor(
    [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)
TRAJ_BIAS  = torch.tensor([0.0] * 12)   # 12 values
TRAJ_SCALE = torch.tensor([0.1] * 12)  # 12 values
INIT_POS = torch.tensor([
    0.0,    # L_HipRoll
    0.18,   # L_HipPitch
    0.0,    # L_HipYaw
    0.35,   # L_Knee
   -0.17,   # L_AnklePitch
    0.0,    # L_AnkleRoll
    0.0,    # R_HipRoll
   -0.18,   # R_HipPitch
    0.0,    # R_HipYaw
   -0.35,   # R_Knee
    0.17,   # R_AnklePitch
    0.0,    # R_AnkleRoll
])

def load_actuator_nets(net_dir: str, device: str):
    """Load one TorchScript LSTM per joint. Returns list of models.

    Args:
        net_dir: Path prefix, e.g. "/path/to/p73_lstm_"
                 Files are expected at: <net_dir><joint_name>.pt
    """
    models = []
    for name in NET_JOINT_NAMES:
        path = os.path.join(net_dir, f"p73_lstm_{name}.pt") 
        if not os.path.exists(path):
            raise FileNotFoundError(f"Actuator-net not found: {path}")
        m = torch.jit.load(path, map_location=device)
        m.eval()
        models.append(m)
    print(f"[INFO]: Loaded {len(models)} actuator nets")
    return models


@torch.no_grad()
def infer_net_torques(models, pos_errors, vels, device):
    """Run one LSTM step for each joint and return predicted torques (scaled).
    Hidden state is reset to None every call
    """
    torques = torch.zeros(12, device=device)
    for j, model in enumerate(models):
        # Input: (batch=1, seq_len=1, features=2)
        x = torch.stack([pos_errors[j], vels[j]]).unsqueeze(0).unsqueeze(0)  # (1,1,2)
        pred, _ = model(x, None)  # reset state each step (matches training)
        torques[j] = pred[0, 0]
    return torques


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
        0.9600, 0.3750, 0.0625, 0.3563, 0.1289, 0.1288,  # left
        0.9600, 0.3750, 0.0625, 0.3563, 0.1289, 0.1288,  # right
        0.1600                                             # WaistYaw
    ], device=device).unsqueeze(0)

    damping = torch.tensor([
        6.9901, 6.9951, 1.7339, 6.8993, 3.1745, 6.9974,  # left  (viscous friction)
        2.6404, 6.9879, 4.5515, 6.9904, 6.2856, 6.9987,  # right
        6.5851                                             # WaistYaw
    ], device=device).unsqueeze(0)

    friction = torch.tensor([
        0.1735, 6.9898, 2.7360, 2.5399, 2.0902, 6.2475,  # left  (static/dynamic)
        1.9378, 4.6317, 0.0063, 1.6559, 1.0097, 6.3659,  # right
        0.9821                                             # WaistYaw
    ], device=device).unsqueeze(0)

    bias_full = torch.tensor([ 
        0.1000,  0.0084, -0.0833, -0.1000,  0.1000,  0.0902,  
        0.1000, -0.0717, -0.0944,  0.1000, -0.1000, -0.0742,
          -0.0106], device=device).unsqueeze(0)   

    time_lag = torch.tensor([[round(0.5598)]], dtype=torch.int, device=device)

    env.reset()
    print(f"  joint limits: {articulation.data.joint_pos_limits[0, joint_ids]}")

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature)))
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping)))
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    # modeling coulomb friction: joint_friction == joint_dynamic_friction
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

    # leg joint articulation indices (same order as joint_order[0:12])
    leg_art_ids = joint_ids[leg_local_ids]  # (12,) articulation DOF indices
    leg_bias = bias_full[0, leg_local_ids]

    # ── chirp trajectory (columns in articulation DOF order) ─────────────────
    dt = env.unwrapped.sim.get_physics_dt()
    sample_rate = 1.0 / dt
    num_steps = int(args_cli.duration * sample_rate)
    t = torch.linspace(0, args_cli.duration, steps=num_steps, device=device)

    phase = 2 * pi * (args_cli.min_frequency * t +
                      ((args_cli.max_frequency - args_cli.min_frequency) / (2 * args_cli.duration)) * t ** 2)
    chirp = torch.sin(phase)

    traj_dir   = TRAJ_DIRECTIONS.to(device)
    traj_bias  = TRAJ_BIAS.to(device)
    traj_scale = TRAJ_SCALE.to(device)

    # Compute modified chirp for the 12 leg joints (in joint_order / leg_art_ids order)
    init_pos = INIT_POS.to(device)
    leg_chirp = chirp.unsqueeze(-1).expand(-1, 12).clone()
    leg_chirp = init_pos.unsqueeze(0) + leg_chirp * traj_dir.unsqueeze(0) * traj_scale.unsqueeze(0)
    # Scatter into trajectory at correct articulation columns; WaistYaw stays 0
    N_dof = len(joint_ids)
    trajectory = torch.zeros((num_steps, N_dof), device=device)
    trajectory[:, leg_art_ids] = leg_chirp  # mirrors data_collection.py's trajectory[:, joint_ids]

    # ── load actuator nets ────────────────────────────────────────────────────
    models = load_actuator_nets(args_cli.actuator_net_dir, "cpu")

    kp = KP.to(device)
    kd = KD.to(device)

    # ── teleport to start pose ────────────────────────────────────────────────
    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0) + bias_full)
    articulation.write_joint_velocity_to_sim(torch.zeros((1, N_dof), device=device))

    # ── buffers ───────────────────────────────────────────────────────────────
    dof_pos_buf        = torch.zeros((num_steps, N_dof), device=device)  # all joints, bias subtracted
    dof_target_pos_buf = torch.zeros((num_steps, N_dof), device=device)  # all joints
    pd_torque_buf      = torch.zeros((num_steps, 12), device=device)
    net_torque_buf     = torch.zeros((num_steps, 12), device=device)
    pos_error_buf      = torch.zeros((num_steps, 12), device=device)
    vel_buf            = torch.zeros((num_steps, 12), device=device)
    actual_torque_buf   = torch.zeros((num_steps, 12), device=device)


    # ── simulation loop ───────────────────────────────────────────────────────
    counter = 0
    for step in range(num_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():

            # Record dof_pos (all joints, encoder-bias subtracted) – before step
            dof_pos_buf[step] = articulation.data.joint_pos[0, joint_ids] - bias_full[0]

            # Leg joint state for PD / net comparison
            q_leg  = articulation.data.joint_pos[0, leg_art_ids]   # (12,) in joint_order
            dq_leg = articulation.data.joint_vel[0, leg_art_ids]  # (12,)

            tgt_leg = leg_chirp[step]  # (12,) modified chirp targets in joint_order
            err_leg = tgt_leg - q_leg + leg_bias        # (12,) position error

            # PD torque (analytical)
            pd_tau = kp * err_leg - kd * dq_leg        # (12,) Nm

            # Actuator net torque (state reset each step, runs on CPU)
            net_tau_cpu = infer_net_torques(models, err_leg.cpu(), dq_leg.cpu(), "cpu")
            net_tau = net_tau_cpu.to(device) * args_cli.torque_scale  # to Nm

            # Record
            pd_torque_buf[step]  = pd_tau
            net_torque_buf[step] = net_tau
            pos_error_buf[step]  = err_leg
            vel_buf[step]        = dq_leg

            # Step with full trajectory action
            actions = trajectory[step, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            env.step(actions)
            actual_torque_buf[step] = articulation.data.applied_torque[0,leg_art_ids]

            # Record joint position target – after step
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
        ax.plot(t_np, pd_np[:, j],  label="PD (sim)",     color="tab:blue",   linewidth=1.2)
        ax.plot(t_np, net_np[:, j], label="Actuator net", color="tab:orange", linewidth=0.8, alpha=0.85)
        ax.plot(t_np, actual_np[:, j], label="Actual torque", color="tab:green", linewidth=0.8, alpha=0.85)
        ax.set_title(LEG_JOINT_NAMES[j], fontsize=9)
        ax.set_ylabel("Torque [Nm]", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Time [s]", fontsize=8)
    axes[-1].set_xlabel("Time [s]", fontsize=8)
    fig.suptitle("PD torque (sim) vs LSTM actuator-net torque — P73 chirp", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "torque_comparison.png", dpi=120)
    plt.close()
    print(f"[INFO]: Saved plot to {out_dir / 'torque_comparison.png'}")

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
        # ax.plot(t_np, dof_pos_raw[:, li], label="Acㄷtual with bias", color="tab:blue", linewidth=0.8, alpha=0.85)
        ax.set_title(LEG_JOINT_NAMES[j], fontsize=9)
        ax.set_ylabel("Position [rad]", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes2[-2].set_xlabel("Time [s]", fontsize=8)
    axes2[-1].set_xlabel("Time [s]", fontsize=8)
    fig2.suptitle("PD position tracking — P73 chirp", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "position_tracking.png", dpi=120)
    plt.close()
    print(f"[INFO]: Saved plot to {out_dir / 'position_tracking.png'}")

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
    plt.savefig(out_dir / "position_error.png", dpi=120)
    plt.close()
    print(f"[INFO]: Saved plot to {out_dir / 'position_error.png'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
