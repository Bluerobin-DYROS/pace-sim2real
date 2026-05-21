import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from pace_sim2real.utils import PaceActuatorNetLSTMCfg, PaceDCMotorCfg, project_root

import os

P73_ASSETS_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
P73_LSTM_DIR = str(project_root() / "data" / "p73_lstm_")

LEG_JOINT_NAMES = [
    "L_HipRoll_Joint",
    "L_HipPitch_Joint",
    "L_HipYaw_Joint",
    "L_Knee_Joint",
    "L_AnklePitch_Joint",
    "L_AnkleRoll_Joint",
    "R_HipRoll_Joint",
    "R_HipPitch_Joint",
    "R_HipYaw_Joint",
    "R_Knee_Joint",
    "R_AnklePitch_Joint",
    "R_AnkleRoll_Joint",
]

# File-name tokens for each leg joint's per-joint LSTM .pt (paired with LEG_JOINT_NAMES).
LEG_NET_JOINT_NAMES = [
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
]

init = {
    # "L_HipRoll_Joint": 0.0,
    # "L_HipPitch_Joint": 0.18,
    # "L_HipYaw_Joint": 0.0,
    # "L_Knee_Joint": 0.35,
    # "L_AnklePitch_Joint": -0.17,
    # "L_AnkleRoll_Joint": 0.0,

    # "R_HipRoll_Joint": 0.0,
    # "R_HipPitch_Joint": -0.18,
    # "R_HipYaw_Joint": 0.0,
    # "R_Knee_Joint": -0.35,
    # "R_AnklePitch_Joint": 0.17,
    # "R_AnkleRoll_Joint": 0.0,

    # "WaistYaw_Joint": 0.0,

    "L_HipRoll_Joint": 0.103292,
    "L_HipPitch_Joint": 0.178421,
    "L_HipYaw_Joint": -0.000215716,
    "L_Knee_Joint": 0.346192,
    "L_AnklePitch_Joint": -0.170041,
    "L_AnkleRoll_Joint": 0.00125072,

    "R_HipRoll_Joint": 0.10316,
    "R_HipPitch_Joint": -0.177822,
    "R_HipYaw_Joint": -0.000119842,
    "R_Knee_Joint": -0.345918,
    "R_AnklePitch_Joint": 0.169777,
    "R_AnkleRoll_Joint": -0.00192714,

    "WaistYaw_Joint": 0.00100667,
}


P73_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{P73_ASSETS_DATA_DIR}/p73_walker_description/urdf/p73_walker.urdf",
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.895),
        joint_pos=init,
    ),
    soft_joint_pos_limit_factor=100.0,
    actuators={
        "leg_motors": PaceActuatorNetLSTMCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            net_joint_names=LEG_NET_JOINT_NAMES,
            network_dir=P73_LSTM_DIR,
            network_prefix="p73_lstm_",
            torque_scale=100.0,
            effort_limit={
                ".*_HipRoll_Joint": 352.0,
                ".*_HipPitch_Joint": 220.0,
                ".*_HipYaw_Joint": 95.0,
                ".*_Knee_Joint": 220.0,
                ".*_AnklePitch_Joint": 95.0,
                ".*_AnkleRoll_Joint": 95.0,
            },
            velocity_limit={
                ".*_HipRoll_Joint": 4.86,
                ".*_HipPitch_Joint": 7.78,
                ".*_HipYaw_Joint": 11.81,
                ".*_Knee_Joint": 7.78,
                ".*_AnklePitch_Joint": 11.81,
                ".*_AnkleRoll_Joint": 11.81,
            },
            encoder_bias={".*": 0.0},
            max_delay=10,
        ),
        "waist_motor": PaceDCMotorCfg(
            joint_names_expr=["WaistYaw_Joint"],
            saturation_effort=152.0,
            effort_limit={"WaistYaw_Joint": 152.0},
            velocity_limit={"WaistYaw_Joint": 4.03},
            stiffness={"WaistYaw_Joint": 576.0},
            damping={"WaistYaw_Joint": 19.2},
            encoder_bias={"WaistYaw_Joint": 0.0},
            max_delay=10,
        ),
    },
)
