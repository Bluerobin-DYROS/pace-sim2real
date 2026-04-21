# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg
from pace_sim2real.assets import P73_CFG
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

@configclass
class P73PaceCfg(PaceCfg):
    """Pace configuration for the P73 walker robot."""
    robot_name: str = "p73_walker"
    data_dir: str = "p73_real/chirp_data.pt"
    bounds_params: torch.Tensor = torch.zeros((53, 2))  # 13 + 13 + 13 + 13 + 1 = 53 parameters to optimize
    armature_fixed: torch.Tensor = torch.tensor(
        [
            0.96,
            0.375,
            0.0625,
            0.35630,
            0.12886,
            0.12883,
            0.96,
            0.375,
            0.0625,
            0.35630,
            0.12886,
            0.12883,
            0.16,
        ]
    )
    joint_order: list[str] = [
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
        "WaistYaw_Joint",
    ]

    def __post_init__(self):
        # set bounds for parameters
        
        DOF_DIM = 13  
        # armature fixed values (not optimized)
        self.bounds_params[:DOF_DIM, 0] = self.armature_fixed
        self.bounds_params[:DOF_DIM, 1] = self.armature_fixed

        # dof damping
        self.bounds_params[DOF_DIM:2*DOF_DIM, 1] = 15.0  # dof_damping between 0.0 - 15.0 [Nm s/rad]
        
        # dof friction
        self.bounds_params[2*DOF_DIM:3*DOF_DIM, 1] = 15.0  # friction between 0.0 - 15.0 [Nm]
        
        # dof bias
        self.bounds_params[3*DOF_DIM:4*DOF_DIM, 0] = -0.1
        self.bounds_params[3*DOF_DIM:4*DOF_DIM, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        
        # dof delay
        self.bounds_params[4*DOF_DIM, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class P73PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for the P73 walker robot in the Pace Sim2Real environment."""
    robot: ArticulationCfg = P73_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
        ),
    )


@configclass
class P73PaceEnvCfg(PaceSim2realEnvCfg):

    scene: P73PaceSceneCfg = P73PaceSceneCfg()
    sim2real: PaceCfg = P73PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.001  # 1000Hz simulation
        self.decimation = 1  # 1000Hz control
