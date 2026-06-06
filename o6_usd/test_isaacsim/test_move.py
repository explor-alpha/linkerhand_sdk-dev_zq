from omni.isaac.core.articulations import Articulation
# [New] Imported ArticulationAction for better version compatibility
from omni.isaac.core.utils.types import ArticulationAction 
import numpy as np
import math

# 1. Your robot's root prim path
ROBOT_PRIM_PATH = "/World/linker_o6_right_v1_0_urdf"

# 2. Bind the articulation interface
hand = Articulation(ROBOT_PRIM_PATH)
hand.initialize() 

# 3. Core function: Set joint angle
def set_joint_angle(joint_name, angle_degrees):
    """
    Set a specific joint to a target angle in degrees.
    joint_name: The name of the joint (e.g., "index_mcp_pitch")
    angle_degrees: The target angle in degrees (e.g., 45)
    """
    idx = hand.get_dof_index(joint_name)
    if idx is None:
        print(f"[Error] Cannot find joint named '{joint_name}'. Please check the spelling.")
        return
        
    # Convert degrees to radians (Isaac Sim physics engine uses radians)
    angle_rad = math.radians(angle_degrees)
    
    # [Fix] Use ArticulationAction which works on all Isaac Sim versions
    action = ArticulationAction(
        joint_positions=np.array([angle_rad]),
        joint_indices=np.array([idx])
    )
    # Apply the action to the robot
    hand.apply_action(action)
    
    print(f"[Success] Target for joint [{joint_name}] set to {angle_degrees} degrees.")

# ==========================================

# 4. Call the function here to test any joint you want!

O6_CONTROL_JOINTS=[    
    "rh_thumb_cmc_yaw", "rh_thumb_cmc_pitch",
    "rh_index_mcp_pitch", "rh_middle_mcp_pitch", 
    "rh_ring_mcp_pitch", "rh_pinky_mcp_pitch",
    ]

set_joint_angle("rh_thumb_cmc_yaw", 30)
set_joint_angle("rh_thumb_cmc_pitch", 30)
set_joint_angle("rh_index_mcp_pitch", 30)
set_joint_angle("rh_middle_mcp_pitch", 30)
set_joint_angle("rh_ring_mcp_pitch", 60)
set_joint_angle("rh_pinky_mcp_pitch", 45)



