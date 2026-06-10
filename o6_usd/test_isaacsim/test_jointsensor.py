import omni.physx
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import _sensor
import numpy as np
import math
import asyncio  # [New] Import asyncio for non-blocking action delays

# 1. Confirm the robot's root prim path
ROBOT_PRIM_PATH = "/World/linker_o6_right_v1_0_urdf" 

# 2. Confirm the fingertip contact sensor's path
FINGERTIP_SENSORS = {
    "Index Finger": f"{ROBOT_PRIM_PATH}/rh_index_distal/collisions/Contact_Sensor"
}

# Bind robot and sensor interfaces
hand = Articulation(ROBOT_PRIM_PATH)
hand.initialize()
cs_interface = _sensor.acquire_contact_sensor_interface()

# Helper function: Used specifically to send joint position commands
def move_joints(joint_names, angles_deg):
    indices, positions = [], []
    for joint_name, angle in zip(joint_names, angles_deg):
        idx = hand.get_dof_index(joint_name)
        if idx is not None:
            indices.append(idx)
            positions.append(math.radians(angle))
            
    if indices:
        action = ArticulationAction(joint_positions=np.array(positions), joint_indices=np.array(indices))
        hand.apply_action(action)

# [Core Modification] Asynchronous sequential grasp function
async def grasp_sequence():
    print("\n[Action 1] Thumb yaw (Pre-grasp) ...")
    # Step 1: Only yaw the thumb (100 degrees)
    move_joints(["rh_thumb_cmc_yaw"], [100])
    
    # Step 2: Wait 1.5 seconds (Note: NEVER use time.sleep, it will freeze the physics engine)
    await asyncio.sleep(1.5)
    
    print("\n[Action 2] Other fingers start closing to grasp! ...")
    # Step 3: Close the 4 fingers (50 degrees) and pitch the thumb base (10 degrees)
    move_joints(
        ["rh_index_mcp_pitch", "rh_middle_mcp_pitch", "rh_ring_mcp_pitch", "rh_pinky_mcp_pitch", "rh_thumb_cmc_pitch"], 
        [50, 50, 50, 50, 10]
    )

# Physics callback function (Monitors tactile data at 60Hz)
global my_sensor_sub 
def physics_step_callback(step_size):
    for finger_name, sensor_path in FINGERTIP_SENSORS.items():
        reading = cs_interface.get_sensor_reading(sensor_path)
        if reading.is_valid and reading.in_contact:
            force_magnitude = reading.value
            if force_magnitude > 0.01:
                print(f"[Tactile Feedback] {finger_name} feels a force of: {force_magnitude:.3f} N")

# Clean up old callbacks to prevent duplicate subscriptions if run multiple times
try:
    if my_sensor_sub:
        my_sensor_sub = None
except NameError:
    pass

# Subscribe to physics step events, turning on the "sensory nervous system"
my_sensor_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(physics_step_callback)

# Execute the asynchronous grasp sequence
asyncio.ensure_future(grasp_sequence())
