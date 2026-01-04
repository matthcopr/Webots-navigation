from controller import Supervisor
import math
import numpy as np
from heapq import heappush, heappop
from collections import defaultdict
from os.path import exists
import os

# Clamp a value into a closed interval
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Wrap an angle into [-pi, pi] to keep heading errors stable
def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

# Initialize the Webots supervisor and simulation timing
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Configure wheel motors for velocity control
motor_left = robot.getDevice('wheel_left_joint')
motor_right = robot.getDevice('wheel_right_joint')
motor_left.setPosition(float('inf'))
motor_right.setPosition(float('inf'))

# Enable camera and object recognition for jar detection
camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass for localization and yaw estimation
compass = robot.getDevice('compass')
compass.enable(timestep)
gps = robot.getDevice('gps')
gps.enable(timestep)

# Load arm torso head gripper motors if present on this robot model
MOTOR_NAMES = [
    'torso_lift_joint',
    'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint',
    'head_1_joint', 'head_2_joint',
    'gripper_left_finger_joint', 'gripper_right_finger_joint',
]
motor_handles = {}
for name in MOTOR_NAMES:
    try:
        motor_handles[name] = robot.getDevice(name)
    except:
        pass

# Enable finger force feedback for contact based grasp confirmation
for j in ['gripper_left_finger_joint', 'gripper_right_finger_joint']:
    if j in motor_handles:
        try:
            motor_handles[j].enableForceFeedback(timestep)
        except:
            pass

# Enable joint encoders used for transforms and torso control
encoders = {}
for joint in ['torso_lift_joint', 'head_1_joint', 'head_2_joint']:
    sensor_name = joint + "_sensor"
    try:
        s = robot.getDevice(sensor_name)
        s.enable(timestep)
        encoders[joint] = s
    except:
        pass

# Robot wheel limits and global speed parameters
WHEEL_MAX = 10.1523

# Navigation gains and speed limits for waypoint following
NAV_P_FWD = 1.4
NAV_P_TURN = 3.0
NAV_MAX_SPEED = 5.5

# Reduce forward speed when heading error is large
NAV_ALPHA_SLOWDOWN_START = 0.35
NAV_ALPHA_SLOWDOWN_FULL = 1.00
NAV_MIN_FWD_SCALE = 0.08

# Path following tolerance and counter waypoint acceptance radius
PATH_POINT_REACHED = 0.22
GOAL_REACHED_COUNTER = 0.80

# Search spin cadence while scanning for jars
SEARCH_SPIN_SPEED = 0.45
SEARCH_PAUSE_EVERY = 18
SEARCH_PAUSE_STEPS = 6

# World frame standoff distance used before starting the arm grasp sequence
TARGET_DIST = 1.45
DIST_EPS = 0.06
SLOWDOWN_RADIUS = 0.75

# Back away behavior if the robot gets too close to the jar during approach
TOO_CLOSE_EPS = 0.10
BACKUP_GAIN = 2.8
BACKUP_MAX_SCALE = 0.65

# Camera visual servo configuration for horizontal centering
CAM_HORIZ_AXIS = 'y'
CAM_HORIZ_SIGN = -1.0

# Camera centering thresholds and stability holds
CAM_CENTER_EPS = 0.06
CAM_CENTER_HOLD_TICKS = 6

# Lock loss thresholds and reacquire grace window
CAM_LOST_MAX = 90
CAM_REACQUIRE_GRACE = 35

# Visual approach controller gains
APPROACH_KP_TURN = 2.0
APPROACH_KP_FWD = 2.2
APPROACH_MAX_FWD = 2.8
APPROACH_MAX_TURN = 1.6

# Visual correction gains during slow creep grasp
CREEP_VISUAL_KP = 1.4
CREEP_VISUAL_TURN_MAX = 0.50

# Arm poses for travel picking and post grasp transport
ARM_SAFE_POSE = {
    'arm_1_joint': 0.71,
    'arm_2_joint': 1.02,
    'arm_3_joint': -2.815,
    'arm_4_joint': 1.011,
    'arm_5_joint': 0.0,
    'arm_6_joint': 0.0,
    'arm_7_joint': 0.0,
    'head_1_joint': 0.0,
    'head_2_joint': 0.0,
}

ARM_PICK_POSE = {
    'arm_1_joint': 1.57,
    'arm_2_joint': 0.95,
    'arm_3_joint': 0.0,
    'arm_4_joint': 1.0,
    'arm_5_joint': 0.00,
    'arm_6_joint': 0.00,
    'arm_7_joint': -1.57,
}

ARM_POST_CLOSE_POSE = {
    'arm_1_joint': 1.57,
    'arm_2_joint': 0.95,
    'arm_3_joint': 0.0,
    'arm_4_joint': -0.32,
    'arm_5_joint': 0.00,
    'arm_6_joint': 0.00,
    'arm_7_joint': -1.57,
}

# Torso travel limits and a heuristic offset to align gripper height to jar height
TORSO_MIN = 0.00
TORSO_MAX = 0.70
GRIPPER_Z_OFFSET = 0.95

# Gripper open close targets
GRIPPER_OPEN_GRASP = 0.045
GRIPPER_CLOSED = 0.0

# Compute a safe maximum gripper opening based on joint limits
def _get_gripper_max_open():
    mx = GRIPPER_OPEN_GRASP
    for j in ['gripper_left_finger_joint', 'gripper_right_finger_joint']:
        if j in motor_handles:
            try:
                mx = min(mx, float(motor_handles[j].getMaxPosition()))
            except:
                pass
    return mx

GRIPPER_MAX_OPEN = _get_gripper_max_open()

# Creep grasp parameters used to gently drive into the jar and detect bilateral contact
CREEP_FWD_SPEED = 2.0
CREEP_HEADING_KP = 3.0
CREEP_TURN_MAX = 0.65

# Side contact latch behavior that biases turn away from a single sided collision
SIDE_DETECT_THRESHOLD = 0.20
SIDE_RELEASE_THRESHOLD = 0.12
SIDE_LATCH_TURN = 0.42
SIDE_LATCH_MAX_HOLD = 40

# Bilateral force threshold used to decide when to stop and close
SIDE_FORCE_THRESHOLD = 0.35
BOTH_CONTACT_HOLD_TICKS = 3
CREEP_LOG_EVERY = 40

# Distance based backup parameters used after grasping and after placing
BACKUP_AFTER_CLOSE_DIST = 0.5
BACKUP_AFTER_PLACE_DIST = 0.5

# Backup controller parameters for keeping a straight reverse line
BACKUP_AFTER_CLOSE_SPEED = 1.6
BACKUP_AFTER_CLOSE_YAW_KP = 2.2
BACKUP_AFTER_CLOSE_TURN_MAX = 0.55

# Table docking parameters to approach an exact placement point
TABLE_SWITCH_TO_DOCK = 0.55
TABLE_DOCK_EPS = 0.05
TABLE_DOCK_MAX_STEPS = 800

# Table heading target and yaw controller settings
TABLE_TARGET_YAW = wrap_pi(math.pi + (math.pi / 4.0))
TABLE_YAW_EPS = 0.05
TABLE_YAW_HOLD_TICKS = 8
TABLE_YAW_KP = 6.5

# Docking gains and maximum speed scaling when close to the table
P_DOCK_FWD = 3.0
P_DOCK_TURN = 2.6
DOCK_MAX_SPEED_SCALE = 0.55

# If heading error is large rotate in place rather than driving forward
DOCK_ALIGN_ONLY_ALPHA = 0.15

# Placement sequence parameters to make release more reliable
PLACE_NUDGE_FWD_DIST = 0.12
PLACE_NUDGE_FWD_SPEED = 0.9
PLACE_NUDGE_YAW_KP = 2.4
PLACE_NUDGE_TURN_MAX = 0.40

PLACE_TORSO_DOWN = 0.00
PLACE_OPEN_HOLD_STEPS = 140

PLACE_WIGGLE_DELTA = 0.20
PLACE_WIGGLE_HOLD_STEPS = 18
PLACE_WIGGLE_CYCLES = 2

# Occupancy grid file and goal waypoints
cspace = None
cspace_nav = None
MAP_FILE = 'cspace.npy'

NAV_WAYPOINTS = {
    'counter': (0.97, 0.33),
    'table': (0.40, -0.90),
}

TOTAL_JARS_TO_PICK = 3

# Stop the base immediately
def stop_base():
    motor_left.setVelocity(0.0)
    motor_right.setVelocity(0.0)

# Hold the base stopped for a fixed number of control steps
def force_stop_for_steps(n):
    for _ in range(n):
        stop_base()
        robot.step(timestep)

# Command a symmetric gripper opening
def set_gripper_open(width):
    width = clamp(width, 0.0, GRIPPER_MAX_OPEN)
    if 'gripper_left_finger_joint' in motor_handles:
        motor_handles['gripper_left_finger_joint'].setPosition(width)
    if 'gripper_right_finger_joint' in motor_handles:
        motor_handles['gripper_right_finger_joint'].setPosition(width)

# Open the gripper and hold briefly to settle motion
def open_gripper(width=GRIPPER_OPEN_GRASP, hold_steps=30):
    stop_base()
    set_gripper_open(width)
    for _ in range(hold_steps):
        stop_base()
        robot.step(timestep)

# Close the gripper and hold briefly to complete the grasp
def close_gripper(hold_steps=60):
    stop_base()
    if 'gripper_left_finger_joint' in motor_handles:
        motor_handles['gripper_left_finger_joint'].setPosition(GRIPPER_CLOSED)
    if 'gripper_right_finger_joint' in motor_handles:
        motor_handles['gripper_right_finger_joint'].setPosition(GRIPPER_CLOSED)
    for _ in range(hold_steps):
        stop_base()
        robot.step(timestep)

# Read absolute finger force feedback values
def get_finger_forces():
    fL = 0.0
    fR = 0.0
    if 'gripper_left_finger_joint' in motor_handles:
        try:
            fL = abs(float(motor_handles['gripper_left_finger_joint'].getForceFeedback()))
        except:
            fL = 0.0
    if 'gripper_right_finger_joint' in motor_handles:
        try:
            fR = abs(float(motor_handles['gripper_right_finger_joint'].getForceFeedback()))
        except:
            fR = 0.0
    return fL, fR

# Move the arm to a named pose while the base is held stationary
def move_arm_pose(pose_dict, hold_steps=40):
    stop_base()
    for j, v in pose_dict.items():
        if j in motor_handles:
            motor_handles[j].setPosition(v)
    for _ in range(hold_steps):
        stop_base()
        robot.step(timestep)

# Set torso height within limits and hold until it settles
def set_torso_lift(pos, hold_steps=60):
    if 'torso_lift_joint' not in motor_handles:
        print("WARNING torso_lift_joint motor not found cannot adjust torso height")
        return
    pos = clamp(pos, TORSO_MIN, TORSO_MAX)
    motor_handles['torso_lift_joint'].setPosition(pos)
    for _ in range(hold_steps):
        stop_base()
        robot.step(timestep)

# Convert a world pose to an occupancy grid pixel index
def world2map(xw, yw):
    world_x_min = -2.25
    world_x_max = 2.25
    world_y_min = -3.85
    world_y_max = 1.81
    px = int((xw - world_x_min) / (world_x_max - world_x_min) * 225)
    py = int((yw - world_y_min) / (world_y_max - world_y_min) * 284)
    py = 283 - py
    px = max(0, min(px, 224))
    py = max(0, min(py, 283))
    return [px, py]

# Convert a map pixel index back into a world coordinate
def map2world(px, py):
    world_x_min = -2.25
    world_x_max = 2.25
    world_y_min = -3.85
    world_y_max = 1.81
    xw = world_x_min + (px / 225) * (world_x_max - world_x_min)
    yw = world_y_min + ((283 - py) / 284) * (world_y_max - world_y_min)
    return (xw, yw)

# Compute A star heuristic cost in map pixel space
def heuristic(u, goal):
    return math.sqrt((goal[0] - u[0]) ** 2 + (goal[1] - u[1]) ** 2)

# Generate neighbors for A star traversal while respecting obstacles
def getNeighbors(u, grid):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            cand = (u[0] + i, u[1] + j)
            if (0 <= cand[0] < len(grid) and
                0 <= cand[1] < len(grid[0]) and
                not grid[cand[0], cand[1]]):
                neighbors.append((1.0, cand))
    return neighbors

# Plan a path in map space using A star and return a downsampled list of world coordinates
def pathfinding(start_map, goal_map, grid):
    queue = [(heuristic(start_map, goal_map), start_map)]
    distances = defaultdict(lambda: float("inf"))
    distances[start_map] = 0
    visited = set()
    parent = {}

    max_iterations = 15000
    it = 0
    while queue and it < max_iterations:
        it += 1
        _, v = heappop(queue)
        if v in visited:
            continue
        visited.add(v)

        if v == goal_map:
            path_map = []
            cur = goal_map
            while cur != start_map:
                path_map.append(cur)
                cur = parent[cur]
            path_map.append(start_map)
            path_map.reverse()

            # Downsample long paths to reduce waypoint churn
            if len(path_map) > 50:
                step = max(2, len(path_map) // 25)
                path_map = path_map[::step]

            return [map2world(p[0], p[1]) for p in path_map]

        for cost, u in getNeighbors(v, grid):
            if u in visited:
                continue
            new_cost = distances[v] + cost
            if new_cost < distances[u]:
                distances[u] = new_cost
                parent[u] = v
                heappush(queue, (new_cost + heuristic(u, goal_map), u))

    return []

# Check whether a grid cell is free
def is_free_cell(grid, px, py):
    try:
        return not bool(grid[px, py])
    except:
        return False

# Snap a requested cell to the nearest free cell
def snap_to_nearest_free(grid, px, py, max_r=30):
    h, w = grid.shape[0], grid.shape[1]
    if 0 <= px < h and 0 <= py < w and is_free_cell(grid, px, py):
        return (px, py)

    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                x = px + dx
                y = py + dy
                if 0 <= x < h and 0 <= y < w and is_free_cell(grid, x, y):
                    return (x, y)
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                x = px + dx
                y = py + dy
                if 0 <= x < h and 0 <= y < w and is_free_cell(grid, x, y):
                    return (x, y)
    return None

# Plan a path between two world coordinates by converting them into map indices first
def plan_path(current_pos_world, goal_world, cspace_grid):
    s_raw = world2map(current_pos_world[0], current_pos_world[1])
    g_raw = world2map(goal_world[0], goal_world[1])

    s = snap_to_nearest_free(cspace_grid, s_raw[0], s_raw[1], max_r=30)
    g = snap_to_nearest_free(cspace_grid, g_raw[0], g_raw[1], max_r=30)

    if s is None or g is None:
        print(f"plan_path could not snap to free cell start {s_raw} goal {g_raw}")
        return []

    return pathfinding(tuple(s), tuple(g), cspace_grid)

# Shift a boolean grid without wraparound to support obstacle inflation
def _roll_no_wrap_bool(arr, dx, dy):
    out = np.roll(arr, shift=(dx, dy), axis=(0, 1))
    if dx > 0:
        out[:dx, :] = False
    elif dx < 0:
        out[dx:, :] = False
    if dy > 0:
        out[:, :dy] = False
    elif dy < 0:
        out[:, dy:] = False
    return out

# Inflate obstacles by a radius in grid cells to create a conservative navigation cspace
def inflate_cspace(grid_bool, radius_cells=12):
    base = grid_bool.astype(bool)
    expanded = base.copy()
    r2 = radius_cells * radius_cells
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            if dx == 0 and dy == 0:
                continue
            if (dx * dx + dy * dy) > r2:
                continue
            shifted = _roll_no_wrap_bool(base, dx, dy)
            expanded = np.logical_or(expanded, shifted)
    return expanded

# Compute yaw from compass and convert into the controller heading convention
def get_yaw():
    comp = compass.getValues()
    yaw = math.atan2(comp[1], comp[0])
    yaw -= math.pi / 2
    yaw = -yaw
    return yaw

# Drive toward a point using proportional control on distance and heading
def nav_drive_to_point(tx, ty, speed_scale=1.0):
    xw, yw, _ = gps.getValues()
    dist = math.sqrt((xw - tx) ** 2 + (yw - ty) ** 2)

    yaw = get_yaw()
    target_angle = math.atan2(ty - yw, tx - xw)
    alpha = wrap_pi(target_angle - yaw)

    a = abs(alpha)
    if a <= NAV_ALPHA_SLOWDOWN_START:
        fwd_scale = 1.0
    elif a >= NAV_ALPHA_SLOWDOWN_FULL:
        fwd_scale = NAV_MIN_FWD_SCALE
    else:
        t = (a - NAV_ALPHA_SLOWDOWN_START) / max(1e-6, (NAV_ALPHA_SLOWDOWN_FULL - NAV_ALPHA_SLOWDOWN_START))
        fwd_scale = (1.0 - t) + t * NAV_MIN_FWD_SCALE

    v_fwd = dist * NAV_P_FWD * speed_scale * fwd_scale
    v_turn = alpha * NAV_P_TURN * speed_scale

    vl = clamp(v_fwd - v_turn, -NAV_MAX_SPEED, NAV_MAX_SPEED)
    vr = clamp(v_fwd + v_turn, -NAV_MAX_SPEED, NAV_MAX_SPEED)

    motor_left.setVelocity(clamp(vl, -WHEEL_MAX, WHEEL_MAX))
    motor_right.setVelocity(clamp(vr, -WHEEL_MAX, WHEEL_MAX))
    return dist, alpha

# Rotation and translation helpers for building camera to base transforms
def rotation_x(theta):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])

def rotation_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(theta), 0, np.cos(theta), 0],
                     [0, 0, 0, 1]])

def rotation_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translation(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

# Compute camera frame to base frame transform using torso lift and head joints
def get_camera_to_base_transform():
    lift_value = encoders['torso_lift_joint'].getValue() if 'torso_lift_joint' in encoders else 0.35
    head_pan = encoders['head_1_joint'].getValue() if 'head_1_joint' in encoders else 0.0
    head_tilt = encoders['head_2_joint'].getValue() if 'head_2_joint' in encoders else 0.0

    T0_1 = translation(0, 0, 0.6 + lift_value)
    Tt = translation(0.182, 0, 0)
    Trz = rotation_z(head_pan)
    T1_2 = Trz @ Tt
    T2_3 = rotation_y(head_tilt) @ rotation_x(-math.pi / 2)
    return T0_1 @ T1_2 @ T2_3

# Transform a point from camera coordinates into base coordinates
def transform_camera_to_base(p_cam):
    T = get_camera_to_base_transform()
    p = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    out = T @ p
    return out[:3] / out[3]

# Convert a camera observed object position into a world coordinate estimate
def compute_object_position_world(p_cam):
    p_base = transform_camera_to_base(p_cam)
    gx, gy, gz = gps.getValues()
    yaw = get_yaw()
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0, 0, 1]])
    p_world_2d = R @ np.array([p_base[0], p_base[1], 0])
    return [gx + p_world_2d[0], gy + p_world_2d[1], gz + p_base[2]]

# Compute jar world position and planar distance from the robot
def jar_world_distance_from_obj(obj):
    p_cam = list(obj.getPosition())
    jar_w = compute_object_position_world(p_cam)
    rx, ry, _ = gps.getValues()
    dist_w = math.sqrt((jar_w[0] - rx) ** 2 + (jar_w[1] - ry) ** 2)
    return jar_w, dist_w

# Model string matching helper for jar recognition objects
def _jar_match(model_str: str) -> bool:
    m = (model_str or "").lower()
    return ("jar" in m) or ("jam" in m)

# Safe object id extraction for recognition objects
def _safe_get_id(obj):
    try:
        return int(obj.getId())
    except:
        return None

# Extract image features used for centering and tracking continuity
def _get_obj_image_features(obj):
    z = float(obj.getPosition()[2])

    cam_w = camera.getWidth()
    cam_h = camera.getHeight()

    # Prefer image based features when available
    if hasattr(obj, "getPositionOnImage") and hasattr(obj, "getSizeOnImage"):
        cx, cy = obj.getPositionOnImage()
        bw, bh = obj.getSizeOnImage()

        nx = (float(cx) - (cam_w / 2.0)) / (cam_w / 2.0)
        ny = (float(cy) - (cam_h / 2.0)) / (cam_h / 2.0)

        ex = nx if (CAM_HORIZ_AXIS == 'x') else ny
        ex *= CAM_HORIZ_SIGN

        area = float(bw) * float(bh)
        return ex, float(cx), float(cy), area, z

    # Fallback feature using object local position in camera space
    x, y, z3 = obj.getPosition()
    z3 = float(z3)
    ex = clamp(float(x) / max(1e-3, z3), -1.0, 1.0) * CAM_HORIZ_SIGN
    return ex, None, None, 0.0, z3

# Ranking score for choosing the best visible jar candidate
def _center_preference_score(ex, z, area):
    return (-abs(ex), -z, area)

# Acquire a jar either by current lock id or by best visible candidate
def get_locked_or_best_jar(prefer_id=None, prefer_model=None, last_cx=None, last_cy=None, last_z=None):
    objs = camera.getRecognitionObjects()
    if not objs:
        return None, None, None, 0.0, None, None, float("inf")

    candidates = []
    for obj in objs:
        model = obj.getModel() or ""
        if not _jar_match(model):
            continue
        oid = _safe_get_id(obj)
        ex, cx, cy, area, z = _get_obj_image_features(obj)
        candidates.append((obj, model, oid, ex, cx, cy, area, z))

    if not candidates:
        return None, None, None, 0.0, None, None, float("inf")

    # Hard lock preference when the same id is still visible
    if prefer_id is not None:
        for (obj, model, oid, ex, cx, cy, area, z) in candidates:
            if oid == prefer_id:
                return obj, model, oid, ex, cx, cy, z

    # Continuity preference when we have a recent image feature history
    if (last_cx is not None) and (last_cy is not None) and (last_z is not None):
        best = None
        best_cost = None
        for (obj, model, oid, ex, cx, cy, area, z) in candidates:
            if cx is None or cy is None:
                continue
            dc = (cx - last_cx) ** 2 + (cy - last_cy) ** 2
            dz = (z - last_z) ** 2
            m_bonus = 0.0
            if prefer_model and (prefer_model.lower() in model.lower()):
                m_bonus = -2500.0
            cost = dc + 6000.0 * dz + m_bonus
            if best is None or cost < best_cost:
                best = (obj, model, oid, ex, cx, cy, z)
                best_cost = cost
        if best is not None:
            return best

    # Otherwise select the best centered closest and largest target
    def key_fn(t):
        _obj, model, oid, ex, cx, cy, area, z = t
        match_bonus = 1 if (prefer_model and (prefer_model.lower() in model.lower())) else 0
        sc = _center_preference_score(ex, z, area)
        return (match_bonus, sc[0], sc[1], sc[2])

    best = max(candidates, key=key_fn)
    obj, model, oid, ex, cx, cy, area, z = best
    return obj, model, oid, ex, cx, cy, z

# Turn in place based on camera horizontal error to center the jar
def visual_turn_from_ex(ex, max_turn):
    turn = clamp(-APPROACH_KP_TURN * ex, -max_turn, max_turn)
    motor_left.setVelocity(clamp(-turn, -WHEEL_MAX, WHEEL_MAX))
    motor_right.setVelocity(clamp(turn, -WHEEL_MAX, WHEEL_MAX))

# Resample the jar world pose multiple times and return a robust median estimate
def acquire_and_refine_locked_target(samples=10, prefer_id=None, prefer_model=None):
    stop_base()
    xs, ys, zs = [], [], []
    last_cx = None
    last_cy = None
    last_z = None

    for _ in range(samples):
        robot.step(timestep)
        stop_base()
        obj, model, oid, ex, cx, cy, z = get_locked_or_best_jar(
            prefer_id=prefer_id, prefer_model=prefer_model,
            last_cx=last_cx, last_cy=last_cy, last_z=last_z
        )
        if obj is None:
            continue
        last_cx, last_cy, last_z = cx, cy, z
        p_cam = list(obj.getPosition())
        world = compute_object_position_world(p_cam)
        xs.append(float(world[0]))
        ys.append(float(world[1]))
        zs.append(float(world[2]))

    if not xs:
        return None
    xs.sort()
    ys.sort()
    zs.sort()
    return (xs[len(xs)//2], ys[len(ys)//2], zs[len(zs)//2])

# Backup primitive that drives backward a fixed distance while maintaining heading
def backup_distance_straight(distance_m, reverse_speed=BACKUP_AFTER_CLOSE_SPEED):
    stop_base()
    force_stop_for_steps(5)

    x0, y0, _ = gps.getValues()
    yaw_ref = get_yaw()

    while robot.step(timestep) != -1:
        xw, yw, _ = gps.getValues()
        d = math.sqrt((xw - x0) ** 2 + (yw - y0) ** 2)
        if d >= distance_m:
            break

        yaw = get_yaw()
        heading_err = wrap_pi(yaw_ref - yaw)
        turn = clamp(BACKUP_AFTER_CLOSE_YAW_KP * heading_err,
                     -BACKUP_AFTER_CLOSE_TURN_MAX, BACKUP_AFTER_CLOSE_TURN_MAX)

        vl = -reverse_speed - turn
        vr = -reverse_speed + turn
        motor_left.setVelocity(clamp(vl, -WHEEL_MAX, WHEEL_MAX))
        motor_right.setVelocity(clamp(vr, -WHEEL_MAX, WHEEL_MAX))

    stop_base()
    force_stop_for_steps(10)

# Nudge forward primitive used during placement
def nudge_forward_distance(distance_m, forward_speed=PLACE_NUDGE_FWD_SPEED):
    stop_base()
    force_stop_for_steps(5)

    x0, y0, _ = gps.getValues()
    yaw_ref = get_yaw()

    while robot.step(timestep) != -1:
        xw, yw, _ = gps.getValues()
        d = math.sqrt((xw - x0) ** 2 + (yw - y0) ** 2)
        if d >= distance_m:
            break

        yaw = get_yaw()
        heading_err = wrap_pi(yaw_ref - yaw)
        turn = clamp(PLACE_NUDGE_YAW_KP * heading_err, -PLACE_NUDGE_TURN_MAX, PLACE_NUDGE_TURN_MAX)

        vl = forward_speed - turn
        vr = forward_speed + turn
        motor_left.setVelocity(clamp(vl, -WHEEL_MAX, WHEEL_MAX))
        motor_right.setVelocity(clamp(vr, -WHEEL_MAX, WHEEL_MAX))

    stop_base()
    force_stop_for_steps(10)

# Tight docking controller to hit an exact table point without driving into the table
def dock_to_exact_point(goalx, goaly):
    xw, yw, _ = gps.getValues()
    dist = math.sqrt((xw - goalx) ** 2 + (yw - goaly) ** 2)

    yaw = get_yaw()
    target_angle = math.atan2(goaly - yw, goalx - xw)
    alpha = wrap_pi(target_angle - yaw)

    # If close enough stop and hand off to yaw facing stage
    if dist <= TABLE_DOCK_EPS:
        stop_base()
        return True, dist, alpha

    # Rotate first when significantly misaligned
    if abs(alpha) > DOCK_ALIGN_ONLY_ALPHA:
        v = 0.0
        w = clamp(P_DOCK_TURN * alpha, -WHEEL_MAX * DOCK_MAX_SPEED_SCALE, WHEEL_MAX * DOCK_MAX_SPEED_SCALE)
    else:
        v = clamp(P_DOCK_FWD * dist, 0.10, WHEEL_MAX * DOCK_MAX_SPEED_SCALE)
        w = clamp(P_DOCK_TURN * alpha, -WHEEL_MAX * DOCK_MAX_SPEED_SCALE, WHEEL_MAX * DOCK_MAX_SPEED_SCALE)

    motor_left.setVelocity(clamp(v - w, -WHEEL_MAX, WHEEL_MAX))
    motor_right.setVelocity(clamp(v + w, -WHEEL_MAX, WHEEL_MAX))
    return False, dist, alpha

# Rotate in place until yaw matches the desired table placement yaw
def face_yaw(target_yaw):
    yaw = get_yaw()
    err = wrap_pi(target_yaw - yaw)
    turn = clamp(TABLE_YAW_KP * err, -WHEEL_MAX, WHEEL_MAX)
    motor_left.setVelocity(-turn)
    motor_right.setVelocity(turn)
    return err

# Place sequence that nudges opens and wiggles to reduce drop jamming
def place_release_sequence():
    stop_base()
    force_stop_for_steps(10)

    # Lower torso before release so the jar is closer to the surface
    if 'torso_lift_joint' in motor_handles:
        set_torso_lift(PLACE_TORSO_DOWN, hold_steps=80)

    # Small controlled forward nudge to place closer to the intended point
    nudge_forward_distance(PLACE_NUDGE_FWD_DIST, forward_speed=PLACE_NUDGE_FWD_SPEED)

    # Open gripper fully and hold to ensure release
    set_gripper_open(GRIPPER_MAX_OPEN)
    for _ in range(10):
        stop_base()
        robot.step(timestep)
    set_gripper_open(GRIPPER_MAX_OPEN)

    # Wiggle wrist joint to reduce sticking
    if 'arm_7_joint' in motor_handles:
        base = float(ARM_PICK_POSE.get('arm_7_joint', -1.57))
        for _ in range(PLACE_WIGGLE_CYCLES):
            motor_handles['arm_7_joint'].setPosition(base + PLACE_WIGGLE_DELTA)
            for _ in range(PLACE_WIGGLE_HOLD_STEPS):
                stop_base()
                robot.step(timestep)
            motor_handles['arm_7_joint'].setPosition(base - PLACE_WIGGLE_DELTA)
            for _ in range(PLACE_WIGGLE_HOLD_STEPS):
                stop_base()
                robot.step(timestep)
        motor_handles['arm_7_joint'].setPosition(base)

    # Keep commanding open during the hold window
    for _ in range(PLACE_OPEN_HOLD_STEPS):
        set_gripper_open(GRIPPER_MAX_OPEN)
        stop_base()
        robot.step(timestep)

# Load the occupancy grid map and inflate to create a navigation cspace
print(f"Map file {os.path.join(os.getcwd(), MAP_FILE)}")
if exists(MAP_FILE):
    print("Map exists loading")
    cspace = np.load(MAP_FILE)
    print("Loaded map")

    if cspace.dtype != np.bool_:
        cspace = (cspace > 0.5)

    print("Inflating cspace for safer navigation")
    cspace_nav = inflate_cspace(cspace, radius_cells=12)
    print("Inflation complete")
else:
    print("ERROR cspace npy not found this controller expects an existing map")
    while robot.step(timestep) != -1:
        stop_base()

# Move the arm to a safe travel posture before starting navigation and manipulation
print("Stopping base before moving arm to safe pose")
stop_base()
force_stop_for_steps(8)

print("Moving arm to safe pose base stopped")
move_arm_pose(ARM_SAFE_POSE, hold_steps=80)
print("Arm safe pose command complete")

# State machine labels for navigation search grasp transport docking and placement
STATE_PLAN_TO_COUNTER = "PLAN_COUNTER"
STATE_FOLLOW_TO_COUNTER = "FOLLOW_COUNTER"
STATE_SEARCH = "SEARCH_FOR_JAR"
STATE_ORIENT = "ORIENT_TO_JAR_VISUAL"
STATE_APPROACH = "APPROACH_TO_DIST_VISUAL"
STATE_ARM_POSE = "MOVE_ARM_PICK_POSE"
STATE_ARM_LEVEL = "ALIGN_ARM_LEVEL"
STATE_OPEN_AND_CREEP = "OPEN_AND_CREEP"
STATE_CLOSE = "CLOSE_GRIPPER"

STATE_PLAN_TO_TABLE = "PLAN_TABLE"
STATE_FOLLOW_TO_TABLE = "FOLLOW_TABLE"
STATE_DOCK_TABLE = "DOCK_TABLE_EXACT"
STATE_FACE_PI = "FACE_TABLE_YAW"
STATE_PLACE_AT_TABLE = "PLACE_AT_TABLE"
STATE_DONE = "DONE"

# Runtime variables for path following and state progression
state = STATE_PLAN_TO_COUNTER
planned_path = []
path_index = 0

# Jar lock state for robust tracking through brief occlusions
locked_target_model = None
locked_target_id = None

# Ignore jars after leaving the table until the robot reaches the counter checkpoint again
ignore_jars_until_counter = False

# Vision lock management and search cadence counters
lock_lost_count = 0
orient_hold = 0
search_pause_count = 0

# Last seen image features for continuity tracking
last_ex = 0.0
last_cx = None
last_cy = None
last_z = None

# Creep grasp internal state variables
creep_steps = 0
both_contact_hold = 0
creep_yaw_ref = None
side_latch = 0
side_latch_hold = 0

# Docking and facing hold counters
dock_steps = 0
face_hold = 0

# Jar count progress
picked_count = 0

print("Starting controller")

# Main control loop implementing navigation jar picking and table placement
while robot.step(timestep) != -1:

    # Plan a path to the counter checkpoint
    if state == STATE_PLAN_TO_COUNTER:
        xw, yw, _ = gps.getValues()
        goal = NAV_WAYPOINTS['counter']
        print(f"Planning path to counter Picked {picked_count} of {TOTAL_JARS_TO_PICK}")
        planned_path = plan_path((xw, yw), goal, cspace_nav)
        if not planned_path:
            print("Plan failed retrying")
            continue
        path_index = 0
        print(f"Planned path with {len(planned_path)} points")
        state = STATE_FOLLOW_TO_COUNTER

    # Follow the planned path or drive directly if path is empty
    elif state == STATE_FOLLOW_TO_COUNTER:
        goalx, goaly = NAV_WAYPOINTS['counter']
        xw, yw, _ = gps.getValues()
        dist_goal = math.sqrt((xw - goalx) ** 2 + (yw - goaly) ** 2)

        # Counter arrival gates search behavior and clears ignore mode
        if dist_goal < GOAL_REACHED_COUNTER:
            ignore_jars_until_counter = False
            print("Reached counter")
            stop_base()
            force_stop_for_steps(12)
            search_pause_count = 0
            state = STATE_SEARCH
            continue

        if not planned_path or path_index >= len(planned_path):
            nav_drive_to_point(goalx, goaly, speed_scale=1.0)
        else:
            tx, ty = planned_path[path_index]
            dist, _ = nav_drive_to_point(tx, ty, speed_scale=1.0)
            if dist < PATH_POINT_REACHED:
                path_index += 1

    # Spin in place with pauses to scan for jars
    elif state == STATE_SEARCH:
        # While ignoring jars just keep spinning and do not lock any targets
        if ignore_jars_until_counter:
            if (search_pause_count % SEARCH_PAUSE_EVERY) < SEARCH_PAUSE_STEPS:
                stop_base()
            else:
                motor_left.setVelocity(-SEARCH_SPIN_SPEED)
                motor_right.setVelocity(+SEARCH_SPIN_SPEED)
            search_pause_count += 1
            continue

        if (search_pause_count % SEARCH_PAUSE_EVERY) < SEARCH_PAUSE_STEPS:
            stop_base()
        else:
            motor_left.setVelocity(-SEARCH_SPIN_SPEED)
            motor_right.setVelocity(+SEARCH_SPIN_SPEED)
        search_pause_count += 1

        obj, model, oid, ex, cx, cy, z = get_locked_or_best_jar()

        # Lock onto the best jar candidate and transition to orient stage
        if obj is not None:
            locked_target_model = model
            locked_target_id = oid
            lock_lost_count = 0
            orient_hold = 0
            last_ex = ex
            last_cx, last_cy, last_z = cx, cy, z

            _, dist_w = jar_world_distance_from_obj(obj)
            print(f"Jar acquired locked model dist {dist_w:.2f}")
            state = STATE_ORIENT
            continue

    # Turn in place until the jar is centered in the camera
    elif state == STATE_ORIENT:
        obj, model, oid, ex, cx, cy, z = get_locked_or_best_jar(
            prefer_id=locked_target_id, prefer_model=locked_target_model,
            last_cx=last_cx, last_cy=last_cy, last_z=last_z
        )

        # If target is missing attempt a brief reacquire before returning to search
        if obj is None:
            lock_lost_count += 1
            if lock_lost_count <= CAM_REACQUIRE_GRACE:
                visual_turn_from_ex(last_ex, APPROACH_MAX_TURN * 0.7)
            else:
                motor_left.setVelocity(+0.20)
                motor_right.setVelocity(-0.20)

            if lock_lost_count >= CAM_LOST_MAX:
                print("Lost jar while orienting back to search")
                stop_base()
                force_stop_for_steps(10)
                locked_target_model = None
                locked_target_id = None
                state = STATE_SEARCH
            continue

        locked_target_model = model
        locked_target_id = oid
        lock_lost_count = 0

        last_ex = ex
        last_cx, last_cy, last_z = cx, cy, z

        visual_turn_from_ex(ex, APPROACH_MAX_TURN)

        # Require a short hold window inside the centering threshold before advancing
        if abs(ex) < CAM_CENTER_EPS:
            orient_hold += 1
            if orient_hold >= CAM_CENTER_HOLD_TICKS:
                stop_base()
                force_stop_for_steps(12)
                print(f"Jar centered id {locked_target_id} approaching to standoff {TARGET_DIST:.2f}")
                state = STATE_APPROACH
        else:
            orient_hold = 0

    # Approach to a fixed standoff distance while maintaining camera centering
    elif state == STATE_APPROACH:
        obj, model, oid, ex, cx, cy, z = get_locked_or_best_jar(
            prefer_id=locked_target_id, prefer_model=locked_target_model,
            last_cx=last_cx, last_cy=last_cy, last_z=last_z
        )

        # Handle lock loss during approach
        if obj is None:
            lock_lost_count += 1
            if lock_lost_count <= CAM_REACQUIRE_GRACE:
                visual_turn_from_ex(last_ex, APPROACH_MAX_TURN * 0.7)
            else:
                motor_left.setVelocity(+0.22)
                motor_right.setVelocity(-0.22)

            if lock_lost_count >= CAM_LOST_MAX:
                print("Lost jar while approaching back to search")
                stop_base()
                force_stop_for_steps(10)
                locked_target_model = None
                locked_target_id = None
                state = STATE_SEARCH
            continue

        locked_target_model = model
        locked_target_id = oid
        lock_lost_count = 0

        last_ex = ex
        last_cx, last_cy, last_z = cx, cy, z

        _, dist_w = jar_world_distance_from_obj(obj)

        # If too close reverse away while still keeping visual centering engaged
        if dist_w < (TARGET_DIST - TOO_CLOSE_EPS):
            dist_err = (TARGET_DIST - dist_w)
            scale = clamp(dist_err / max(SLOWDOWN_RADIUS, 1e-3), 0.15, BACKUP_MAX_SCALE)

            turn = clamp(-APPROACH_KP_TURN * ex, -APPROACH_MAX_TURN, APPROACH_MAX_TURN)
            fwd = -BACKUP_GAIN * scale

            vl = clamp(fwd - turn, -NAV_MAX_SPEED, NAV_MAX_SPEED)
            vr = clamp(fwd + turn, -NAV_MAX_SPEED, NAV_MAX_SPEED)
            motor_left.setVelocity(clamp(vl, -WHEEL_MAX, WHEEL_MAX))
            motor_right.setVelocity(clamp(vr, -WHEEL_MAX, WHEEL_MAX))
            continue

        # When within standoff band and centered transition to arm pose stage
        if (dist_w <= (TARGET_DIST + DIST_EPS)) and (dist_w >= (TARGET_DIST - TOO_CLOSE_EPS)) and (abs(ex) < CAM_CENTER_EPS):
            stop_base()
            force_stop_for_steps(18)
            print(f"At standoff {TARGET_DIST:.2f} and centered id {locked_target_id} dist {dist_w:.2f} z {z:.2f}")
            state = STATE_ARM_POSE
            continue

        # Scale forward speed down as we get close to the standoff radius
        dist_to_go = max(0.0, dist_w - TARGET_DIST)
        speed_scale = clamp(dist_to_go / SLOWDOWN_RADIUS, 0.15, 1.0) if dist_to_go < SLOWDOWN_RADIUS else 1.0

        turn = clamp(-APPROACH_KP_TURN * ex, -APPROACH_MAX_TURN, APPROACH_MAX_TURN)
        fwd = clamp(APPROACH_KP_FWD * (dist_w - TARGET_DIST) * speed_scale, 0.0, APPROACH_MAX_FWD)

        vl = clamp(fwd - turn, -NAV_MAX_SPEED, NAV_MAX_SPEED)
        vr = clamp(fwd + turn, -NAV_MAX_SPEED, NAV_MAX_SPEED)
        motor_left.setVelocity(clamp(vl, -WHEEL_MAX, WHEEL_MAX))
        motor_right.setVelocity(clamp(vr, -WHEEL_MAX, WHEEL_MAX))

    # Move arm into pick pose before torso leveling and creep grasp
    elif state == STATE_ARM_POSE:
        stop_base()
        force_stop_for_steps(10)
        print("Moving arm into pick pose for leveling")
        move_arm_pose(ARM_PICK_POSE, hold_steps=80)
        state = STATE_ARM_LEVEL

    # Level torso height to jar height then open gripper and start creep stage
    elif state == STATE_ARM_LEVEL:
        stop_base()
        force_stop_for_steps(10)

        # Resample jar to improve torso estimate robustness
        resampled = acquire_and_refine_locked_target(samples=10, prefer_id=locked_target_id, prefer_model=locked_target_model)

        if resampled is None:
            print("Warning could not resample jar for torso leveling proceeding with default torso")
            desired_torso = 0.35
        else:
            _tx, _ty, jar_z = resampled
            desired_torso = clamp(jar_z - GRIPPER_Z_OFFSET, TORSO_MIN, TORSO_MAX)

        print(f"Setting torso lift to {desired_torso:.3f} gripper offset {GRIPPER_Z_OFFSET:.3f}")
        set_torso_lift(desired_torso, hold_steps=100)

        print("Opening gripper then creeping close when both sides contact")
        open_gripper(width=GRIPPER_OPEN_GRASP, hold_steps=40)

        creep_steps = 0
        both_contact_hold = 0
        creep_yaw_ref = get_yaw()
        side_latch = 0
        side_latch_hold = 0
        state = STATE_OPEN_AND_CREEP

    # Creep forward until bilateral contact is detected then stop and close gripper
    elif state == STATE_OPEN_AND_CREEP:
        yaw = get_yaw()
        heading_err = wrap_pi(creep_yaw_ref - yaw)
        turn_h = clamp(CREEP_HEADING_KP * heading_err, -CREEP_TURN_MAX, CREEP_TURN_MAX)

        obj, model, oid, ex, cx, cy, z = get_locked_or_best_jar(
            prefer_id=locked_target_id, prefer_model=locked_target_model,
            last_cx=last_cx, last_cy=last_cy, last_z=last_z
        )
        if obj is not None:
            locked_target_model = model
            locked_target_id = oid
            last_ex = ex
            last_cx, last_cy, last_z = cx, cy, z
            turn_vis = clamp(-CREEP_VISUAL_KP * ex, -CREEP_VISUAL_TURN_MAX, CREEP_VISUAL_TURN_MAX)
        else:
            turn_vis = 0.0

        fL, fR = get_finger_forces()

        # Side latch sets a turning bias away from one sided collisions
        if (fL >= SIDE_DETECT_THRESHOLD) and (fL > fR + 0.05):
            side_latch = +1
            side_latch_hold = SIDE_LATCH_MAX_HOLD
        elif (fR >= SIDE_DETECT_THRESHOLD) and (fR > fL + 0.05):
            side_latch = -1
            side_latch_hold = SIDE_LATCH_MAX_HOLD
        else:
            if (fL <= SIDE_RELEASE_THRESHOLD) and (fR <= SIDE_RELEASE_THRESHOLD):
                side_latch_hold = max(0, side_latch_hold - 1)
                if side_latch_hold == 0:
                    side_latch = 0

        base_turn = clamp(turn_h + turn_vis, -CREEP_TURN_MAX, CREEP_TURN_MAX)

        if side_latch != 0:
            turn = clamp((0.25 * base_turn) + (side_latch * SIDE_LATCH_TURN),
                         -CREEP_TURN_MAX, CREEP_TURN_MAX)
        else:
            turn = base_turn

        vl = CREEP_FWD_SPEED - turn
        vr = CREEP_FWD_SPEED + turn
        motor_left.setVelocity(clamp(vl, -WHEEL_MAX, WHEEL_MAX))
        motor_right.setVelocity(clamp(vr, -WHEEL_MAX, WHEEL_MAX))

        creep_steps += 1

        # Require multiple consecutive ticks above threshold for robust contact confirmation
        if (fL >= SIDE_FORCE_THRESHOLD) and (fR >= SIDE_FORCE_THRESHOLD):
            both_contact_hold += 1
        else:
            both_contact_hold = 0

        if both_contact_hold >= BOTH_CONTACT_HOLD_TICKS:
            print(f"Both sides contact detected fL {fL:.2f} fR {fR:.2f} stopping and closing gripper")
            stop_base()
            force_stop_for_steps(10)
            state = STATE_CLOSE

    # Close gripper then back up and transition to table navigation
    elif state == STATE_CLOSE:
        close_gripper(hold_steps=80)

        print(f"Gripper closed backing up {BACKUP_AFTER_CLOSE_DIST:.2f}")
        backup_distance_straight(BACKUP_AFTER_CLOSE_DIST)

        print("Moving arm to post close pose then going to table")
        move_arm_pose(ARM_POST_CLOSE_POSE, hold_steps=80)

        state = STATE_PLAN_TO_TABLE

    # Plan a path to the table waypoint
    elif state == STATE_PLAN_TO_TABLE:
        xw, yw, _ = gps.getValues()
        goal = NAV_WAYPOINTS['table']
        print(f"Planning path to table goal {goal}")
        planned_path = plan_path((xw, yw), goal, cspace_nav)
        if not planned_path:
            print("Plan to table failed retrying")
            continue

        path_index = 0
        print(f"Planned path to table with {len(planned_path)} points")
        state = STATE_FOLLOW_TO_TABLE

    # Follow the path to the table then hand off to tight docking
    elif state == STATE_FOLLOW_TO_TABLE:
        goalx, goaly = NAV_WAYPOINTS['table']
        xw, yw, _ = gps.getValues()
        dist_goal = math.sqrt((xw - goalx) ** 2 + (yw - goaly) ** 2)

        if dist_goal <= TABLE_SWITCH_TO_DOCK:
            stop_base()
            force_stop_for_steps(10)
            dock_steps = 0
            print(f"Near table dist {dist_goal:.3f} switching to tight docking")
            state = STATE_DOCK_TABLE
            continue

        if not planned_path or path_index >= len(planned_path):
            nav_drive_to_point(goalx, goaly, speed_scale=1.0)
        else:
            tx, ty = planned_path[path_index]
            dist, _ = nav_drive_to_point(tx, ty, speed_scale=1.0)
            if dist < PATH_POINT_REACHED:
                path_index += 1

    # Dock precisely to the table waypoint coordinate
    elif state == STATE_DOCK_TABLE:
        goalx, goaly = NAV_WAYPOINTS['table']
        done, dist, alpha = dock_to_exact_point(goalx, goaly)
        dock_steps += 1

        if done:
            stop_base()
            force_stop_for_steps(12)
            face_hold = 0
            print(f"Docked to table within {TABLE_DOCK_EPS:.3f} now facing yaw {TABLE_TARGET_YAW:.3f}")
            state = STATE_FACE_PI
            continue

        if dock_steps > TABLE_DOCK_MAX_STEPS:
            print("Warning docking max steps exceeded proceeding to face yaw anyway")
            stop_base()
            force_stop_for_steps(12)
            face_hold = 0
            state = STATE_FACE_PI

    # Rotate in place to the placement yaw before releasing
    elif state == STATE_FACE_PI:
        err = face_yaw(TABLE_TARGET_YAW)
        if abs(err) < TABLE_YAW_EPS:
            face_hold += 1
            if face_hold >= TABLE_YAW_HOLD_TICKS:
                stop_base()
                force_stop_for_steps(15)
                print("At table point and facing target yaw placing jar")
                state = STATE_PLACE_AT_TABLE
        else:
            face_hold = 0

    # Place jar then back up and decide whether to continue or finish
    elif state == STATE_PLACE_AT_TABLE:
        stop_base()
        force_stop_for_steps(10)

        print("Moving arm to pick pose at table for placement")
        move_arm_pose(ARM_PICK_POSE, hold_steps=80)

        print("Releasing jar")
        place_release_sequence()

        print("Restoring arm to safe pose")
        move_arm_pose(ARM_SAFE_POSE, hold_steps=80)

        print(f"Backing up {BACKUP_AFTER_PLACE_DIST:.2f} away from table")
        backup_distance_straight(BACKUP_AFTER_PLACE_DIST)

        picked_count += 1
        print(f"Placed jar progress {picked_count} of {TOTAL_JARS_TO_PICK}")

        if picked_count >= TOTAL_JARS_TO_PICK:
            print("All jars placed done")
            state = STATE_DONE
        else:
            # Ignore jars until reaching counter again to prevent immediate relock near table
            ignore_jars_until_counter = True

            # Clear lock and tracking history before returning to counter
            locked_target_model = None
            locked_target_id = None
            lock_lost_count = 0
            orient_hold = 0
            search_pause_count = 0
            last_ex = 0.0
            last_cx = None
            last_cy = None
            last_z = None

            planned_path = []
            path_index = 0
            dock_steps = 0
            face_hold = 0

            state = STATE_PLAN_TO_COUNTER

    # Done state holds the base stopped
    elif state == STATE_DONE:
        stop_base()
        continue
