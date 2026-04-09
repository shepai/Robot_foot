import mujoco as mj
import mujoco.viewer
import numpy as np
import cv2
import numpy as np

def show_heatmap(grid):
    # Normalize to 0–255
    norm = grid.copy()
    if norm.max() > 0:
        norm = norm / norm.max()

    img = (norm * 255).astype(np.uint8)

    # Resize to make it visible
    img = cv2.resize(img, (300, 180), interpolation=cv2.INTER_NEAREST)

    # Apply color map
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    cv2.imshow("Force Pads", img)
    cv2.waitKey(1)
import time

# Load model
model = mj.MjModel.from_xml_path("C:/Users/dexte/Documents/GitHub/Robot_foot/Mujuco/ball_presstip_env2.xml")
data = mj.MjData(model)

rows, cols = 3, 3
pad_names_center = [[f"pad_{i}_{j}" for j in range(cols)] for i in range(rows)]

# 4 wedges
pad_names_wedges = ["wedge_top", "wedge_bottom", "wedge_left", "wedge_right"]
pad_geom_ids = {
    name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
    for row in pad_names_center for name in row
}
for name in pad_names_wedges:
    pad_geom_ids[name] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)

def get_pad_forces_grid():
    # 5x5 grid for visualization
    grid = np.zeros((5, 5), dtype=float)

    # --- central 3x3 ---
    for r in range(3):
        for c in range(3):
            name = pad_names_center[r][c]
            gid = pad_geom_ids[name]
            force = 0.0
            for i in range(data.ncon):
                contact = data.contact[i]
                if contact.geom1 == gid or contact.geom2 == gid:
                    f = np.zeros(6)
                    mj.mj_contactForce(model, data, i, f)
                    force += f[0]
            grid[r+1, c+1] = force  # offset by 1 to leave borders for wedges

    # --- wedges ---
    wedge_positions = {
        "wedge_top": (0, 2),
        "wedge_bottom": (4, 2),
        "wedge_left": (2, 0),
        "wedge_right": (2, 4)
    }
    for name, (r, c) in wedge_positions.items():
        gid = pad_geom_ids[name]
        force = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == gid or contact.geom2 == gid:
                f = np.zeros(6)
                mj.mj_contactForce(model, data, i, f)
                force += f[0]
        grid[r, c] = force

    return grid

# --- Viewer loop ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Running... close window or Ctrl+C to exit")

    while viewer.is_running():
        start = time.time()

        # Step simulation
        mj.mj_step(model, data)

        # Get force grid
        grid = get_pad_forces_grid()

        # Update heatmap
        show_heatmap(grid)      
        
        # Sync MuJoCo viewer
        viewer.sync()

        # Real-time pacing
        time.sleep(max(0, model.opt.timestep - (time.time() - start)))
