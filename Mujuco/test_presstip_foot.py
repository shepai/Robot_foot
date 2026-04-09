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
model = mj.MjModel.from_xml_path("C:/Users/dexte/Documents/GitHub/Robot_foot/Mujuco/ball_presstip_env.xml")
data = mj.MjData(model)

# --- Pad setup ---
rows, cols = 3, 5
pad_names = [[f"pad_{i}_{j}" for j in range(cols)] for i in range(rows)]

pad_geom_ids = {
    name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
    for row in pad_names for name in row
}

def get_pad_forces_grid():
    grid = np.zeros((rows, cols))

    for i in range(data.ncon):
        contact = data.contact[i]

        for r in range(rows):
            for c in range(cols):
                name = pad_names[r][c]
                gid = pad_geom_ids[name]

                if contact.geom1 == gid or contact.geom2 == gid:
                    f = np.zeros(6)
                    mj.mj_contactForce(model, data, i, f)

                    # Use normal force only
                    grid[r, c] += f[0]

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
