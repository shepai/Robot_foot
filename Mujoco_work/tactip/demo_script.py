import time
import mujoco
import mujoco.viewer

print(f"--- ACTIVE ENVIRONMENT: MuJoCo {mujoco.__version__} ---")


def main():
    try:
        model = mujoco.MjModel.from_xml_path("/home/dexter/Documents/GitHub/Robot_foot/Mujoco_work/tactip/tactile_env.xml")
        data = mujoco.MjData(model)
        print("Success! Model parsed, reference angles assigned, and scene compiled perfectly.")
    except Exception as e:
        print("\n[CRITICAL ERROR] Parser failed:")
        print(e)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
