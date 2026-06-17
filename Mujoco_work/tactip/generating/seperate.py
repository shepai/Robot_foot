import xml.etree.ElementTree as ET

tree = ET.parse("/home/dexter/Documents/GitHub/Robot_foot/Mujoco_work/tactip/generating/generated.xml")
root = tree.getroot()

for tag in ["asset", "worldbody", "tendon", "actuator", "sensor","deformable"]:
    elem = root.find(tag)

    if elem is not None:
        ET.ElementTree(elem).write(
            f"/home/dexter/Documents/GitHub/Robot_foot/Mujoco_work/tactip/generating/{tag}.xml",
            encoding="utf-8",
            xml_declaration=False,
        )