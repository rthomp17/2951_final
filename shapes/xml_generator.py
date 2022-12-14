

import xml.etree.ElementTree as ET

'''
inputs: 
name -> string of file name without extension
shape -> string: circle, triangle, or rectangle
'''
def generateXML(name, shape): 
    color = "0 0 0 1"
    if shape == "circle":
        color = "1 0 0 1"
    elif shape == "triangle":
        color = "0 1 0 1"
    elif shape == "rectangle":
        color = "0 0 1 1"

    mujoco = ET.Element("mujoco", name=name)
    asset = ET.SubElement(mujoco, "asset")

    ET.SubElement(asset, "mesh", file="./shapes/shape_objs/" + name + ".stl", name=name + "_mesh").text = ""

    worldbody = ET.SubElement(mujoco, "worldbody")
    body = ET.SubElement(worldbody, "body")

    subbody = ET.SubElement(body, "body", name="object")
    ET.SubElement(subbody, "geom", pos="0 0 0", mesh=name, type="mesh", group="0", rgba=color)

    ET.SubElement(body, "site", rgba="0 0 0 0", size="0.005", pos="0 0 -0.06", name="bottom_site")
    ET.SubElement(body, "site", rgba="0 0 0 0", size="0.005", pos="0 0 0.04", name="top_site")
    ET.SubElement(body, "site", rgba="0 0 0 0", size="0.005", pos="0.025 0.025 0", name="horizontal_radius_site")

    tree = ET.ElementTree(mujoco)
    tree.write("./shapes/xml_objs/" + name + ".xml")

def main():
    for i in range(0, 10):
        generateXML("test-shape-" + str(i), "triangle")

    for i in range(10, 20):
        generateXML("test-shape-" + str(i), "rectangle")

    for i in range(0, 20):
        generateXML("train-shape-" + str(i), "triangle")

    for i in range(20, 40):
        generateXML("train-shape-" + str(i), "rectangle")


if __name__ == "__main__":
    main()
