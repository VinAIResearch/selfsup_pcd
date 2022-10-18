import numpy as np


def standardize_bbox(pcl):

    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.0
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


xml_head = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,0" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
    </sensor>
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""

xml_ball_segment = """
    <shape type="sphere">
        <float name="radius" value="0.025"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def write_file(pcl, file_name):
    xml_segments = [xml_head]

    pcl = standardize_bbox(pcl)
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    for i in range(pcl.shape[0]):
        color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join("", xml_segments)

    with open(file_name, "w") as f:
        f.write(xml_content)
