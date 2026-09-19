"""Create a shallow-crowned, MVAE-safe variant of the 5000-face proxy.

The previous bump removal flattened twelve rear/top vertices to exactly the
same height. Stonefish refines large triangles on import; a perfectly flat
extreme patch can then produce repeated extreme coordinates during MVAE
initialisation. This script replaces that flat patch with a sub-millimetre
crown having one unique highest vertex.
"""

import os

import bmesh
import bpy


SOURCE = os.path.abspath(
    "src/aquaflow_stonefish/data/bricsbot/"
    "bricsbot_stonefish_physical_proxy_5000.obj"
)
OUTPUT = os.path.abspath(
    "src/aquaflow_stonefish/data/bricsbot/"
    "bricsbot_stonefish_physical_proxy_5000_no_top_bump_mvae_safe.obj"
)

# Source-OBJ coordinates: +X forward, +Y up. These limits select only the
# former rear/top bump. The crown centre is an existing central patch vertex.
REAR_LIMIT_X = -0.08
COVER_HEIGHT_Y = 0.138
CROWN_CENTER_X = -0.142286
CROWN_CENTER_Z = -0.004262
CROWN_X_RADIUS = 0.040
CROWN_Z_RADIUS = 0.012
MAX_DROP = 0.001
CROWN_DROP_SCALE = 0.0008


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=SOURCE)
objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
if not objects:
    raise RuntimeError("Source OBJ did not contain a mesh")

bpy.context.view_layer.objects.active = objects[0]
for obj in objects:
    obj.select_set(True)
if len(objects) > 1:
    bpy.ops.object.join()
obj = bpy.context.object
obj.name = "bricsbot_physical_proxy_5000_no_top_bump_mvae_safe"

changed = []
for vertex in obj.data.vertices:
    if vertex.co.x < REAR_LIMIT_X and vertex.co.y > COVER_HEIGHT_Y:
        dx = (vertex.co.x - CROWN_CENTER_X) / CROWN_X_RADIUS
        dz = (vertex.co.z - CROWN_CENTER_Z) / CROWN_Z_RADIUS
        drop = min(MAX_DROP, CROWN_DROP_SCALE * (dx * dx + dz * dz))
        old_coordinate = tuple(vertex.co)
        vertex.co.y = COVER_HEIGHT_Y - drop
        changed.append((vertex.index, old_coordinate, tuple(vertex.co)))
obj.data.update()

# Moving vertices preserves the validated face connectivity and watertightness.
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")

obj.data.materials.clear()
for layer in list(obj.data.uv_layers):
    obj.data.uv_layers.remove(layer)

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.wm.obj_export(
    filepath=OUTPUT,
    export_materials=False,
    export_uv=False,
    export_normals=True,
    export_triangulated_mesh=True,
    forward_axis="NEGATIVE_Z",
    up_axis="Y",
)

bm = bmesh.new()
bm.from_mesh(obj.data)
bad_edges = sum(1 for edge in bm.edges if len(edge.link_faces) != 2)
zero_faces = sum(1 for face in bm.faces if face.calc_area() < 1e-12)
bm.free()

top = max(vertex.co.y for vertex in obj.data.vertices)
top_count = sum(abs(vertex.co.y - top) < 1e-9 for vertex in obj.data.vertices)
print("OUTPUT", OUTPUT)
print("CROWN_VERTICES", len(changed))
for index, old, new in changed:
    print("CROWN_VERTEX", index, old, "->", new)
print("VERTICES", len(obj.data.vertices))
print("FACES", len(obj.data.polygons))
print("EXTENTS", tuple(round(value, 6) for value in obj.dimensions))
print("TOP_Y", round(top, 9))
print("TOP_VERTEX_COUNT", top_count)
print("OPEN_OR_NONMANIFOLD_EDGES", bad_edges)
print("ZERO_AREA_FACES", zero_faces)
