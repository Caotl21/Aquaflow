"""Remove the small rear/top bump from the 5000-face BricsBot proxy.

Run from the catkin workspace root:

    blender --background --python scripts/remove_bricsbot_5000_top_bump.py

The baked mesh uses +X forward and +Y up.  The unwanted appendage is the
small cluster near x=-0.17 m whose top reaches y=0.194 m.  Flattening only
that cluster at the surrounding cover height preserves every other part of
the already validated proxy, including its watertight topology.
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
    "bricsbot_stonefish_physical_proxy_5000_no_top_bump.obj"
)

# Bounds of the bump in the source OBJ coordinate system.
REAR_LIMIT_X = -0.08
COVER_HEIGHT_Y = 0.138


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
obj.name = "bricsbot_physical_proxy_5000_no_top_bump"

changed = []
for vertex in obj.data.vertices:
    if vertex.co.x < REAR_LIMIT_X and vertex.co.y > COVER_HEIGHT_Y:
        changed.append((vertex.index, tuple(vertex.co)))
        vertex.co.y = COVER_HEIGHT_Y
obj.data.update()

# Preserve topology and only refresh face orientation after moving vertices.
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
bm.free()

print("OUTPUT", OUTPUT)
print("FLATTENED_VERTICES", len(changed))
print("VERTICES", len(obj.data.vertices))
print("FACES", len(obj.data.polygons))
print("EXTENTS", tuple(round(value, 6) for value in obj.dimensions))
print("OPEN_OR_NONMANIFOLD_EDGES", bad_edges)
