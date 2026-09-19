"""Pre-refine the MVAE-safe hull so Stonefish's Refine() is a no-op.

Stonefish subdivides triangles larger than three times the average face area
(with an average-area floor of 1e-4 m^2). This script performs conforming
subdivision in Blender until every triangle is below the same threshold.
Shared edges remain shared, so no duplicate midpoint vertices are introduced.
"""

import os

import bmesh
import bpy


SOURCE = os.path.abspath(
    "src/aquaflow_stonefish/data/bricsbot/"
    "bricsbot_stonefish_physical_proxy_5000_no_top_bump_mvae_safe.obj"
)
OUTPUT = os.path.abspath(
    "src/aquaflow_stonefish/data/bricsbot/"
    "bricsbot_stonefish_physical_proxy_5000_no_top_bump_pre_refined_no_normals.obj"
)

MIN_AVERAGE_AREA = 0.0001
SIZE_THRESHOLD = 3.0
MAX_ITERATIONS = 20


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
obj.name = "bricsbot_physical_proxy_5000_no_top_bump_pre_refined"

bm = bmesh.new()
bm.from_mesh(obj.data)
bmesh.ops.triangulate(bm, faces=list(bm.faces))

for iteration in range(MAX_ITERATIONS):
    faces = list(bm.faces)
    average_area = max(
        sum(face.calc_area() for face in faces) / len(faces),
        MIN_AVERAGE_AREA,
    )
    threshold = SIZE_THRESHOLD * average_area
    large_faces = [face for face in faces if face.calc_area() > threshold]
    print(
        "REFINE_ITERATION",
        iteration,
        "FACES",
        len(faces),
        "THRESHOLD",
        threshold,
        "LARGE_FACES",
        len(large_faces),
    )
    if not large_faces:
        break

    # Subdivide every edge of each selected face. BMesh also updates adjacent
    # faces, yielding a conforming manifold rather than T-junctions.
    edges = {edge for face in large_faces for edge in face.edges}
    bmesh.ops.subdivide_edges(
        bm,
        edges=list(edges),
        cuts=1,
        use_grid_fill=True,
    )
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
else:
    raise RuntimeError("Pre-refinement did not converge")

bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-9)
bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
bm.to_mesh(obj.data)
bm.free()
obj.data.update()

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
    export_normals=False,
    export_triangulated_mesh=True,
    forward_axis="NEGATIVE_Z",
    up_axis="Y",
)

bm = bmesh.new()
bm.from_mesh(obj.data)
areas = [face.calc_area() for face in bm.faces]
average_area = max(sum(areas) / len(areas), MIN_AVERAGE_AREA)
threshold = SIZE_THRESHOLD * average_area
bad_edges = sum(1 for edge in bm.edges if len(edge.link_faces) != 2)
zero_faces = sum(area < 1e-12 for area in areas)
print("OUTPUT", OUTPUT)
print("VERTICES", len(obj.data.vertices))
print("FACES", len(obj.data.polygons))
print("AVERAGE_AREA", sum(areas) / len(areas))
print("MAX_AREA", max(areas))
print("STONEFISH_THRESHOLD", threshold)
print("STONEFISH_REFINE_FACES", sum(area > threshold for area in areas))
print("OPEN_OR_NONMANIFOLD_EDGES", bad_edges)
print("ZERO_AREA_FACES", zero_faces)
bm.free()
