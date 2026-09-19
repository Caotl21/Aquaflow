import bpy
import bmesh
import os
from mathutils import Vector

OUT = os.path.abspath('src/aquaflow_stonefish/data/bricsbot/bricsbot_stonefish_primitive_proxy.obj')
# Coordinates follow the baked OBJ frame: +X forward, +Y up, +Z lateral.

bpy.ops.wm.read_factory_settings(use_empty=True)


def rounded_box(name, location, dimensions, bevel):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mod = obj.modifiers.new(name + ' rounded edges', 'BEVEL')
    mod.width = bevel
    mod.segments = 2
    mod.limit_method = 'ANGLE'
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def cylinder(name, location, radius, depth, axis='Y', vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    if axis == 'X':
        obj.rotation_euler[1] = 1.5707963267948966
    elif axis == 'Y':
        obj.rotation_euler[0] = 1.5707963267948966
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return obj


parts = []
# Main lower hull: the largest rounded box preserves the overall footprint.
parts.append(rounded_box('main_hull', (0.0, -0.012, 0.0), (0.382, 0.135, 0.340), 0.035))
# Upper cover: intentionally simplified from the complex curved top shell.
parts.append(rounded_box('top_cover', (-0.006, 0.070, 0.0), (0.326, 0.095, 0.270), 0.034))
# Lower keel / equipment belly.
parts.append(rounded_box('lower_keel', (0.0, -0.082, 0.0), (0.285, 0.040, 0.235), 0.014))
# Small raised mast / antenna block visible above the cover.
parts.append(cylinder('top_mast', (-0.145, 0.162, 0.0), 0.019, 0.065, axis='Y', vertices=16))
# Short rounded side rails stand in for small external brackets without retaining texture.
for z in (-0.145, 0.145):
    parts.append(cylinder('side_rail_' + ('port' if z < 0 else 'starboard'), (0.0, -0.055, z), 0.012, 0.255, axis='X', vertices=12))

# Remove any material/UV dependencies and triangulate every primitive.
for obj in parts:
    obj.data.materials.clear()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)

# Export all named primitives as one OBJ while keeping object boundaries.
for obj in parts:
    obj.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.wm.obj_export(filepath=OUT, export_materials=False, export_uv=False,
                      export_normals=True, export_triangulated_mesh=True,
                      forward_axis='NEGATIVE_Z', up_axis='Y')

# Report dimensions, face count and per-object closure.
all_vertices = []
face_count = 0
bad_edges = 0
for obj in parts:
    all_vertices.extend([obj.matrix_world @ v.co for v in obj.data.vertices])
    face_count += len(obj.data.polygons)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bad_edges += sum(1 for e in bm.edges if len(e.link_faces) != 2)
    bm.free()
lo = Vector((min(v[i] for v in all_vertices) for i in range(3)))
hi = Vector((max(v[i] for v in all_vertices) for i in range(3)))
print('PRIMITIVE_PROXY_OUT', OUT)
print('PRIMITIVE_PARTS', len(parts))
print('PRIMITIVE_FACES', face_count)
print('PRIMITIVE_EXTENTS', tuple(round(hi[i] - lo[i], 6) for i in range(3)))
print('PRIMITIVE_OPEN_OR_NONMANIFOLD_EDGES', bad_edges)
