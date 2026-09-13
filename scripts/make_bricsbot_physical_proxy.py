import bpy
import bmesh
import os

SRC = os.path.abspath('src/aquaflow_stonefish/data/bricsbot/bricsbot_stonefish_baked.obj')
OUT = os.path.abspath('src/aquaflow_stonefish/data/bricsbot/bricsbot_stonefish_physical_proxy.obj')

# Start from an empty scene and import the baked visual mesh.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=SRC)
objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
if not objs:
    raise RuntimeError('OBJ did not contain a mesh')

# Join meshes if the exporter ever emits more than one object.
bpy.context.view_layer.objects.active = objs[0]
for o in objs:
    o.select_set(True)
if len(objs) > 1:
    bpy.ops.object.join()
obj = bpy.context.object
obj.name = 'bricsbot_physical_proxy'
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Remove visual-only material slots and consolidate duplicate vertices.
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=1e-6)
bpy.ops.object.mode_set(mode='OBJECT')

remesh = obj.modifiers.new('Closed volume remesh', 'REMESH')
remesh.mode = 'VOXEL'
remesh.voxel_size = 0.003
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=remesh.name)

# Quadratic decimation keeps the silhouette while reducing solver cost.
dec = obj.modifiers.new('Physical proxy decimation', 'DECIMATE')
dec.decimate_type = 'COLLAPSE'
dec.ratio = 0.055
dec.use_collapse_triangulate = True
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=dec.name)

# Ensure all output faces are triangles and recalculate outward normals.
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

# Strip materials and UV/color layers: these are irrelevant to physics.
obj.data.materials.clear()
for layer in list(obj.data.uv_layers):
    obj.data.uv_layers.remove(layer)

# Export an OBJ with the same world coordinates and no MTL/texture dependency.
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.wm.obj_export(filepath=OUT, export_materials=False, export_uv=False,
                      export_normals=True, export_triangulated_mesh=True,
                      forward_axis='NEGATIVE_Z', up_axis='Y')

# Report topology for the caller.
bm = bmesh.new()
bm.from_mesh(obj.data)
bad = sum(1 for e in bm.edges if len(e.link_faces) != 2)
print('PROXY_OUT', OUT)
print('PROXY_VERTICES', len(obj.data.vertices))
print('PROXY_FACES', len(obj.data.polygons))
print('PROXY_EXTENTS', tuple(round(v, 6) for v in obj.dimensions))
print('PROXY_OPEN_OR_NONMANIFOLD_EDGES', bad)
bm.free()
