#!/usr/bin/env python3
"""Decimate a BricsBot DAE/OBJ visual asset to a Stonefish-friendly OBJ.

Run with Blender 4.x from the package root, for example:

  blender -b --python scripts/convert_bricsbot_mesh.py -- \
    data/bricsbot/bot_shark.obj data/bricsbot/bot_shark_stonefish.obj 0.08

Arguments are: INPUT_MESH OUTPUT_OBJ DECIMATE_RATIO [SCALE].  The default
scale is 0.001 because the supplied Collada asset is in millimetres.  The
script deliberately does not change rotation: verify the vehicle forward axis
in Stonefish before adding any fixed visual orientation offset.
"""
import os
import sys

import bpy


def cli_args():
    if "--" not in sys.argv:
        raise SystemExit("Missing '--'. See this script's header for usage.")
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) not in (3, 4):
        raise SystemExit("Usage: INPUT_MESH OUTPUT_OBJ DECIMATE_RATIO [SCALE]")
    source, destination = map(os.path.abspath, args[:2])
    ratio = float(args[2])
    scale = float(args[3]) if len(args) == 4 else 0.001
    if not os.path.isfile(source):
        raise SystemExit("Input file does not exist: %s" % source)
    if os.path.splitext(source)[1].lower() not in (".dae", ".obj"):
        raise SystemExit("INPUT_MESH must be a .dae or .obj file.")
    if not 0.0 < ratio <= 1.0:
        raise SystemExit("DECIMATE_RATIO must be in (0, 1].")
    if scale <= 0.0:
        raise SystemExit("SCALE must be positive.")
    output_dir = os.path.dirname(destination)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return source, destination, ratio, scale


def triangle_count(objects):
    return sum(max(0, len(face.vertices) - 2)
               for obj in objects for face in obj.data.polygons)


def select_only(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0] if objects else None


def apply_object_scale(obj, scale):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.scale = tuple(component * scale for component in obj.scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def decimate(obj, ratio):
    if ratio >= 0.999999:
        return
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new(name="stonefish_decimate", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def export_obj(destination, mesh_objects):
    select_only(mesh_objects)
    if bpy.app.version >= (4, 0, 0):
        bpy.ops.wm.obj_export(filepath=destination,
                              export_selected_objects=True,
                              export_materials=True,
                              export_triangulated_mesh=True)
    else:
        bpy.ops.export_scene.obj(filepath=destination,
                                 use_selection=True,
                                 use_materials=True,
                                 use_triangles=True)


def main():
    source, destination, ratio, scale = cli_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    extension = os.path.splitext(source)[1].lower()
    if extension == ".dae":
        bpy.ops.wm.collada_import(filepath=source)
    else:
        bpy.ops.wm.obj_import(filepath=source)
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise SystemExit("No mesh objects found in: %s" % source)
    before = triangle_count(mesh_objects)
    for obj in mesh_objects:
        apply_object_scale(obj, scale)
        decimate(obj, ratio)
    after = triangle_count(mesh_objects)
    export_obj(destination, mesh_objects)
    print("BricsBot mesh conversion complete")
    print("  input:     %s" % source)
    print("  output:    %s" % destination)
    print("  objects:   %d" % len(mesh_objects))
    print("  triangles: %d -> %d (ratio %.3f)" % (before, after, ratio))
    print("  scale:     %.6f" % scale)


if __name__ == "__main__":
    main()
