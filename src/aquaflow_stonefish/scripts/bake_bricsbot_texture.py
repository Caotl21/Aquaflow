#!/usr/bin/env python3
"""Bake a multi-material BricsBot DAE/OBJ into one textured OBJ for Stonefish.

Usage (from aquaflow_stonefish):
  blender -b --python scripts/bake_bricsbot_texture.py -- \
    data/bricsbot/bot_shark.obj \
    data/bricsbot/bot_shark_stonefish_baked.obj \
    data/bricsbot/bricsbot_baked.png 0.08 2048

The DAE materials are evaluated by Blender, baked into one atlas, and the
exported OBJ has one material using that atlas.  This avoids relying on
Stonefish's OBJ loader to interpret mtllib/usemtl (it currently does not).
"""
import os
import sys

import bpy


def args():
    if "--" not in sys.argv:
        raise SystemExit("Missing '--'.")
    a = sys.argv[sys.argv.index("--") + 1:]
    if len(a) not in (3, 4, 5):
        raise SystemExit("Usage: INPUT_MESH OUTPUT_OBJ OUTPUT_PNG [DECIMATE_RATIO] [SIZE]")
    source, obj_out, png_out = (os.path.abspath(x) for x in a[:3])
    ratio = float(a[3]) if len(a) >= 4 else 1.0
    size = int(a[4]) if len(a) >= 5 else 2048
    if not os.path.isfile(source) or os.path.splitext(source)[1].lower() not in (".dae", ".obj"):
        raise SystemExit("INPUT_MESH must be an existing .dae or .obj file")
    if not 0.0 < ratio <= 1.0 or size < 256:
        raise SystemExit("DECIMATE_RATIO must be in (0,1], SIZE >= 256")
    os.makedirs(os.path.dirname(obj_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(png_out) or ".", exist_ok=True)
    return source, obj_out, png_out, ratio, size


def select_only(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0] if objects else None


def triangles(objects):
    return sum(max(0, len(p.vertices) - 2) for o in objects for p in o.data.polygons)


def mesh_extent(objects):
    points = [component for obj in objects for vertex in obj.data.vertices
              for component in vertex.co]
    if not points:
        return 0.0
    return max(points) - min(points)


def main():
    source, obj_out, png_out, ratio, size = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    extension = os.path.splitext(source)[1].lower()
    if extension == ".dae":
        # Blender 4 distributions often omit the Collada add-on.  Convert
        # DAE with Assimp first, then use Blender's built-in OBJ importer.
        raise SystemExit(
            "This Blender has no Collada importer. Run `assimp export INPUT.dae INPUT.obj` "
            "and pass the generated OBJ to this script."
        )
    bpy.ops.wm.obj_import(filepath=source)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("Input mesh contains no mesh objects")

    # bot_shark.obj is millimetres while the manually aligned bricsbot.obj is
    # already metres.  Infer that distinction from the imported bounding box.
    # A lab ROV visual asset wider than 10 units is plainly expressed in mm.
    extent_before_scale = mesh_extent(meshes)
    unit_scale = 0.001 if extent_before_scale > 10.0 else 1.0
    for obj in meshes:
        obj.scale = tuple(component * unit_scale for component in obj.scale)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.select_set(False)
        if ratio < 0.999999:
            mod = obj.modifiers.new("stonefish_decimate", "DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = ratio
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.modifier_apply(modifier=mod.name)
            obj.select_set(False)

    before = triangles(meshes)
    select_only(meshes)
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active

    # One UV atlas is sufficient because the final Stonefish look is one image.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Stonefish samples the visual texture directly.  Use an opaque RGB atlas:
    # a transparent bake target makes the otherwise correct colours appear
    # almost black underwater.
    image = bpy.data.images.new("bricsbot_baked", width=size, height=size,
                                alpha=False)
    image.filepath_raw = png_out
    image.file_format = "PNG"

    # Add the bake target to every imported material and make it active.
    for mat in obj.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        node = nodes.get("StonefishBakeTarget") or nodes.new("ShaderNodeTexImage")
        node.name = "StonefishBakeTarget"
        node.image = image
        nodes.active = node
        node.select = True

    scene = bpy.context.scene
    # Diffuse baking is only available in Cycles; use CPU for headless runs.
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    scene.render.bake.use_clear = True
    scene.render.bake.margin = 8
    scene.render.bake.margin_type = "ADJACENT_FACES"
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.target = "IMAGE_TEXTURES"
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    select_only([obj])
    bpy.ops.object.bake(type="DIFFUSE")
    image.save()

    # Replace all imported material slots with one textured Stonefish material.
    baked = bpy.data.materials.new("bricsbot_baked")
    baked.use_nodes = True
    nodes = baked.node_tree.nodes
    links = baked.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.45
    obj.data.materials.clear()
    obj.data.materials.append(baked)
    for poly in obj.data.polygons:
        poly.material_index = 0

    if bpy.app.version >= (4, 0, 0):
        bpy.ops.wm.obj_export(filepath=obj_out, export_selected_objects=True,
                              export_materials=True, export_triangulated_mesh=True)
    else:
        bpy.ops.export_scene.obj(filepath=obj_out, use_selection=True,
                                 use_materials=True, use_triangles=True)
    print("Baked Stonefish asset: %s" % obj_out)
    print("Texture: %s" % png_out)
    print("Triangles: %d -> %d" % (before, triangles([obj])))
    print("Input extent: %.6f; applied unit scale: %.6f" %
          (extent_before_scale, unit_scale))
    print("Output extent: %.6f" % mesh_extent([obj]))


if __name__ == "__main__":
    main()
