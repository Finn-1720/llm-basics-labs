# -*- coding: utf-8 -*-
"""实验七 步骤1：生成演示白模场景（球/圆环/倒角立方体/圆柱 + 三点布光）
用法: blender -b -P make_demo_scene.py -- --out out.blend [--res 512]
"""
import argparse
import bpy
import sys


def look_at(obj, target):
    from mathutils import Vector
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy, data.size = energy, size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)
    return obj


def make_mat(name, color, roughness=0.9):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def main():
    # 解析 -- 之后的参数
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/demo_white_model.blend")
    ap.add_argument("--res", type=int, default=512, choices=(512, 768))
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res

    body_mat = make_mat("BodyWhite", (0.92, 0.92, 0.92, 1.0), roughness=0.88)
    floor_mat = make_mat("FloorWhite", (0.86, 0.86, 0.86, 1.0), roughness=0.95)

    # 地面
    bpy.ops.mesh.primitive_plane_add(size=24, location=(0, 0, 0))
    floor = bpy.context.object
    floor.data.materials.append(floor_mat)

    # 球
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.62, segments=64, location=(-0.95, 0, 0.62))
    sphere = bpy.context.object
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(body_mat)

    # 圆环
    bpy.ops.mesh.primitive_torus_add(major_radius=0.55, minor_radius=0.2,
                                     major_segments=64, minor_segments=24,
                                     location=(0.0, -0.15, 0.55))
    torus = bpy.context.object
    bpy.ops.object.shade_smooth()
    torus.data.materials.append(body_mat)

    # 倒角立方体（cube + Bevel modifier）
    bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0.95, 0.05, 0.4))
    cube = bpy.context.object
    cube.data.materials.append(body_mat)
    bevel = cube.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.08
    bevel.segments = 3
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # 圆柱
    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=1.1,
                                        vertices=64, location=(0.0, 1.05, 0.55))
    cyl = bpy.context.object
    bpy.ops.object.shade_smooth()
    cyl.data.materials.append(body_mat)

    # 相机 63mm
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 63
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (4.8, -6.0, 2.8)
    look_at(camera, (0, 0, 0.55))
    scene.camera = camera

    # 三点布光
    add_area_light("Key", (3.0, -3.2, 4.2), 300, 3.5, (0, 0, 0.5))
    add_area_light("Fill", (-3.6, -2.8, 2.2), 80, 5.0, (0, 0, 0.6))
    add_area_light("Rim", (0.5, 3.5, 3.2), 220, 2.2, (0, 0, 0.8))

    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[scene] 已保存白模场景 → {args.out}")


if __name__ == "__main__":
    main()
