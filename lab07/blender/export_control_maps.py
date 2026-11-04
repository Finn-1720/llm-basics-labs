# -*- coding: utf-8 -*-
"""实验七 步骤2：无头导出 depth / normal / clay 三张控制图
用法: blender -b assets/demo_white_model.blend -P export_control_maps.py -- --outdir output --name demo --res 512
（Blender 5.x 适配版：合成器经 Viewer 节点 + save_render 写出，绕开已移除的 MixRGB / File Output API）
"""
import argparse
import os
import sys

import bpy
import numpy as np


def scene_depth_range(scene, camera):
    """用相机视锥内可见顶点估算深度范围，避免地面远角拉爆动态范围。"""
    from bpy_extras.object_utils import world_to_camera_view
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc = camera.matrix_world.translation
    distances = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        for vert in obj_eval.data.vertices:
            world_co = obj_eval.matrix_world @ vert.co
            ndc = world_to_camera_view(scene, camera, world_co)
            if ndc.z <= 0 or not (0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0):
                continue
            distances.append((world_co - cam_loc).length)
    if not distances:
        return 0.1, 100.0
    near = float(np.percentile(distances, 2))
    far = float(np.percentile(distances, 98))
    return near, (far if far > near else near + 1.0)


def make_clay_material():
    mat = bpy.data.materials.new("ClayOverride")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.94, 0.94, 0.94, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.92
    return mat


def build_viewer_tree(scene, mode):
    """建合成器树：RLayers → (depth 反相 / normal 曲线映射) → GroupOutput。mode: depth|normal
    Blender 5.x 合成器输出节点为 NodeGroupOutput（旧版 Composite/File Output 已移除）"""
    scene.use_nodes = True
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("Compositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
    tree.nodes.clear()
    # Blender 5.x：NodeGroupOutput 需 interface 显式定义输出 socket，否则 pass 数据被丢弃
    if len(tree.interface.items_tree) == 0:
        tree.interface.new_socket("Out", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = tree.nodes.new("CompositorNodeRLayers")
    out = tree.nodes.new("NodeGroupOutput")
    if mode == "depth":
        # depth: mist 0..1 (near..far) -> 反相 -> 近白远黑
        invert = tree.nodes.new("CompositorNodeInvert")
        invert.inputs["Fac"].default_value = 1.0
        tree.links.new(rl.outputs["Mist"], invert.inputs[1])
        tree.links.new(invert.outputs[0], out.inputs[0])
    else:
        # normal: n(-1..1) -> y=0.5x+0.5
        # ColorBalance 的 Offset/Power/Slope 模式: out=(in*slope+offset)^power，
        # 取 slope=0.5、offset=0.5、power=1.0 即得 0.5x+0.5
        # （MixRGB 已移除；CurveRGB/BrightContrast 在 Blender 5 合成器会灰度化，弃用）
        cb = tree.nodes.new("CompositorNodeColorBalance")
        cb.inputs["Type"].default_value = "Offset/Power/Slope (ASC-CDL)"
        cb.inputs["Offset"].default_value = 0.5
        cb.inputs["Power"].default_value = 1.0
        cb.inputs["Slope"].default_value = 0.5
        tree.links.new(rl.outputs["Normal"], cb.inputs[0])
        tree.links.new(cb.outputs[0], out.inputs[0])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--name", default="demo")
    ap.add_argument("--res", type=int, default=512, choices=(512, 768))
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res

    view_layer = bpy.context.view_layer
    view_layer.use_pass_mist = True
    view_layer.use_pass_normal = True

    # Mist falloff 按可见顶点 2%/98% 百分位
    near, far = scene_depth_range(scene, scene.camera)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    mist = scene.world.mist_settings
    mist.use_mist = True
    mist.start, mist.depth = near, far - near
    mist.falloff = "LINEAR"
    print(f"[export] mist 范围 near={near:.3f} far={far:.3f}")

    # 渲染 1：depth（Mist 反相 → GroupOutput → 场景 filepath）
    build_viewer_tree(scene, "depth")
    scene.render.filepath = os.path.join(args.outdir, f"{args.name}_depth.png")
    bpy.ops.render.render(write_still=True)
    print(f"[export] depth 已写出")

    # 渲染 2：normal（×0.5+0.5 → GroupOutput）
    build_viewer_tree(scene, "normal")
    scene.render.filepath = os.path.join(args.outdir, f"{args.name}_normal.png")
    bpy.ops.render.render(write_still=True)
    print(f"[export] normal 已写出")

    # 渲染 3：clay（白模材质覆盖，直接渲染）
    view_layer.material_override = make_clay_material()
    scene.render.filepath = os.path.join(args.outdir, f"{args.name}_clay.png")
    bpy.ops.render.render(write_still=True)
    print(f"[export] clay 已写出 → {scene.render.filepath}")


if __name__ == "__main__":
    main()
