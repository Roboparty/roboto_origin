# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from isaaclab.app import AppLauncher
import argparse
import numpy as np
from PIL import Image
import os
import coacd
import trimesh

parser = argparse.ArgumentParser(description="Export Terrain")
parser.add_argument("--export_mesh", action='store_true', help="Export combined mesh")
parser.add_argument("--export_meshes", action='store_true', help="Export meshes")
parser.add_argument("--export_hfield", action='store_true', help="Export hfield")
parser.add_argument("--coacd_threshold", type=float, default=0.01, help="Threshold for CoACD decomposition")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator

def decomposite(mesh, base_name, output_dir, threshold=0.01):
    print(f"Decomposing {base_name} with threshold {threshold}...")
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    mesh.process()
    # 运行 CoACD 算法
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=-1,
        preprocess_mode="auto",
        resolution=4000,
        mcts_nodes=20,
        mcts_iterations=150,
        mcts_max_depth=3,
        pca=False,
        merge=True,
        decimate=False,
        max_ch_vertex=256,
        extrude=False,
        seed=42
    )
    
    asset_lines = []
    geom_lines = []
    
    for i, (verts, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        part_filename = f"{base_name}_part_{i}.obj"
        part_path = os.path.join(assets_dir, part_filename)
        part_mesh.export(part_path)
        
        rel_path = os.path.join("assets", part_filename)
        asset_lines.append(f'<mesh name="{base_name}_part_{i}" file="{rel_path}"/>')
        geom_lines.append(f'<geom type="mesh" mesh="{base_name}_part_{i}"/>')
        
    xml_content = [
        f'<mujoco model="{base_name}">',
        '    <asset>',
        *[f'        {line}' for line in asset_lines],
        '    </asset>',
        '    <worldbody>',
        f'        <body name="{base_name}_collision">',
        *[f'            {line}' for line in geom_lines],
        '        </body>',
        '    </worldbody>',
        '</mujoco>'
    ]
    
    xml_filename = f"{base_name}.xml"
    xml_path = os.path.join(output_dir, xml_filename)
    
    with open(xml_path, 'w') as f:
        f.write("\n".join(xml_content))
        
    print(f"Saved decomposed XML to {xml_path}")

if __name__ == "__main__":
    cfg = TerrainGeneratorCfg(
        curriculum=False,
        size=(8.0, 8.0),
        border_width=2.0,
        num_rows=2,
        num_cols=2,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        use_cache=False,
        difficulty_range=(0.6, 0.6),
        seed=1,
        sub_terrains={
            "inv_pyramid_stairs_25": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.1,
                step_height_range=(0.05, 0.2),
                step_width=0.25,
                platform_width=2.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_25": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.1,
                step_height_range=(0.05, 0.2),
                step_width=0.25,
                platform_width=2.0,
                border_width=1.0,
                holes=False,
            ),
            "star": terrain_gen.MeshStarTerrainCfg(
                proportion=0.1, num_bars=12, bar_width_range=(0.25, 0.4), bar_height_range=(2.0, 2.0), platform_width=2.0
            ),
            "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.1,
                stone_height_max=0.0,
                stone_width_range=(0.3, 0.4),
                stone_distance_range=(0.1, 0.2),
                platform_width=2.0,
                border_width=1.0,
                holes_depth=-2.0,
            ),
        },
    )
    terrain_generator = TerrainGenerator(cfg)
    output_dir = "exported_terrains"
    os.makedirs(output_dir, exist_ok=True)

    if args.export_mesh:
        terrain_mesh = terrain_generator.terrain_mesh
        filename = "terrain_combined.obj"
        file_path = os.path.join(output_dir, filename)
        terrain_mesh.export(file_path)
        decomposite(terrain_mesh, "terrain_combined", output_dir)
        print(f"Successfully exported combined mesh to {output_dir}")

    if args.export_meshes:
        terrain_meshes = terrain_generator.terrain_meshes
        for i, mesh in enumerate(terrain_meshes):
            row = i // cfg.num_cols
            col = i % cfg.num_cols
            base_name = f"terrain_{row}_{col}"
            filename = f"terrain_{row}_{col}.obj"
            file_path = os.path.join(output_dir, filename)
            mesh.export(file_path)
            decomposite(mesh, base_name, output_dir)
        print(f"Successfully exported all meshes to {output_dir}")

    if args.export_hfield:
        terrain_mesh = terrain_generator.terrain_mesh
        # 1. 设置采样网格参数
        bounds = terrain_mesh.bounds
        min_x, min_y, min_z = bounds[0]
        max_x, max_y, max_z = bounds[1]

        # 设置分辨率
        res = 0.03

        x_vals = np.arange(min_x, max_x, res)
        y_vals = np.arange(min_y, max_y, res)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        # 2. 准备射线
        z_high = max_z + 1.0
        ray_origins = np.column_stack([grid_x.flatten(), grid_y.flatten(), np.full(grid_x.size, z_high)])
        ray_directions = np.tile([0, 0, -1], (grid_x.size, 1))

        # 3. 执行射线检测
        locations, index_ray, index_tri = terrain_mesh.ray.intersects_location(
            ray_origins=ray_origins, 
            ray_directions=ray_directions
        )

        # 4. 重建高度图
        hfield_flat = np.full(grid_x.size, min_z)

        if len(locations) > 0:
            hit_z = locations[:, 2]
            # 如果同一条射线有多个交点，我们需要取最大值。
            np.maximum.at(hfield_flat, index_ray, hit_z)

        # 恢复为 2D 形状 (Rows=Y, Cols=X)
        hfield_data = hfield_flat.reshape(grid_y.shape)

        # 翻转 Y 轴以匹配图像坐标系
        hfield_data = np.flipud(hfield_data)

        # 5. 归一化并保存为 PNG
        h_min = np.min(hfield_data)
        h_max = np.max(hfield_data)
        h_range = h_max - h_min

        if h_range > 1e-6:
            # 归一化到 [0, 1]
            hfield_norm = (hfield_data - h_min) / h_range
            # 转换为 16-bit 灰度
            hfield_uint16 = (hfield_norm * 65535).astype(np.uint16)
        else:
            hfield_uint16 = np.zeros_like(hfield_data, dtype=np.uint16)

        filename = "terrain_hfield.png"
        file_path = os.path.join(output_dir, filename)
        Image.fromarray(hfield_uint16).save(file_path)
        print(f"Successfully exported hfield image to {output_dir}")

        # 6. XML 输出
        x_half = (max_x - min_x) / 2
        y_half = (max_y - min_y) / 2

        print("\n=== MuJoCo XML Snippet ===")
        print(f'<asset>')
        print(f'    <hfield name="terrain" file="{file_path}" size="{x_half:.4f} {y_half:.4f} {h_range:.4f} {abs(h_min):.4f}"/>')
        print(f'</asset>')
        print(f'<worldbody>')
        print(f'    <geom type="hfield" hfield="terrain" pos="{min_x + x_half:.4f} {min_y + y_half:.4f} 0"/>')
        print(f'</worldbody>')
        print("==========================\n")
        # XML 里机器人 base_link pos_z 需要加上 abs(h_min) 的 offset

    simulation_app.close()