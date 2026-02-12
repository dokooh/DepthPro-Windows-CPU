# LICENSE_2 applies to this file
# Author JZ from LatteByte.ai 2024

import os
import numpy as np
import open3d as o3d  # Open3D for 3D visualization
import matplotlib.pyplot as plt
import csv

INPUT_GLB_PATH = "/kaggle/input/digsite-ply/glbscene_50_All_maskbFalse_maskwFalse_camTrue_skyFalse_predDepthmap_and_Camera_Branch.glb"


def _get_point_cloud_from_glb(glb_path: str) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(glb_path)
    if mesh.is_empty():
        raise ValueError(f"Could not read a valid mesh from: {glb_path}")

    point_cloud = mesh.sample_points_poisson_disk(number_of_points=200000)
    if len(point_cloud.points) == 0:
        raise ValueError("Point cloud sampling returned zero points.")
    return point_cloud


def _save_point_cloud_plot(point_cloud, output_png_path: str):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    if points.shape[0] > 50000:
        idx = np.random.choice(points.shape[0], 50000, replace=False)
        points = points[idx]
        colors = colors[idx] if colors.shape[0] == idx.shape[0] else colors

    if colors.size == 0 or colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 0.6, dtype=np.float32)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=1,
        linewidths=0,
        alpha=0.9,
    )
    ax.set_title("Point Cloud Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close(fig)


def _cluster_objects(point_cloud: o3d.geometry.PointCloud):
    points = np.asarray(point_cloud.points)
    if points.shape[0] == 0:
        return np.array([]), []

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    scene_diag = float(np.linalg.norm(bounds_max - bounds_min))

    eps = max(scene_diag * 0.01, 0.02)
    min_points = 80
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))

    object_rows = []
    object_id = 0
    for cluster_id in np.unique(labels):
        if cluster_id < 0:
            continue

        mask = labels == cluster_id
        cluster_pts = points[mask]
        if cluster_pts.shape[0] < min_points:
            continue

        object_id += 1
        min_xyz = cluster_pts.min(axis=0)
        max_xyz = cluster_pts.max(axis=0)
        extent = max_xyz - min_xyz
        volume_m3 = float(extent[0] * extent[1] * extent[2])

        object_rows.append(
            {
                "object_id": object_id,
                "cluster_label": int(cluster_id),
                "num_points": int(cluster_pts.shape[0]),
                "x_min": float(min_xyz[0]),
                "y_min": float(min_xyz[1]),
                "z_min": float(min_xyz[2]),
                "x_max": float(max_xyz[0]),
                "y_max": float(max_xyz[1]),
                "z_max": float(max_xyz[2]),
                "width_x": float(extent[0]),
                "height_y": float(extent[1]),
                "depth_z": float(extent[2]),
                "bbox_volume_m3": volume_m3,
            }
        )

    return labels, object_rows


def _save_clustered_point_cloud(point_cloud, labels, output_path: str):
    points = np.asarray(point_cloud.points)
    max_label = int(labels.max()) if labels.size > 0 else -1

    if max_label >= 0:
        cmap = plt.get_cmap("tab20")
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)
        for idx in range(points.shape[0]):
            label = labels[idx]
            if label < 0:
                colors[idx] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            else:
                colors[idx] = np.array(cmap(label % 20)[:3], dtype=np.float32)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, point_cloud)


def main():
    current_dir = os.path.dirname(__file__)
    debug_dir = os.path.join(current_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    glb_path = INPUT_GLB_PATH
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found at path: {glb_path}")

    point_cloud = _get_point_cloud_from_glb(glb_path)

    # Save base point cloud prediction output
    base_pcd_path = os.path.join(debug_dir, "point_cloud_prediction.ply")
    o3d.io.write_point_cloud(base_pcd_path, point_cloud)

    # Save visualization
    vis_png_path = os.path.join(debug_dir, "point_cloud_visualization.png")
    _save_point_cloud_plot(point_cloud, vis_png_path)

    # Detect objects and measure 3D sizes
    labels, object_rows = _cluster_objects(point_cloud)

    # Save clustered point cloud
    clustered_pcd_path = os.path.join(debug_dir, "point_cloud_clustered.ply")
    _save_clustered_point_cloud(point_cloud, labels, clustered_pcd_path)

    # Save object size measurements
    object_csv_path = os.path.join(debug_dir, "object_sizes_3d.csv")
    headers = [
        "object_id",
        "cluster_label",
        "num_points",
        "x_min",
        "y_min",
        "z_min",
        "x_max",
        "y_max",
        "z_max",
        "width_x",
        "height_y",
        "depth_z",
        "bbox_volume_m3",
    ]
    with open(object_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(object_rows)

    print(f"Input GLB: {glb_path}")
    print(f"Saved point cloud prediction: {base_pcd_path}")
    print(f"Saved visualization: {vis_png_path}")
    print(f"Saved clustered point cloud: {clustered_pcd_path}")
    print(f"Saved object measurements: {object_csv_path}")
    print(f"Objects measured: {len(object_rows)}")


if __name__ == "__main__":
    main()
