import open3d as o3d
import numpy as np

def visualize_camera(extrinsics):
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=320.0,
        cy=240.0
    )
    geometries = []
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for extrinsic in extrinsics:
        camera_visualization = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic, extrinsic, scale=0.02
        )
        num_lines = len(camera_visualization.lines)
        camera_visualization.colors = o3d.utility.Vector3dVector(
            [[1,0,0]] * num_lines
        )
        geometries.append(camera_visualization)
        
    for geometry in geometries:
        vis.add_geometry(geometry)  # Add one geometry at a time
    vis.run()
    
def visualize(extrinsics, points, colors):
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=320.0,
        cy=240.0
    )
    geometries = []
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0

    for extrinsic in extrinsics:
        camera_visualization = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic, extrinsic, scale=0.02
        )
        num_lines = len(camera_visualization.lines)
        camera_visualization.colors = o3d.utility.Vector3dVector(
            [[1,0,0]] * num_lines
        )
        geometries.append(camera_visualization)
        
    for geometry in geometries:
        vis.add_geometry(geometry)  # Add one geometry at a time
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(pcd)
    vis.run()