#!/usr/bin/env python3
"""Publish the privileged static map for RViz inspection only."""
import math
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def marker(marker_id, name, shape, x, y, z, sx, sy, sz, color):
    msg = Marker()
    msg.header.frame_id = "world_ned"
    msg.ns = "aquaflow_privileged_map"
    msg.id = marker_id
    msg.type = shape
    msg.action = Marker.ADD
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    msg.scale.x = sx
    msg.scale.y = sy
    msg.scale.z = sz
    msg.color.r, msg.color.g, msg.color.b, msg.color.a = color
    msg.lifetime = rospy.Duration(0)
    return msg


def costmap_inflation_cost(clearance, cost_scaling_factor):
    """Mirror the exponential inflation cost used by the teacher A*."""
    if clearance <= 0.0:
        return 254.0
    return 252.0 * math.exp(-cost_scaling_factor * clearance)


def costmap_color(clearance, max_clearance, cost_scaling_factor, alpha):
    """Map the costmap_2d-like cost to a high-to-low cost heatmap palette."""
    cost = costmap_inflation_cost(min(clearance, max_clearance), cost_scaling_factor)
    if cost >= 253.0:
        return ColorRGBA(r=1.0, g=0.0, b=0.0, a=alpha)
    level = max(0.0, min(1.0, cost / 252.0))
    # High cost: red/orange; medium: yellow/green; low: blue.
    if level >= 0.75:
        return ColorRGBA(r=1.0, g=4.0 * (1.0 - level), b=0.0, a=alpha)
    if level >= 0.50:
        return ColorRGBA(r=1.0, g=1.0, b=4.0 * (0.75 - level), a=alpha)
    if level >= 0.25:
        return ColorRGBA(r=0.0, g=1.0, b=4.0 * (0.50 - level), a=alpha)
    return ColorRGBA(r=0.0, g=4.0 * level, b=1.0,
                     a=max(0.12, alpha * (0.25 + 0.75 * level)))


def main():
    rospy.init_node("privileged_map_visualizer")
    obstacles = rospy.get_param("~obstacles", [])
    x_min = float(rospy.get_param("~pool_x_min", -7.0))
    x_max = float(rospy.get_param("~pool_x_max", 7.0))
    y_min = float(rospy.get_param("~pool_y_min", -3.8))
    y_max = float(rospy.get_param("~pool_y_max", 3.8))
    depth = float(rospy.get_param("~pool_depth", 4.0))
    robot_radius = float(rospy.get_param("~robot_radius", 0.365))
    safety_margin = float(rospy.get_param("~safety_margin", 0.25))
    resolution = max(0.05, float(rospy.get_param("~grid_resolution", 0.10)))
    max_clearance = max(resolution, float(rospy.get_param("~heatmap_max_clearance", 2.0)))
    cost_scaling_factor = float(rospy.get_param("~cost_scaling_factor", 3.0))
    heatmap_z = float(rospy.get_param("~heatmap_z", 0.05))
    heatmap_alpha = max(0.05, min(1.0, float(rospy.get_param("~heatmap_alpha", 0.48))))
    show_marker_heatmap = bool(rospy.get_param("~show_marker_heatmap", False))
    show_semantic_costmap = bool(rospy.get_param("~show_semantic_costmap", True))
    show_pool_walls = bool(rospy.get_param("~show_pool_walls", False))
    visual_cost_range = max(resolution, float(rospy.get_param("~visual_cost_range", 0.80)))
    rospy.loginfo("privileged map visualizer loaded %d obstacles", len(obstacles))
    pub = rospy.Publisher("/aquaflow/privileged_map", MarkerArray, queue_size=1, latch=True)
    costmap_pub = rospy.Publisher("/aquaflow/esdf_costmap", OccupancyGrid, queue_size=1, latch=True)

    array = MarkerArray()
    # Muted free-space background and dark physical boundaries.  The cost
    # layers below are deliberately pastel so trajectories remain prominent.
    floor_color = (0.33, 0.43, 0.41, 0.28)
    wall_color = (0.07, 0.04, 0.10, 0.92)
    array.markers.append(marker(0, "floor", Marker.CUBE, 0.0, 0.0, depth + 0.1,
                                x_max - x_min, y_max - y_min, 0.2, floor_color))
    if show_pool_walls:
        array.markers.append(marker(1, "wall_start", Marker.CUBE, x_min - 0.1, 0.0, depth / 2.0,
                                    0.2, y_max - y_min + 0.4, depth, wall_color))
        array.markers.append(marker(2, "wall_end", Marker.CUBE, x_max + 0.1, 0.0, depth / 2.0,
                                    0.2, y_max - y_min + 0.4, depth, wall_color))
        array.markers.append(marker(3, "wall_left", Marker.CUBE, 0.0, y_min - 0.1, depth / 2.0,
                                    x_max - x_min, 0.2, depth, wall_color))
        array.markers.append(marker(4, "wall_right", Marker.CUBE, 0.0, y_max + 0.1, depth / 2.0,
                                    x_max - x_min, 0.2, depth, wall_color))
    else:
        # The topic is latched. Explicitly remove wall markers that may have
        # been published by a previous configuration in the same RViz session.
        for marker_id in range(1, 5):
            delete = Marker()
            delete.header.frame_id = "world_ned"
            delete.ns, delete.id, delete.action = "aquaflow_privileged_map", marker_id, Marker.DELETE
            array.markers.append(delete)

    # Build the same costmap_2d-style exponential inflation field used by A*.
    width = int((x_max - x_min) / resolution) + 1
    height = int((y_max - y_min) / resolution) + 1
    inflation = robot_radius + safety_margin

    def clearance_at(x, y):
        # Match ESDFCostmap.signed_distance(): the four pool-wall inner faces
        # are obstacles too, not merely drawing bounds.
        wall_clearance = min(x - x_min, x_max - x, y - y_min, y_max - y)
        pillar_clearance = min((
            ((x - float(o["x"])) ** 2 + (y - float(o["y"])) ** 2) ** 0.5
            - float(o["radius"]) for o in obstacles), default=float("inf"))
        raw_clearance = min(wall_clearance, pillar_clearance)
        return raw_clearance - inflation

    def raw_distance_at(x, y):
        """Distance before footprint/safety inflation, used only for color bands."""
        wall_distance = min(x - x_min, x_max - x, y - y_min, y_max - y)
        pillar_distance = min((
            ((x - float(o["x"])) ** 2 + (y - float(o["y"])) ** 2) ** 0.5
            - float(o["radius"]) for o in obstacles), default=float("inf"))
        return min(wall_distance, pillar_distance)

    def cube_layer(marker_id, namespace):
        layer = Marker()
        layer.header.frame_id = "world_ned"
        layer.header.stamp = rospy.Time.now()
        layer.ns = namespace
        layer.id = marker_id
        layer.type = Marker.CUBE_LIST
        layer.action = Marker.ADD
        layer.pose.orientation.w = 1.0
        layer.scale.x = layer.scale.y = resolution
        layer.scale.z = 0.026
        return layer

    # Standard ROS costmap message for the RViz Map display.  The continuous
    # 0..254 inflation cost is quantized to OccupancyGrid's 0..100 range;
    # 100 is lethal/inscribed and 0 is free space.
    grid = OccupancyGrid()
    grid.header.frame_id = "world_ned"
    grid.header.stamp = rospy.Time.now()
    grid.info.resolution = resolution
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.position.x = x_min - 0.5 * resolution
    grid.info.origin.position.y = y_min - 0.5 * resolution
    grid.info.origin.position.z = heatmap_z
    grid.info.origin.orientation.w = 1.0
    for iy in range(height):
        y = y_min + iy * resolution
        for ix in range(width):
            x = x_min + ix * resolution
            cost = costmap_inflation_cost(clearance_at(x, y), cost_scaling_factor)
            grid.data.append(100 if cost >= 253.0 else int(round(99.0 * cost / 252.0)))

    if show_semantic_costmap:
        # Layer order and palette: pale violet outer preference field, pink
        # safety margin, then cyan robot-body collision margin.  Physical
        # obstacle/wall cores are the dark markers above.
        outer = cube_layer(6, "aquaflow_semantic_cost")
        safety = cube_layer(7, "aquaflow_semantic_cost")
        footprint = cube_layer(8, "aquaflow_semantic_cost")
        for ix in range(width):
            x = x_min + ix * resolution
            for iy in range(height):
                y = y_min + iy * resolution
                raw_distance = raw_distance_at(x, y)
                point = Point(x=x, y=y, z=heatmap_z)
                if 0.0 < raw_distance <= robot_radius:
                    footprint.points.append(point)
                    footprint.colors.append(ColorRGBA(r=0.76, g=1.0, b=1.0, a=0.82))
                elif robot_radius < raw_distance <= inflation:
                    safety.points.append(point)
                    safety.colors.append(ColorRGBA(r=1.0, g=0.79, b=0.87, a=0.58))
                elif inflation < raw_distance <= inflation + visual_cost_range:
                    ratio = (raw_distance - inflation) / visual_cost_range
                    outer.points.append(point)
                    outer.colors.append(ColorRGBA(r=0.75, g=0.72, b=0.94,
                                                   a=0.30 * (1.0 - ratio) + 0.04))
        array.markers.extend((outer, safety, footprint))

    # Optional 3-D debug rendering of exactly the same costmap.  It is off by
    # default because the RViz Map display is closer to costmap_2d.
    if show_marker_heatmap:
        heatmap = Marker()
        heatmap.header.frame_id = "world_ned"
        heatmap.ns = "aquaflow_privileged_map"
        heatmap.id = 5
        heatmap.type = Marker.CUBE_LIST
        heatmap.action = Marker.ADD
        heatmap.pose.orientation.w = 1.0
        heatmap.scale.x = resolution
        heatmap.scale.y = resolution
        heatmap.scale.z = 0.035
        for ix in range(width):
            x = x_min + ix * resolution
            for iy in range(height):
                y = y_min + iy * resolution
                clearance = clearance_at(x, y)
                heatmap.points.append(Point(x=x, y=y, z=heatmap_z))
                heatmap.colors.append(costmap_color(clearance,
                                                    max_clearance, cost_scaling_factor,
                                                    heatmap_alpha))
        array.markers.append(heatmap)

    for i, obstacle in enumerate(obstacles, start=10):
        radius = float(obstacle["radius"])
        array.markers.append(marker(i, str(obstacle.get("name", i)), Marker.CYLINDER,
                                    float(obstacle["x"]), float(obstacle["y"]), depth / 2.0,
                                    2.0 * radius, 2.0 * radius, depth,
                                    (0.05, 0.05, 0.05, 0.85)))
    pub.publish(array)
    costmap_pub.publish(grid)
    rospy.spin()


if __name__ == "__main__":
    main()
