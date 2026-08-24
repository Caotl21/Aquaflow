"""Core algorithms for the privileged static-map teacher; no ROS dependency."""
import heapq
import math


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ESDFCostmap:
    """Analytical 2-D ESDF sampled on a grid for static circular obstacles."""
    def __init__(self, x_min, x_max, y_min, y_max, resolution, obstacles,
                 robot_radius, safety_margin, clearance_cost_weight=None,
                 cost_scaling_factor=None):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.resolution = resolution
        self.obstacles = obstacles
        self.inflation = robot_radius + safety_margin
        # ``clearance_cost_weight`` is retained as a compatibility alias for
        # callers of the previous inverse-clearance implementation.
        if cost_scaling_factor is None:
            cost_scaling_factor = 3.0 if clearance_cost_weight is None else clearance_cost_weight
        self.cost_scaling_factor = float(cost_scaling_factor)
        self.lethal_cost = 254.0
        self.inscribed_cost = 253.0
        self.width = int(math.floor((x_max - x_min) / resolution)) + 1
        self.height = int(math.floor((y_max - y_min) / resolution)) + 1
        self.esdf = [[self.signed_distance(*self.grid_to_world((ix, iy)))
                      for iy in range(self.height)] for ix in range(self.width)]

    def grid_to_world(self, cell):
        return (self.x_min + cell[0] * self.resolution,
                self.y_min + cell[1] * self.resolution)

    def world_to_grid(self, point):
        ix = int(round((point[0] - self.x_min) / self.resolution))
        iy = int(round((point[1] - self.y_min) / self.resolution))
        return (max(0, min(self.width - 1, ix)), max(0, min(self.height - 1, iy)))

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def signed_distance(self, x, y):
        """Distance to the closest pillar or pool wall inner face.

        The map bounds coincide with the pool's collision-wall inner faces.
        Treating them as ESDF obstacles gives the planner the same inflated
        robot/safety clearance from a wall as from a pillar.
        """
        wall_clearance = min(x - self.x_min, self.x_max - x,
                             y - self.y_min, self.y_max - y)
        pillar_clearance = min((math.hypot(x - float(o["x"]), y - float(o["y"]))
                                - float(o["radius"]) for o in self.obstacles),
                               default=float("inf"))
        return min(wall_clearance, pillar_clearance)

    def distance_at_cell(self, cell):
        return self.esdf[cell[0]][cell[1]]

    def free(self, cell):
        return self.in_bounds(cell) and self.distance_at_cell(cell) > self.inflation

    def traversal_cost(self, cell):
        clearance = self.distance_at_cell(cell) - self.inflation
        if clearance <= 0.0:
            return float("inf")
        return 1.0 + self.inflation_cost(clearance)

    def inflation_cost(self, clearance):
        """Costmap_2D-like exponential inflation cost for a free cell.

        ``clearance`` is measured from the robot's inflated footprint boundary.
        A cell at that boundary has the inscribed cost (253), and the cost
        decays exponentially toward the free-space cost (0) as clearance
        increases.  Cells inside the boundary are lethal (254).
        """
        if clearance <= 0.0:
            return self.lethal_cost
        return (self.inscribed_cost - 1.0) * math.exp(
            -self.cost_scaling_factor * clearance)

    def nearest_free(self, cell):
        if self.free(cell):
            return cell
        queue, visited = [cell], {cell}
        while queue:
            current = queue.pop(0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in visited or not self.in_bounds(nxt):
                    continue
                if self.free(nxt):
                    return nxt
                visited.add(nxt)
                queue.append(nxt)
        return None

    def line_is_free(self, a, b):
        aw, bw = self.grid_to_world(a), self.grid_to_world(b)
        steps = max(1, int(math.ceil(distance(aw, bw) / (0.5 * self.resolution))))
        for i in range(steps + 1):
            u = float(i) / float(steps)
            point = (aw[0] + u * (bw[0] - aw[0]), aw[1] + u * (bw[1] - aw[1]))
            if self.signed_distance(*point) <= self.inflation:
                return False
        return True


class GlobalAStarPlanner:
    def __init__(self, costmap):
        self.costmap = costmap

    def plan(self, start_world, goal_world):
        start = self.costmap.nearest_free(self.costmap.world_to_grid(start_world))
        goal = self.costmap.nearest_free(self.costmap.world_to_grid(goal_world))
        if start is None or goal is None:
            return None
        open_heap = [(0.0, 0, start)]
        g_score, parent, serial = {start: 0.0}, {}, 0
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1))
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal:
                cells = [current]
                while current in parent:
                    current = parent[current]
                    cells.append(current)
                return list(reversed(cells))
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not self.costmap.free(nxt):
                    continue
                step = math.sqrt(2.0) if dx and dy else 1.0
                candidate = g_score[current] + step * self.costmap.traversal_cost(nxt)
                if candidate >= g_score.get(nxt, float("inf")):
                    continue
                parent[nxt] = current
                g_score[nxt] = candidate
                serial += 1
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(open_heap, (candidate + heuristic, serial, nxt))
        return None


def shortcut_smooth(costmap, grid_path):
    if not grid_path:
        return []
    result, i = [grid_path[0]], 0
    while i < len(grid_path) - 1:
        j = len(grid_path) - 1
        while j > i + 1 and not costmap.line_is_free(grid_path[i], grid_path[j]):
            j -= 1
        result.append(grid_path[j])
        i = j
    return [costmap.grid_to_world(cell) for cell in result]


def _world_segment_is_free(costmap, start, end):
    """Collision-check a world-coordinate segment against the inflated ESDF."""
    steps = max(1, int(math.ceil(distance(start, end) / (0.25 * costmap.resolution))))
    for i in range(steps + 1):
        u = float(i) / float(steps)
        x = start[0] + u * (end[0] - start[0])
        y = start[1] + u * (end[1] - start[1])
        if costmap.signed_distance(x, y) <= costmap.inflation:
            return False
    return True


def round_corners(costmap, points, radius=0.35, samples_per_corner=8):
    """Replace feasible polyline corners with collision-checked quadratic fillets.

    A* and shortcutting deliberately return a shortest free-space polyline, which
    can contain discontinuous headings.  This function trims each corner and
    inserts a tangent-continuous quadratic Bezier segment.  A candidate is used
    only when every sampled piece stays outside the inflated obstacle boundary.
    """
    if len(points) < 3 or radius <= 0.0:
        return list(points)
    result = [points[0]]
    for index in range(1, len(points) - 1):
        previous, corner, following = points[index - 1], points[index], points[index + 1]
        in_len, out_len = distance(previous, corner), distance(corner, following)
        if in_len < 1e-6 or out_len < 1e-6:
            continue
        in_dir = ((corner[0] - previous[0]) / in_len, (corner[1] - previous[1]) / in_len)
        out_dir = ((following[0] - corner[0]) / out_len, (following[1] - corner[1]) / out_len)
        # Do not alter a straight or near-U-turn section.
        cross = in_dir[0] * out_dir[1] - in_dir[1] * out_dir[0]
        dot = in_dir[0] * out_dir[0] + in_dir[1] * out_dir[1]
        if abs(cross) < 1e-3 or dot < -0.95:
            result.append(corner)
            continue
        trim = min(float(radius), 0.45 * in_len, 0.45 * out_len)
        enter = (corner[0] - in_dir[0] * trim, corner[1] - in_dir[1] * trim)
        leave = (corner[0] + out_dir[0] * trim, corner[1] + out_dir[1] * trim)
        curve = []
        for sample in range(1, max(2, int(samples_per_corner)) + 1):
            u = float(sample) / float(max(2, int(samples_per_corner)))
            one_minus_u = 1.0 - u
            curve.append((one_minus_u * one_minus_u * enter[0] + 2.0 * one_minus_u * u * corner[0] + u * u * leave[0],
                          one_minus_u * one_minus_u * enter[1] + 2.0 * one_minus_u * u * corner[1] + u * u * leave[1]))
        candidate = [enter] + curve
        chain = [result[-1]] + candidate
        if all(_world_segment_is_free(costmap, a, b) for a, b in zip(chain[:-1], chain[1:])):
            result.extend(candidate)
        else:
            # A narrow inflated corridor has no safe curvature margin: keep
            # the original route rather than smoothing into an obstacle.
            result.append(corner)
    result.append(points[-1])
    return result


def path_is_free(costmap, points):
    """Collision-check every segment of a world-coordinate route."""
    return bool(points) and all(_world_segment_is_free(costmap, a, b)
                                for a, b in zip(points[:-1], points[1:]))


def time_parameterize(points, z, max_speed, max_yaw_rate, output_spacing,
                      max_accel=0.12, max_decel=0.18):
    """Arc-length resample a geometric route and assign smooth timing.

    Speed is first limited by curvature/yaw-rate, then passed through forward
    and backward acceleration envelopes.  This avoids the old pointwise speed
    jumps at corners and guarantees the final point is stationary.
    """
    if not points:
        return []
    sampled = []
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, int(math.ceil(distance(a, b) / output_spacing)))
        for i in range(count):
            u = float(i) / float(count)
            sampled.append((a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])))
    sampled.append(points[-1])
    if len(sampled) == 1:
        return [(sampled[0][0], sampled[0][1], z, 0.0, 0.0, 0.0)]
    # Remove numerical duplicates before differentiating the tangent.
    compact = [sampled[0]]
    for p in sampled[1:]:
        if distance(compact[-1], p) > 1e-8:
            compact.append(p)
    sampled = compact
    if len(sampled) == 1:
        return [(sampled[0][0], sampled[0][1], z, 0.0, 0.0, 0.0)]
    cumulative = [0.0]
    for a, b in zip(sampled[:-1], sampled[1:]):
        cumulative.append(cumulative[-1] + distance(a, b))

    yaws = []
    for i, point in enumerate(sampled):
        nxt = sampled[min(i + 1, len(sampled) - 1)]
        prev = sampled[max(i - 1, 0)]
        yaws.append(math.atan2(nxt[1] - prev[1], nxt[0] - prev[0]))
    curvature = [0.0] * len(sampled)
    for i in range(1, len(sampled) - 1):
        ds = max(cumulative[i + 1] - cumulative[i - 1], 1e-6)
        curvature[i] = wrap(yaws[i + 1] - yaws[i - 1]) / ds
    if len(sampled) > 1:
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
    speeds = [min(max_speed, max_yaw_rate / max(abs(k), 1e-6))
              for k in curvature]
    speeds = [max(0.08, v) for v in speeds]
    speeds[-1] = 0.0
    for i in range(1, len(speeds)):
        ds = max(cumulative[i] - cumulative[i - 1], 1e-6)
        speeds[i] = min(speeds[i], math.sqrt(max(0.0, speeds[i - 1] ** 2 + 2.0 * max_accel * ds)))
    for i in range(len(speeds) - 2, -1, -1):
        ds = max(cumulative[i + 1] - cumulative[i], 1e-6)
        speeds[i] = min(speeds[i], math.sqrt(max(0.0, speeds[i + 1] ** 2 + 2.0 * max_decel * ds)))
    times = [0.0]
    for i in range(1, len(sampled)):
        ds = max(cumulative[i] - cumulative[i - 1], 1e-6)
        times.append(times[-1] + ds / max(0.08, 0.5 * (speeds[i - 1] + speeds[i])))
    return [(point[0], point[1], z, yaws[i], speeds[i], times[i])
            for i, point in enumerate(sampled)]


class LocalReferenceSampler:
    """Extract a stable horizon and resample it uniformly by arc length."""
    def __init__(self, horizon_points, spacing=0.20):
        self.horizon_points = horizon_points
        self.spacing = max(0.02, float(spacing))
        self.progress = 0

    def reset(self):
        self.progress = 0

    def sample(self, plan, current_xy):
        if not plan:
            return []
        begin = max(0, self.progress - 2)
        nearest = min(range(begin, len(plan)), key=lambda i: distance(current_xy, plan[i][:2]))
        self.progress = max(self.progress, nearest)
        # Global path point spacing varies at rounded corners. Re-sample the
        # local horizon along arc length so all visible/local-control points
        # are uniformly distributed regardless of global point density.
        source = plan[self.progress:]
        if len(source) == 1:
            return [source[0]] * self.horizon_points
        cumulative = [0.0]
        for a, b in zip(source[:-1], source[1:]):
            cumulative.append(cumulative[-1] + distance(a[:2], b[:2]))
        horizon = [source[0]]
        segment = 0
        target_distance = self.spacing
        while len(horizon) < self.horizon_points and target_distance <= cumulative[-1] + 1e-9:
            while segment < len(source) - 2 and cumulative[segment + 1] < target_distance:
                segment += 1
            a, b = source[segment], source[segment + 1]
            length = max(1e-9, cumulative[segment + 1] - cumulative[segment])
            u = min(1.0, max(0.0, (target_distance - cumulative[segment]) / length))
            yaw_delta = wrap(b[3] - a[3])
            horizon.append((a[0] + u * (b[0] - a[0]),
                            a[1] + u * (b[1] - a[1]),
                            a[2] + u * (b[2] - a[2]),
                            wrap(a[3] + u * yaw_delta),
                            a[4] + u * (b[4] - a[4]),
                            a[5] + u * (b[5] - a[5])))
            target_distance += self.spacing
        while len(horizon) < self.horizon_points:
            last = horizon[-1]
            horizon.append((last[0], last[1], last[2], last[3], 0.0, last[5]))
        return horizon


def point_to_polyline_distance(point, plan):
    if not plan:
        return float("inf")
    return min(distance(point, item[:2]) for item in plan)
