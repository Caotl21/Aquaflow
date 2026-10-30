"""Common physical selection rules; zero residual remains a candidate."""
import statistics
import math


def aggregate_metrics(metrics):
    def avg(values):
        return statistics.mean(values) if values and all(value is not None and math.isfinite(value) for value in values) else None
    def error(name):
        return avg([(m.get(name) or {}).get("rmse") for m in metrics])
    return {
        "success_rate": avg([float(m["finish_reason"] == "goal_reached") for m in metrics]),
        "cross_track_rmse_m": error("cross_track_error_m"),
        "yaw_rmse_deg": error("yaw_error_deg"),
        "turn_yaw_rmse_deg": error("turn_yaw_error_deg"),
        "turn_yaw_rate_rmse_radps": error("turn_yaw_rate_error_radps"),
        "final_distance_m": avg([m["final_distance_to_goal_m"] for m in metrics]),
        "duration_s": avg([m["duration_s"] for m in metrics]),
        "comparison_return": avg([m.get("comparison_reward", m["reward"]) for m in metrics]),
        "training_return": avg([m["reward"] for m in metrics]),
    }


def assess_candidate(candidate, baseline, rules):
    violations = []
    if candidate["success_rate"] < baseline["success_rate"]:
        violations.append("success_rate")
    for metric in ("yaw_rmse_deg", "turn_yaw_rmse_deg", "turn_yaw_rate_rmse_radps"):
        if candidate[metric] is None or baseline[metric] is None:
            violations.append("missing_" + metric)
            continue
        allowed = baseline[metric] * (1 + rules["relative_error_tolerance"])
        if candidate[metric] > allowed + 1e-12:
            violations.append(metric)
    for metric, key in (("final_distance_m", "final_distance_tolerance_m"),
                        ("duration_s", "duration_tolerance_s")):
        if candidate[metric] > baseline[metric] + rules[key] + 1e-12:
            violations.append(metric)
    if candidate["cross_track_rmse_m"] is None or baseline["cross_track_rmse_m"] is None:
        violations.append("missing_cross_track_rmse_m")
        improvement = None
    else:
        improvement = baseline["cross_track_rmse_m"] - candidate["cross_track_rmse_m"]
    return dict(guard_passed=not violations, violations=violations,
                cross_track_improvement_m=improvement,
                preferred_to_zero=(not violations and improvement >= rules["minimum_improvement_m"]))


def candidate_key(metrics, assessment):
    cross = metrics["cross_track_rmse_m"]
    common = metrics["comparison_return"]
    return (assessment["guard_passed"], metrics["success_rate"],
            -cross if cross is not None else -float("inf"),
            common if common is not None else -float("inf"))
