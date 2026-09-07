#!/usr/bin/env python3
"""Measure and freeze the S0/S1 baseline artifacts of a running simulation.

Writes three files under ``artifacts/``:

* ``stonefish_env_report.json``  (S0) container/ROS/Stonefish versions, package
  git state, GPU, the resolved scenario tree, and the declared actuator/sensor
  facts parsed out of the ``.scn`` files.
* ``ros_topic_schema.json``      (S1) per-topic message type, md5, recursively
  expanded field list, live ``frame_id`` and a sample summary.
* ``ros_topic_rates.json``       (S1) per-topic measured publish rate, inter
  arrival statistics, stamp monotonicity and stamp-vs-wall-clock lag.

Everything except the static environment section is *measured* from a live
master; nothing here is hand-written.  A sensor that the scenario declares but
that publishes nothing during the capture window (a commented-out camera in a
headless run, for instance) is reported under ``missing_expected_topics`` so a
partial capture can never be mistaken for the frozen full schema.

Usage::

    roslaunch hofa_mpc_ros simulation.launch headless:=true   # in one terminal
    rosrun aquaflow_stonefish export_baseline_artifacts.py --duration 20
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

import rospy

PRIMITIVE_TYPES = {
    "bool", "byte", "char", "int8", "uint8", "int16", "uint16", "int32",
    "uint32", "int64", "uint64", "float32", "float64", "string", "time",
    "duration",
}

# Bookkeeping topics that describe the ROS plumbing rather than the robot.
DEFAULT_TOPIC_EXCLUDE = (
    re.compile(r"^/rosout(_agg)?$"),
    re.compile(r"/parameter_(descriptions|updates)$"),
)

# Topics that are advertised but legitimately never publish, with the reason.
# These are recorded like any other topic and are not treated as a gap, since
# reporting them as missing data would train the reader to ignore the one
# signal that is supposed to mean something.
EXPECTED_SILENT = (
    (re.compile(r"/compressedDepth$"),
     "compressed_depth_image_transport advertises a compressedDepth topic for "
     "every image topic but only publishes for depth encodings (16UC1/32FC1); "
     "this camera is rgb8, so the topic stays silent by design"),
)


def expected_silence_reason(name):
    """Return why a topic is expected to stay silent, or None if it is not."""
    for pattern, reason in EXPECTED_SILENT:
        if pattern.search(name):
            return reason
    return None


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def run_command(command):
    """Return stripped stdout of ``command`` or None when it is unavailable."""
    return run_command_checked(command)[0]


def run_command_checked(command):
    """Run ``command`` and return ``(stdout, error)``.

    Unlike :func:`run_command` this keeps the failure reason, so a caller that
    is recording provenance can say *why* it came up empty instead of emitting
    a plausible-looking negative result.
    """
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        raw_out, raw_err = process.communicate()
    except OSError as error:
        return None, str(error)
    out = raw_out.decode("utf-8", "replace").strip()
    err = raw_err.decode("utf-8", "replace").strip()
    if process.returncode != 0:
        return None, err or ("exit status %d" % process.returncode)
    return (out or None), None


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# S0: static environment
# ---------------------------------------------------------------------------

def git_state(path):
    """Record commit and dirtiness so a capture can be tied back to source.

    ``safe.directory=*`` is passed per invocation because this workspace mixes
    root-owned and user-owned package directories: git's dubious-ownership
    check rejects whichever set the caller does not own, so running as either
    user would otherwise silently drop half the provenance.  The override
    applies to this one process and writes no config file.
    """
    git = ["git", "-c", "safe.directory=*", "-C", path]
    commit, error = run_command_checked(git + ["rev-parse", "HEAD"])
    if commit is None:
        # Report the reason: an empty provenance record that looks deliberate
        # is worse than no record at all.
        return {"path": path, "tracked": False, "error": error}
    status, status_error = run_command_checked(
        git + ["status", "--porcelain", "--", path])
    record = {"path": path, "tracked": True, "commit": commit,
              "dirty": bool(status),
              "dirty_files": status.splitlines() if status else []}
    if status_error:
        record["error"] = status_error
    return record


def stonefish_info(search_roots):
    """Locate the Stonefish sources and read the version out of CMakeLists."""
    found = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        entry = {"path": root, "version": None}
        cmake = os.path.join(root, "CMakeLists.txt")
        if os.path.isfile(cmake):
            with open(cmake, "r", errors="replace") as handle:
                match = re.search(r"project\s*\(\s*Stonefish\s+VERSION\s+([0-9.]+)",
                                  handle.read())
            if match:
                entry["version"] = match.group(1)
        found.append(entry)
    return found


def gpu_info():
    """Record the renderer actually in use; a headless run has no GL context."""
    info = {}
    smi = run_command(["nvidia-smi",
                       "--query-gpu=name,driver_version,memory.total",
                       "--format=csv,noheader"])
    if smi:
        info["nvidia_smi"] = smi.splitlines()
    glx = run_command(["glxinfo", "-B"])
    if glx:
        for line in glx.splitlines():
            for key, label in (("OpenGL renderer string", "opengl_renderer"),
                               ("OpenGL version string", "opengl_version"),
                               ("OpenGL core profile version string", "opengl_core_version")):
                if line.strip().startswith(key):
                    info[label] = line.split(":", 1)[1].strip()
    return info


def parse_scenario(path, package_paths, vehicle_name, seen=None):
    """Recursively read a ``.scn`` and report its actuators and sensors.

    Comments are kept by the parser so that a sensor which is present in the
    file but commented out (a camera disabled for a headless run) is reported
    with ``enabled: false`` instead of silently vanishing from the record.
    """
    seen = seen if seen is not None else set()
    real = os.path.realpath(path)
    if real in seen or not os.path.isfile(real):
        return {"file": path, "exists": os.path.isfile(real), "error": "missing_or_cycle"}
    seen.add(real)

    entry = {"file": real, "sha256": sha256_of(real),
             "actuators": [], "sensors": [], "includes": []}
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    try:
        with open(real, "r", errors="replace") as handle:
            parser.feed(handle.read())
        root = parser.close()
    except ET.ParseError as error:
        entry["error"] = "parse_error: %s" % error
        return entry

    def substitute(text):
        if text is None:
            return None
        text = text.replace("$(arg vehicle_name)", vehicle_name)
        for name, pkg_path in package_paths.items():
            text = text.replace("$(find %s)" % name, pkg_path)
        return text

    def record_element(element, enabled):
        if element.tag == "actuator":
            specs = element.find("specs")
            propeller = element.find("propeller")
            thrust_model = element.find("thrust_model")
            entry["actuators"].append({
                "name": element.get("name"),
                "type": element.get("type"),
                "enabled": enabled,
                "specs": dict(specs.attrib) if specs is not None else None,
                "right_handed": (propeller.get("right") if propeller is not None else None),
                "thrust_model": (thrust_model.get("type") if thrust_model is not None else None),
            })
        elif element.tag == "sensor":
            publisher = element.find("ros_publisher")
            specs = element.find("specs")
            entry["sensors"].append({
                "name": element.get("name"),
                "type": element.get("type"),
                "enabled": enabled,
                "declared_rate_hz": (float(element.get("rate"))
                                     if element.get("rate") else None),
                "topic": substitute(publisher.get("topic")) if publisher is not None else None,
                "origin": (element.find("origin").attrib
                           if element.find("origin") is not None else None),
                "specs": dict(specs.attrib) if specs is not None else None,
            })

    def walk(node, enabled):
        for child in node:
            if child.tag is ET.Comment:
                # Re-parse the comment body: a disabled sensor block is still a
                # declaration we must report, not an untracked hole.
                fragment = (child.text or "").strip()
                if "<sensor" not in fragment and "<actuator" not in fragment:
                    continue
                try:
                    walk(ET.fromstring("<commented>%s</commented>" % fragment), False)
                except ET.ParseError:
                    entry["sensors"].append({"name": None, "enabled": False,
                                             "raw_comment": fragment[:200]})
                continue
            if child.tag == "include":
                included = substitute(child.get("file"))
                if included:
                    entry["includes"].append(
                        parse_scenario(included, package_paths, vehicle_name, seen))
                continue
            record_element(child, enabled)
            walk(child, enabled)

    walk(root, True)
    return entry


def collect_env_report(args, package_paths, scenario_tree, topics):
    ros_root = os.environ.get("ROS_ROOT", "")
    return {
        "generated_at_utc": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "container_hint": os.environ.get("CONTAINER_NAME") or os.environ.get("HOSTNAME"),
            "kernel": run_command(["uname", "-sr"]),
            "os_release": run_command(["lsb_release", "-ds"]),
        },
        "ros": {
            "distro": os.environ.get("ROS_DISTRO"),
            "version": run_command(["rosversion", "-d"]),
            "root": ros_root,
            "master_uri": os.environ.get("ROS_MASTER_URI"),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
        },
        "stonefish": stonefish_info([
            "/home/bricsbot/simulators/stonefish-1.6",
            "/home/bricsbot/simulators/stonefish",
        ]),
        "packages": {name: git_state(path) for name, path in sorted(package_paths.items())},
        "gpu": gpu_info(),
        "scenario": scenario_tree,
        "topics_present": sorted(topics),
    }


# ---------------------------------------------------------------------------
# S1: live schema and rates
# ---------------------------------------------------------------------------

def get_message_class(type_string):
    """Resolve a message class, or None when the type cannot be introspected.

    The master reports ``*`` for a topic registered with a wildcard type (an
    ``AnyMsg`` subscriber such as ``rosbag record -a`` or ``rostopic echo``),
    and genpy raises for any name without a package.  A single such topic must
    not abort the whole capture, so failures degrade to None and the caller
    records the topic as unresolved.
    """
    if not type_string or "/" not in type_string:
        return None
    import roslib.message
    try:
        return roslib.message.get_message_class(type_string)
    except Exception:
        return None


def expand_fields(type_string, prefix="", depth=0, max_depth=8):
    """Flatten a message definition into dotted ``name -> type`` entries.

    Array fields are expanded once, with ``[]`` in the path, so that a Path's
    pose stamps and frame ids appear in the record.  ``uint8[]`` payloads (image
    data) stay leaves.
    """
    base = type_string.split("[")[0]
    is_array = "[" in type_string
    if base in PRIMITIVE_TYPES or depth >= max_depth:
        return [{"name": prefix or ".", "type": type_string}]
    message_class = get_message_class(base)
    if message_class is None:
        return [{"name": prefix or ".", "type": type_string, "unresolved": True}]
    fields = []
    for name, slot_type in zip(message_class.__slots__, message_class._slot_types):
        child_prefix = "%s%s%s" % (prefix, "[]." if is_array else ("." if prefix else ""), name)
        fields.extend(expand_fields(slot_type, child_prefix, depth + 1, max_depth))
    return fields


def summarize_sample(message):
    """Pull the few concrete values a data pipeline actually has to know."""
    summary = {}
    if hasattr(message, "header"):
        summary["header.frame_id"] = message.header.frame_id
        summary["header.stamp_is_zero"] = (message.header.stamp.to_sec() == 0.0)
    for slot in getattr(message, "__slots__", []):
        if slot == "header":
            continue
        value = getattr(message, slot)
        if isinstance(value, (bool, int, float)):
            summary[slot] = value
        elif isinstance(value, str):
            summary[slot] = value
        elif isinstance(value, (list, tuple)):
            summary["len(%s)" % slot] = len(value)
        elif hasattr(value, "__slots__") and hasattr(value, "header"):
            summary["%s.header.frame_id" % slot] = value.header.frame_id
    return summary


class TopicProbe(object):
    """Subscribe to one topic and accumulate arrival and stamp statistics."""

    def __init__(self, name, type_string):
        self.name = name
        self.type_string = type_string
        self.wall_times = []
        self.stamps = []
        self.tf_pairs = set()
        self.first_message = None
        message_class = get_message_class(type_string)
        self.unresolved = message_class is None
        # A 640x480 RGB frame is ~0.9 MB; a deep queue on an image topic would
        # buffer hundreds of megabytes for a callback that only appends floats.
        queue_size = 5 if "Image" in type_string else 200
        self.subscriber = (rospy.Subscriber(name, message_class, self.callback,
                                            queue_size=queue_size)
                           if message_class is not None else None)
        if self.unresolved:
            rospy.logwarn("cannot resolve message type %r of %s; recorded as "
                          "unresolved and not sampled", type_string, name)

    def callback(self, message):
        self.wall_times.append(rospy.get_time())
        stamp = None
        if hasattr(message, "header"):
            stamp = message.header.stamp.to_sec()
        self.stamps.append(stamp)
        if self.first_message is None:
            self.first_message = message
        # /tf carries the frame graph; record it, since S1 has to freeze frames.
        for transform in getattr(message, "transforms", []) or []:
            self.tf_pairs.add((transform.header.frame_id, transform.child_frame_id))

    def unsubscribe(self):
        if self.subscriber is not None:
            self.subscriber.unregister()

    def rate_record(self, window_s, declared_rate_hz):
        count = len(self.wall_times)
        record = {"type": self.type_string, "count": count,
                  "window_s": round(window_s, 3),
                  "declared_rate_hz": declared_rate_hz}
        if self.unresolved:
            record["unresolved_type"] = True
            record["measured_rate_hz"] = None
            record["note"] = "message type could not be resolved; not sampled"
            return record
        if count < 2:
            record["measured_rate_hz"] = None
            record["note"] = "fewer than two messages during the capture window"
            return record
        deltas = [b - a for a, b in zip(self.wall_times[:-1], self.wall_times[1:])]
        mean_dt = sum(deltas) / len(deltas)
        variance = sum((d - mean_dt) ** 2 for d in deltas) / len(deltas)
        record.update({
            "measured_rate_hz": round(1.0 / mean_dt, 4) if mean_dt > 0 else None,
            "interarrival_s": {"mean": round(mean_dt, 6),
                               "min": round(min(deltas), 6),
                               "max": round(max(deltas), 6),
                               "std": round(variance ** 0.5, 6)},
        })
        if declared_rate_hz and mean_dt > 0:
            record["measured_over_declared"] = round(
                (1.0 / mean_dt) / declared_rate_hz, 4)
        stamps = [s for s in self.stamps if s is not None]
        if stamps:
            record["stamp_monotonic"] = all(b >= a for a, b in zip(stamps[:-1], stamps[1:]))
            lags = [w - s for w, s in zip(self.wall_times, self.stamps) if s is not None]
            record["stamp_lag_s"] = {"mean": round(sum(lags) / len(lags), 6),
                                     "min": round(min(lags), 6),
                                     "max": round(max(lags), 6)}
        else:
            record["stamp_monotonic"] = None
            record["note"] = "message has no header stamp; align on arrival time"
        return record

    def schema_record(self, declared):
        record = {"type": self.type_string,
                  "declared_by_scenario": declared}
        if self.unresolved:
            record["unresolved_type"] = True
            record["fields"] = None
            record["sample"] = None
            record["note"] = ("master reports a wildcard or unknown message "
                              "type; schema could not be introspected")
            return record
        record["fields"] = expand_fields(self.type_string)
        message_class = get_message_class(self.type_string)
        if message_class is not None:
            record["md5sum"] = message_class._md5sum
        if self.first_message is not None:
            record["sample"] = summarize_sample(self.first_message)
        else:
            record["sample"] = None
            record["note"] = "no message received during the capture window"
        if self.tf_pairs:
            record["observed_frames"] = sorted(
                {"%s -> %s" % (parent, child) for parent, child in self.tf_pairs})
        return record


def declared_topic_map(scenario_tree):
    """Map every scenario-declared sensor topic to its rate and enabled flag."""
    declared = {}

    def walk(node):
        for sensor in node.get("sensors", []):
            topic = sensor.get("topic")
            if not topic:
                continue
            declared[topic] = {"sensor": sensor.get("name"),
                               "sensor_type": sensor.get("type"),
                               "enabled": sensor.get("enabled"),
                               "declared_rate_hz": sensor.get("declared_rate_hz")}
        for child in node.get("includes", []):
            walk(child)

    walk(scenario_tree)
    return declared


def topic_is_excluded(name, include_all, extra_patterns=()):
    if any(pattern.search(name) for pattern in extra_patterns):
        return True
    if include_all:
        return False
    return any(pattern.search(name) for pattern in DEFAULT_TOPIC_EXCLUDE)


def resolve_declared_topics(declared, present):
    """Match scenario-declared sensor topics against the live topic list.

    A ``ros_publisher topic`` is not always a topic name.  ROSScenarioParser
    treats it as a *prefix* for image-producing sensors, advertising
    ``<topic>/image_color`` and ``<topic>/camera_info`` instead of ``<topic>``
    itself, so an exact-match test would report a perfectly healthy camera as
    missing.  Accept either an exact match or any child under ``<topic>/`` and
    record which live topics satisfied each declaration.

    Returns ``(satisfied, missing)`` where ``satisfied`` maps a declared topic
    to the sorted list of live topics that realize it.
    """
    satisfied, missing = {}, []
    for topic in sorted(declared):
        matches = sorted(name for name in present
                         if name == topic or name.startswith(topic + "/"))
        if matches:
            satisfied[topic] = matches
        else:
            missing.append(topic)
    return satisfied, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=20.0,
                        help="rate measurement window in seconds (default: 20)")
    parser.add_argument("--vehicle-name", default="bricsbot")
    parser.add_argument("--scene", default="random_pillars",
                        choices=("empty_pool", "single_obstacle", "random_pillars"))
    parser.add_argument("--output-root", default=None,
                        help="artifact directory (default: <workspace>/artifacts)")
    parser.add_argument("--include-all-topics", action="store_true",
                        help="also record /rosout and dynamic_reconfigure topics")
    parser.add_argument("--exclude", action="append", default=[], metavar="REGEX",
                        help="skip topics matching this regex; repeatable. Use "
                             "for stray topics left by a tool rather than the "
                             "simulation (the exclusions are stored in the "
                             "artifacts so a capture stays auditable)")
    parser.add_argument("--tag", default=None,
                        help="free-form label stored in the capture metadata")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("export_baseline_artifacts", anonymous=True)

    import rospkg
    rospack = rospkg.RosPack()
    package_paths = {}
    for name in ("aquaflow_stonefish", "aquaflow_ref", "hofa_mpc_ros", "stonefish_ros"):
        try:
            package_paths[name] = rospack.get_path(name)
        except rospkg.ResourceNotFound:
            rospy.logwarn("package %s not found; omitted from the report", name)

    workspace = os.path.realpath(
        os.path.join(package_paths["aquaflow_stonefish"], os.pardir, os.pardir))
    output_root = args.output_root or os.path.join(workspace, "artifacts")
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    scenario_file = {"empty_pool": "aquaflow_empty_pool.scn",
                     "single_obstacle": "aquaflow_single_obstacle.scn",
                     "random_pillars": "aquaflow_random_pillars.scn"}[args.scene]
    scenario_path = os.path.join(package_paths["aquaflow_stonefish"],
                                 "scenarios", scenario_file)
    scenario_tree = parse_scenario(scenario_path, package_paths, args.vehicle_name)
    declared = declared_topic_map(scenario_tree)

    extra_patterns = [re.compile(pattern) for pattern in args.exclude]
    published = [(name, type_string)
                 for name, type_string in rospy.get_published_topics()
                 if not topic_is_excluded(name, args.include_all_topics,
                                          extra_patterns)]
    if not published:
        rospy.logerr("no topics published; is the simulation running?")
        return 1

    rospy.loginfo("probing %d topics for %.1f s ...", len(published), args.duration)
    probes = [TopicProbe(name, type_string) for name, type_string in sorted(published)]
    start = rospy.get_time()
    rospy.sleep(args.duration)
    window = rospy.get_time() - start
    for probe in probes:
        probe.unsubscribe()

    present = {probe.name for probe in probes}
    # A scenario-declared sensor with no traffic is the single most misleading
    # gap in a partial capture, so name it explicitly in every artifact.
    satisfied, missing = resolve_declared_topics(declared, present)
    unresolved = sorted(probe.name for probe in probes if probe.unresolved)
    quiet = sorted(probe.name for probe in probes
                   if not probe.unresolved and not probe.wall_times)
    expected_silent = [{"topic": name, "reason": expected_silence_reason(name)}
                       for name in quiet if expected_silence_reason(name)]
    silent = [name for name in quiet if not expected_silence_reason(name)]
    capture = {
        "generated_at_utc": utc_now(),
        "scene": args.scene,
        "vehicle_name": args.vehicle_name,
        "window_s": round(window, 3),
        "tag": args.tag,
        "excluded_topic_patterns": list(args.exclude),
        "complete": not missing and not silent and not unresolved,
        "declared_sensor_topics": {
            topic: {"sensor": declared[topic]["sensor"],
                    "sensor_type": declared[topic]["sensor_type"],
                    "enabled_in_scenario": declared[topic]["enabled"],
                    "declared_rate_hz": declared[topic]["declared_rate_hz"],
                    "live_topics": satisfied[topic]}
            for topic in satisfied},
        "missing_expected_topics": [
            {"topic": topic, "enabled_in_scenario": declared[topic]["enabled"],
             "sensor": declared[topic]["sensor"], "sensor_type": declared[topic]["sensor_type"]}
            for topic in missing],
        "advertised_but_silent_topics": silent,
        "expected_silent_topics": expected_silent,
        "unresolved_type_topics": unresolved,
    }
    if missing:
        rospy.logwarn("PARTIAL capture: %d scenario sensor topic(s) never published: %s",
                      len(missing), ", ".join(missing))
    if silent:
        rospy.logwarn("PARTIAL capture: %d topic(s) advertised but silent: %s",
                      len(silent), ", ".join(silent))
    if unresolved:
        rospy.logwarn("PARTIAL capture: %d topic(s) with unresolvable type: %s "
                      "(re-run with --exclude to drop them if they are stray)",
                      len(unresolved), ", ".join(unresolved))
    for item in expected_silent:
        rospy.loginfo("silent by design, not a gap: %s", item["topic"])

    env_report = collect_env_report(args, package_paths, scenario_tree, present)
    env_report["capture"] = capture

    # Attribute every live topic back to the sensor that declared it, so a
    # camera sub-topic carries its sensor name and declared rate too.
    origin = {}
    for topic, live_topics in satisfied.items():
        for name in live_topics:
            origin[name] = dict(declared[topic], declared_topic=topic)

    schema = {"capture": capture, "topics": {}}
    rates = {"capture": capture, "topics": {}}
    for probe in probes:
        declared_rate = (origin.get(probe.name) or {}).get("declared_rate_hz")
        schema_entry = probe.schema_record(origin.get(probe.name))
        rate_entry = probe.rate_record(window, declared_rate)
        reason = expected_silence_reason(probe.name) if not probe.wall_times else None
        if reason:
            schema_entry["expected_silent"] = reason
            rate_entry["expected_silent"] = reason
        schema["topics"][probe.name] = schema_entry
        rates["topics"][probe.name] = rate_entry

    for filename, payload in (("stonefish_env_report.json", env_report),
                              ("ros_topic_schema.json", schema),
                              ("ros_topic_rates.json", rates)):
        path = os.path.join(output_root, filename)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        rospy.loginfo("wrote %s", path)

    rospy.loginfo("capture %s (%d topics, %.1f s)",
                  "COMPLETE" if capture["complete"] else "PARTIAL",
                  len(probes), window)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass
