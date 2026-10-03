import json

from hofa_policy.trajectory_suite import (FAMILIES, SUITE_SCHEMA,
                                          generate_suite, validate_track)


def test_generated_suite_has_disjoint_splits_and_valid_tracks(tmp_path):
    suite = generate_suite(tmp_path / "suite", seed=9, train=12, val=3, test=3)
    assert suite["schema"] == SUITE_SCHEMA
    assert len(suite["tracks"]) == 18
    assert len({entry["id"] for entry in suite["tracks"]}) == 18
    assert {entry["family"] for entry in suite["tracks"]} == set(FAMILIES)
    for entry in suite["tracks"]:
        path = tmp_path / "suite" / entry["path"]
        track = validate_track(json.loads(path.read_text()))
        assert track["split"] == entry["split"]
        assert track["length_m"] > 8.0
