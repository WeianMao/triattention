from triattention_v2.effective_len_tracker import EffectiveCacheLenTracker


def test_tracker_without_compression_follows_num_computed():
    t = EffectiveCacheLenTracker()
    assert t.observe_num_computed("r1", 10) == 10
    assert t.observe_num_computed("r1", 15) == 15


def test_tracker_applies_compression_and_accumulates_future_delta():
    t = EffectiveCacheLenTracker()
    assert t.observe_num_computed("r1", 20) == 20

    t.apply_compression(req_id="r1", cache_len_after=8, num_computed_tokens=20)
    assert t.observe_num_computed("r1", 20) == 8
    assert t.observe_num_computed("r1", 23) == 11


def test_tracker_reset_and_remove():
    t = EffectiveCacheLenTracker()
    t.observe_num_computed("r1", 12)
    t.apply_compression(req_id="r1", cache_len_after=6, num_computed_tokens=12)
    assert t.observe_num_computed("r1", 12) == 6

    t.reset_request("r1", num_computed_tokens=0)
    assert t.observe_num_computed("r1", 3) == 3

    t.remove_request("r1")
    assert t.observe_num_computed("r1", 5) == 5

