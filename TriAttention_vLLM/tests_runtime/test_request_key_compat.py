from types import SimpleNamespace

from triattention_runtime.request_key_compat import (
    get_scheduled_token_items,
    req_id_from_scheduled_key,
)


def test_req_id_from_scheduled_key_accepts_int():
    assert req_id_from_scheduled_key(7) == 7
    assert req_id_from_scheduled_key(SimpleNamespace(request_id=9)) == 9
    assert req_id_from_scheduled_key(SimpleNamespace(req_id=11)) == 11


def test_get_scheduled_token_items_accepts_int_keys():
    scheduler_output = SimpleNamespace(num_scheduled_tokens={0: 3, "r1": 1})
    items = get_scheduled_token_items(scheduler_output)
    assert items == [(0, 0, 3), ("r1", "r1", 1)]

