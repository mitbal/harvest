import logging
import threading
import time
import traceback

import pytest

import harvest.memwatch as mw


MB = 1024 * 1024


@pytest.fixture
def watcher_cleanup():
    yield
    mw.stop()


def test_cgroup_v2(tmp_path):
    (tmp_path / "memory.current").write_text(f"{560 * MB}\n")
    (tmp_path / "memory.stat").write_text(
        f"anon {520 * MB}\nfile {30 * MB}\nkernel {5 * MB}\nshmem 0\nslab 123\n"
    )
    assert mw.read_cgroup(tmp_path) == {
        "cgroup_mb": 560.0, "anon_mb": 520.0, "file_mb": 30.0, "kernel_mb": 5.0, "shmem_mb": 0.0,
    }


def test_cgroup_v1_fallback(tmp_path):
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.usage_in_bytes").write_text(f"{100 * MB}\n")
    (v1 / "memory.stat").write_text(f"cache {40 * MB}\nrss {60 * MB}\nshmem {MB}\ntotal_rss 1\n")
    assert mw.read_cgroup(tmp_path) == {"cgroup_mb": 100.0, "anon_mb": 60.0, "file_mb": 40.0, "shmem_mb": 1.0}


def test_cgroup_missing(tmp_path):
    assert mw.read_cgroup(tmp_path / "nope") == {}
    assert mw.read_cgroup(tmp_path) == {}


def test_format_line_order_and_rounding():
    fields = {"uptime_min": 35, "threads": 24, "rss_mb": 541.234, "top_caches": "a:6.1,b:2.0", "cache_data_mb": 12.36}
    assert mw.format_line("MEMORY", fields) == (
        "MEMORY | rss_mb=541.2 | cache_data_mb=12.4 | top_caches=a:6.1,b:2.0 | threads=24 | uptime_min=35"
    )


def test_format_line_jump_head_and_tail():
    line = mw.format_line("MEMORY_JUMP", {"rss_mb": 600.0}, head={"delta_mb": "+62.0", "seconds": 15},
                          tail={"stacks": "t: x.py:1 f"})
    assert line == "MEMORY_JUMP | delta_mb=+62.0 | seconds=15 | rss_mb=600.0 | stacks=t: x.py:1 f"


def test_cache_data_top_caches(monkeypatch):
    from collections import namedtuple
    import sys
    import types

    Stat = namedtuple("Stat", "category_name cache_name byte_length")
    stats = [Stat("st_cache_data", f"mod.f{i}", (i + 1) * MB) for i in range(7)] + [Stat("st_cache_data", "mod.f0", MB)]
    provider = types.SimpleNamespace(get_stats=lambda: {"cache_memory": stats})
    fake = types.ModuleType("cache_data_api")
    fake.get_data_cache_stats_provider = lambda: provider
    monkeypatch.setitem(sys.modules, "streamlit.runtime.caching.cache_data_api", fake)
    fields = mw._cache_data()
    assert fields["cache_data_mb"] == pytest.approx(29.0)
    assert fields["top_caches"] == "f6:7.0,f5:6.0,f4:5.0,f3:4.0,f2:3.0"


def test_jump_decision():
    assert not mw.is_jump(549.9, 500, 50)
    assert mw.is_jump(550, 500, 50)
    assert mw.is_jump(600, 500, 50)
    # After a jump the baseline is the new RSS, so the same level is no longer a jump.
    assert not mw.is_jump(600, 600, 50)
    assert not mw.is_jump(10_000, 0, 0)


def test_read_config_defaults_and_values():
    assert mw.read_config({}) == mw.DEFAULTS
    cfg = mw.read_config({"MEMWATCH_INTERVAL_SECONDS": "60", "MEMWATCH_POLL_SECONDS": "5",
                          "MEMWATCH_JUMP_MB": "20", "MEMWATCH_TRIM": "Yes"})
    assert cfg == mw.Config(interval=60.0, poll=5.0, jump_mb=20.0, trim=True)


def test_read_config_bad_values():
    cfg = mw.read_config({"MEMWATCH_INTERVAL_SECONDS": "abc", "MEMWATCH_POLL_SECONDS": "-1",
                          "MEMWATCH_JUMP_MB": "", "MEMWATCH_TRIM": "maybe"})
    assert cfg == mw.DEFAULTS


def test_start_disabled():
    # Another test may have imported server.py, which starts the real watcher, so only check the return value.
    assert mw.start(logger=logging.getLogger("t"), env={"MEMWATCH_INTERVAL_SECONDS": "0"}) is None


def test_start_idempotent(watcher_cleanup):
    logger = logging.getLogger("memwatch-test")
    env = {"MEMWATCH_INTERVAL_SECONDS": "3600", "MEMWATCH_POLL_SECONDS": "3600"}
    first = mw.start(logger=logger, env=env)
    second = mw.start(logger=logger, env=env)
    assert first is second and first.is_alive() and first.daemon
    assert sum(t.name == mw.THREAD_NAME for t in threading.enumerate()) == 1
    mw.stop()
    assert not first.is_alive()


def test_failing_collector_still_produces_line(caplog):
    def broken():
        raise RuntimeError("boom")

    logger = logging.getLogger("memwatch-collect")
    with caplog.at_level(logging.WARNING, logger="memwatch-collect"):
        fields = mw.collect(0, collectors=(broken, lambda: {"rss_mb": 1.0}), logger=logger)
        mw.collect(0, collectors=(broken,), logger=logger)
    assert fields["rss_mb"] == 1.0 and "uptime_min" in fields
    assert mw.format_line("MEMORY", fields).startswith("MEMORY | rss_mb=1.0 | uptime_min=")
    assert sum("boom" in r.message for r in caplog.records) == 1


def _frames(*specs):
    """Fake stack, outermost first, from ``(path, function)`` pairs; relative paths sit in the repo."""
    root = mw._REPO_ROOT
    return [traceback.FrameSummary(path if path.startswith("/") else root + path, n, name, lookup_line=False)
            for n, (path, name) in enumerate(specs, start=1)]


STDLIB = "/usr/lib/python3.12/"
SITE = "/usr/lib/python3.12/site-packages/"
IDLE = _frames((STDLIB + "threading.py", "_bootstrap"), (STDLIB + "threading.py", "wait"))
POOL_IDLE = _frames((STDLIB + "threading.py", "run"), (STDLIB + "concurrent/futures/thread.py", "_worker"))
LOOP_IDLE = _frames((SITE + "uvicorn/main.py", "run"),
                    (STDLIB + "asyncio/base_events.py", "run_forever"), (STDLIB + "selectors.py", "select"))
WATCHDOG = _frames((STDLIB + "threading.py", "run"), (SITE + "watchdog/observers/fsevents.py", "run"))
BUSY = _frames((STDLIB + "threading.py", "run"), (SITE + "pyarrow/parquet.py", "read"))


def test_app_thread_shows_repo_frames_past_stdlib_tip():
    deep = [(STDLIB + "threading.py", "run"), ("main.py", "render"), ("apps/screener/stock_picker.py", "page"),
            ("harvest/data.py", "get_specific_stock_detail")]
    deep += [(SITE + "requests/api.py", f"call{i}") for i in range(4)]
    deep += [(STDLIB + "ssl.py", f"read{i}") for i in range(4)]
    text = mw.format_stacks({1: _frames(*deep)}, {1: "ScriptRunner.scriptThread"})
    assert text.startswith("ScriptRunner.scriptThread: ssl.py:12 read3 < ")
    assert text.endswith("harvest/data.py:4 get_specific_stock_detail < apps/screener/stock_picker.py:3 page; idle=0")
    assert "main.py" not in text and "..." not in text


def test_app_thread_far_from_tip_keeps_repo_frames_with_gap():
    deep = [("main.py", "render"), ("harvest/data.py", "fetch")] + [(STDLIB + "ssl.py", f"f{i}") for i in range(20)]
    text = mw.format_stacks({1: _frames(*deep)}, {1: "t"})
    assert text.count(" < ") == 10 and "< ... <" in text
    assert text.endswith("harvest/data.py:2 fetch < main.py:1 render; idle=0")


def test_app_thread_with_repo_tip_uses_innermost_six():
    deep = [(STDLIB + "threading.py", "run")] * 3 + [("harvest/x.py", f"f{i}") for i in range(8)]
    text = mw.format_stacks({1: _frames(*deep)}, {1: "t"})
    assert text.count(" < ") == 5 and "threading.py" not in text


def test_idle_threads_collapse_and_app_before_other():
    stacks = {1: BUSY, 2: IDLE, 3: _frames(("harvest/data.py", "load"), (STDLIB + "threading.py", "wait")),
              4: POOL_IDLE, 5: LOOP_IDLE, 6: WATCHDOG}
    text = mw.format_stacks(stacks, {1: "busy", 3: "app"})
    assert text == "app: threading.py:2 wait < harvest/data.py:1 load; busy: pyarrow/parquet.py:2 read < threading.py:1 run; idle=4"


def test_memwatch_file_is_not_app_code():
    own = [traceback.FrameSummary(mw._SELF, 1, "_run", lookup_line=False)] + IDLE
    assert mw.format_stacks({1: own}) == "idle=1"


def test_format_stacks_caps():
    stacks = {i: BUSY for i in range(10)} | {100 + i: IDLE for i in range(3)}
    text = mw.format_stacks(stacks, max_threads=2)
    assert text.count("parquet.py") == 2 and text.endswith("idle=3")
    short = mw.format_stacks(stacks, max_chars=50)
    assert len(short) == 50 and short.endswith("...")


def _blocked_in_app_code(stop):
    stop.wait()


def test_capture_stacks_real_threads():
    stop = threading.Event()
    threads = [threading.Thread(target=stop.wait, name=f"idle-{i}", daemon=True) for i in range(5)]
    threads.append(threading.Thread(target=_blocked_in_app_code, args=(stop,), name="app-worker", daemon=True))
    for t in threads:
        t.start()
    try:
        time.sleep(0.05)
        text = mw.capture_stacks(exclude_ident=threading.get_ident())
        assert text.startswith("app-worker: threading.py:")
        assert "tests/test_memwatch.py:" in text and "_blocked_in_app_code" in text
        assert "idle-0" not in text and "test_capture_stacks_real_threads" not in text
        assert int(text.rsplit("idle=", 1)[1]) >= 5
    finally:
        stop.set()


def test_capture_stacks_shortens_repo_paths():
    text = mw.capture_stacks()
    assert "tests/test_memwatch.py:" in text and mw._REPO_ROOT not in text
