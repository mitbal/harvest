"""Memory heartbeat for the web process, logged on a timer rather than per request.

A daemon thread polls process RSS every few seconds and writes one ``MEMORY``
line every few minutes with numbers that separate the likely causes of slow
growth: Streamlit caches, sessions, allocator retention (RSS up while Python
numbers stay flat), container memory outside the process, and PyArrow's pool.
A sudden rise also writes a ``MEMORY_JUMP`` line with what every thread was
running at that moment. Nothing starts until ``start()`` is called.
"""
import os
import sys
import time
import ctypes
import logging
import platform
import threading
import traceback
from pathlib import Path
from collections import namedtuple


THREAD_NAME = 'memwatch'
MB = 1024 * 1024
FIELD_ORDER = ('rss_mb', 'cgroup_mb', 'anon_mb', 'file_mb', 'shmem_mb', 'kernel_mb', 'cache_data_mb',
               'top_caches', 'sessions', 'active_sessions', 'threads', 'py_blocks', 'arrow_mb', 'arrow_pool',
               'uptime_min')
_SELF = str(Path(__file__).resolve())
_REPO_ROOT = str(Path(_SELF).parent.parent) + os.sep

Config = namedtuple('Config', 'interval poll jump_mb trim')
DEFAULTS = Config(interval=300.0, poll=15.0, jump_mb=50.0, trim=False)

_lock = threading.Lock()
_stop = threading.Event()
_warned = set()


def _number(env, name, default):
    try:
        value = float(env.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def read_config(env=None):
    """Settings from ``MEMWATCH_*`` environment variables; bad values fall back to the defaults."""
    env = os.environ if env is None else env
    return Config(
        interval=_number(env, 'MEMWATCH_INTERVAL_SECONDS', DEFAULTS.interval),
        poll=_number(env, 'MEMWATCH_POLL_SECONDS', DEFAULTS.poll) or DEFAULTS.poll,
        jump_mb=_number(env, 'MEMWATCH_JUMP_MB', DEFAULTS.jump_mb),
        trim=str(env.get('MEMWATCH_TRIM', '')).strip().lower() in ('1', 'true', 'yes', 'on'),
    )


def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / MB


def read_cgroup(root='/sys/fs/cgroup'):
    """Container memory from cgroup v2, else v1; an empty dict when neither is readable."""
    root = Path(root)
    v2 = {'anon': 'anon_mb', 'file': 'file_mb', 'shmem': 'shmem_mb', 'kernel': 'kernel_mb'}
    v1 = {'rss': 'anon_mb', 'cache': 'file_mb', 'shmem': 'shmem_mb'}
    for usage, stat, keys in ((root / 'memory.current', root / 'memory.stat', v2),
                              (root / 'memory' / 'memory.usage_in_bytes', root / 'memory' / 'memory.stat', v1)):
        if not usage.is_file():
            continue
        fields = {}
        try:
            fields['cgroup_mb'] = int(usage.read_text().split()[0]) / MB
        except (OSError, ValueError, IndexError):
            pass
        try:
            for line in stat.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] in keys:
                    fields[keys[parts[0]]] = int(parts[1]) / MB
        except (OSError, ValueError):
            pass
        return fields
    return {}


def _process():
    import psutil
    proc = psutil.Process()
    return {'rss_mb': proc.memory_info().rss / MB, 'threads': proc.num_threads(), 'py_blocks': sys.getallocatedblocks()}


def _cache_data():
    """Sizes of ``st.cache_data`` entries; these are just lengths of the pickled values, so cheap."""
    from streamlit.runtime.caching.cache_data_api import get_data_cache_stats_provider
    per_function = {}
    for stats in get_data_cache_stats_provider().get_stats().values():
        for stat in stats:
            name = stat.cache_name.rsplit('.', 1)[-1]
            per_function[name] = per_function.get(name, 0) + stat.byte_length
    top = sorted(per_function.items(), key=lambda item: item[1], reverse=True)[:5]
    fields = {'cache_data_mb': sum(per_function.values()) / MB}
    if top:
        fields['top_caches'] = ','.join(f'{name}:{size / MB:.1f}' for name, size in top)
    return fields


def _sessions():
    from streamlit.runtime import Runtime
    if not Runtime.exists():
        return {}
    # Runtime registers itself before building its session manager, so the first beat can land in between.
    mgr = getattr(Runtime.instance(), '_session_mgr', None)
    if mgr is None:
        return {}
    return {'sessions': mgr.num_sessions(), 'active_sessions': mgr.num_active_sessions()}


def _arrow():
    pa = sys.modules.get('pyarrow')
    if pa is None:
        return {}
    return {'arrow_mb': pa.total_allocated_bytes() / MB, 'arrow_pool': pa.default_memory_pool().backend_name}


COLLECTORS = (_process, read_cgroup, _cache_data, _sessions, _arrow)


def _warn_once(logger, key, exc):
    if key not in _warned:
        _warned.add(key)
        logger.warning(f'MEMORY_ERROR | where={key[0]} | error={type(exc).__name__}: {exc}')


def collect(started_at, collectors=COLLECTORS, logger=None):
    """Run every collector, each guarded, and merge their fields."""
    fields = {}
    for collector in collectors:
        try:
            fields.update(collector())
        except Exception as exc:
            if logger is not None:
                _warn_once(logger, (collector.__name__, type(exc).__name__), exc)
    fields['uptime_min'] = int((time.monotonic() - started_at) / 60)
    return fields


def format_line(prefix, fields, head=None, tail=None):
    """``PREFIX | key=value | ...`` with known fields in a fixed order and MB values to one decimal."""
    def fmt(key, value):
        return f'{key}={value:.1f}' if isinstance(value, float) else f'{key}={value}'
    ordered = [k for k in FIELD_ORDER if k in fields] + [k for k in fields if k not in FIELD_ORDER]
    parts = [prefix, *(fmt(k, v) for k, v in (head or {}).items()), *(fmt(k, fields[k]) for k in ordered),
             *(fmt(k, v) for k, v in (tail or {}).items())]
    return ' | '.join(parts)


def is_jump(rss, baseline, jump_mb):
    return jump_mb > 0 and rss - baseline >= jump_mb


def _short_path(path):
    if path.startswith(_REPO_ROOT):
        return path[len(_REPO_ROOT):]
    marker = 'site-packages' + os.sep
    return path.split(marker, 1)[1] if marker in path else os.path.basename(path)


_IDLE_FILES = {'threading.py', 'queue.py', 'selectors.py', 'base_events.py', 'runners.py', 'selector_events.py'}


def _is_app_frame(frame):
    return frame.filename.startswith(_REPO_ROOT) and frame.filename != _SELF


def _is_idle(stack):
    """Whether a thread's innermost frame (``stack`` is innermost first) is a plain wait or an idle event loop."""
    tip = stack[0]
    base = os.path.basename(tip.filename)
    return (base in _IDLE_FILES or (base == 'thread.py' and tip.name == '_worker')
            or 'watchdog' in tip.filename or 'polling_path_watcher' in tip.filename)


def _render(stack, max_frames, app_frames):
    """Innermost frames first; for app threads, reach out to the innermost two repo frames."""
    keep = list(range(min(max_frames, len(stack))))
    repo = [i for i, f in enumerate(stack) if _is_app_frame(f)][:2]
    if repo and repo[-1] >= len(keep):
        if repo[-1] < app_frames:
            keep = list(range(repo[-1] + 1))
        else:
            keep = sorted(set(range(app_frames - len(repo))) | set(repo))
    parts, previous = [], -1
    for i in keep:
        if i != previous + 1:
            parts.append('...')
        f = stack[i]
        parts.append(f'{_short_path(f.filename)}:{f.lineno} {f.name}')
        previous = i
    return ' < '.join(parts)


def format_stacks(stacks, names=None, max_threads=40, max_frames=6, app_frames=10, max_chars=4000):
    """``{ident: [FrameSummary, ...] outermost first}`` as text: app threads, then busy ones, then ``idle=N``."""
    names = names or {}
    app, other, idle = [], [], 0
    for ident, stack in stacks.items():
        stack = list(stack)[::-1]
        if not stack:
            continue
        if any(_is_app_frame(f) for f in stack):
            app.append((ident, stack))
        elif _is_idle(stack):
            idle += 1
        else:
            other.append((ident, stack))
    entries = [f'{names.get(ident, ident)}: {_render(stack, max_frames, app_frames)}'
               for ident, stack in (app + other)[:max_threads]]
    text = '; '.join([*entries, f'idle={idle}'])
    return text if len(text) <= max_chars else text[:max_chars - 3] + '...'


def capture_stacks(exclude_ident=None, max_threads=40, max_frames=6, max_chars=4000, frames=None):
    """What every other thread is running, app code first and idle threads only counted."""
    names = {t.ident: t.name for t in threading.enumerate()}
    frames = sys._current_frames() if frames is None else frames
    stacks = {ident: traceback.extract_stack(frame) for ident, frame in list(frames.items()) if ident != exclude_ident}
    return format_stacks(stacks, names, max_threads=max_threads, max_frames=max_frames, max_chars=max_chars)


def _malloc_trim():
    """Ask glibc to return freed memory to the OS; None when not on Linux/glibc."""
    if sys.platform != 'linux' or platform.libc_ver()[0] != 'glibc':
        return None
    return ctypes.CDLL('libc.so.6').malloc_trim(0)


def _run(cfg, logger, stop):
    started_at = last_at = time.monotonic()
    next_beat = started_at
    baseline = None
    while not stop.is_set():
        try:
            now = time.monotonic()
            if now >= next_beat:
                fields = collect(started_at, logger=logger)
                logger.info(format_line('MEMORY', fields))
                baseline, last_at, next_beat = fields.get('rss_mb', rss_mb()), now, now + cfg.interval
                if cfg.trim:
                    before = rss_mb()
                    if _malloc_trim() is not None:
                        after = rss_mb()
                        logger.info(format_line('MEMORY_TRIM', {'rss_mb': after}, head={'trimmed_mb': before - after}))
                        baseline = after
            else:
                rss = rss_mb()
                baseline = rss if baseline is None else baseline
                if is_jump(rss, baseline, cfg.jump_mb):
                    fields = collect(started_at, logger=logger)
                    head = {'delta_mb': f'{rss - baseline:+.1f}', 'seconds': int(now - last_at)}
                    stacks = capture_stacks(exclude_ident=threading.get_ident())
                    logger.info(format_line('MEMORY_JUMP', fields, head=head, tail={'stacks': stacks}))
                    baseline, last_at = fields.get('rss_mb', rss), now
        except Exception as exc:
            _warn_once(logger, ('loop', type(exc).__name__), exc)
        stop.wait(cfg.poll)


def start(logger=None, env=None):
    """Start the watcher thread once per process; returns it, or None when disabled."""
    cfg = read_config(env)
    if cfg.interval == 0:
        return None
    with _lock:
        for thread in threading.enumerate():
            if thread.name == THREAD_NAME and thread.is_alive():
                return thread
        if logger is None:
            from harvest.utils import setup_logging
            logger = setup_logging('memory')
        _stop.clear()
        thread = threading.Thread(target=_run, args=(cfg, logger, _stop), name=THREAD_NAME, daemon=True)
        thread.start()
        return thread


def stop(timeout=5):
    """Stop the watcher thread, for tests and dev reloads."""
    _stop.set()
    for thread in threading.enumerate():
        if thread.name == THREAD_NAME:
            thread.join(timeout)
