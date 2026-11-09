"""GPU telemetry for a run: a background sampler that appends to ``telemetry/gpu.jsonl``.

Derived data only. Nothing here can fail a run: a sample that raises is recorded as an error row
and the run goes on. Each row carries the tag of the trial or task-run in flight, so peaks join
to the work that caused them. Rows: ``{"t": epoch seconds, "tag": str|None, ...fields}``.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time

DEFAULT_INTERVAL = 5.0


class GpuSampler:
    def __init__(self, sample, path, interval=DEFAULT_INTERVAL, clock=time.time):
        self.sample, self.path, self.interval, self.clock = sample, Path(path), interval, clock
        self.tag = None
        self.errors = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def snap(self):
        """One sample now, written with the current tag."""
        row = {"t": self.clock(), "tag": self.tag}
        try:
            values = self.sample()
            if values is None:
                return
            row.update(values)
        except Exception as exc:  # telemetry never fails a run
            self.errors += 1
            row["error"] = str(exc)[:200]
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError:  # a full disk loses samples, never the run
            self.errors += 1

    def start(self):
        if self.interval and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
            self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.snap()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self.snap()


def _samplers(journal):
    return [s for s in (getattr(journal, "sampler", None), *getattr(journal, "extra_samplers", ())) if s is not None]


@contextmanager
def sampling(journal, tag):
    """Tag the samples taken while ``tag`` runs, with one at each end so a short trial has data.
    Also asks the journal's ``server_probe(tag, phase)`` (if any) for a llama-server snapshot at each end."""
    samplers = _samplers(journal)
    probe = getattr(journal, "server_probe", None)
    if not samplers and probe is None:
        yield
        return
    for sampler in samplers:
        sampler.tag = tag
        sampler.snap()
    _probe(probe, tag, "start")
    try:
        yield
    finally:
        _probe(probe, tag, "end")
        for sampler in samplers:
            sampler.snap()
            sampler.tag = None


def _probe(probe, tag, phase):
    if probe is None:
        return
    try:
        probe(tag, phase)
    except Exception:  # telemetry never fails a run
        pass


def parse_nvidia_smi(line):
    """One ``nvidia-smi --query-gpu`` CSV line -> named fields; ``[N/A]`` and blanks are None."""
    names = ("mem_used_mib", "mem_free_mib", "util_gpu_pct", "util_mem_pct", "temp_c", "power_w",
             "clock_sm_mhz", "clock_mem_mhz")
    cells = [c.strip() for c in line.split(",")]
    if len(cells) != len(names):
        raise ValueError(f"unexpected nvidia-smi row: {line!r}")
    def number(cell):
        try:
            return float(cell)
        except ValueError:
            return None
    return {name: number(cell) for name, cell in zip(names, cells)}


QUERY = "memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem"
