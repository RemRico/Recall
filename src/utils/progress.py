# -*- coding: utf-8 -*-
import time

class ETAMeter:
    """Simple ETA estimator updated incrementally by processed steps."""
    def __init__(self, total_steps: int):
        self.total = max(1, int(total_steps))
        self.start = time.time()
        self.n = 0

    def step(self, k: int = 1):
        self.n += k

    def eta_seconds(self) -> float:
        elapsed = time.time() - self.start
        rate = self.n / max(1e-9, elapsed)
        remain = max(0, self.total - self.n)
        return remain / max(1e-9, rate)

    def eta_str(self) -> str:
        s = int(self.eta_seconds())
        return f"{s//60:02d}:{s%60:02d}"

    def progress_str(self) -> str:
        pct = 100.0 * self.n / self.total
        return f"{self.n}/{self.total} ({pct:.1f}%) ETA {self.eta_str()}"
