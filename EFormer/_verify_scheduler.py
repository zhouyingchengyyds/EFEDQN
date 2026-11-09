"""Behavioural checks for DynamicParamScheduler (pure Python, no torch).

Simulates three performance regimes and verifies that the constraints always
hold.
"""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) if False else None
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DynamicParamScheduler import DynamicParamScheduler


def check(name, sched, sim, batches):
    print(f"--- {name} ---")
    for i in range(batches):
        avg, to = sim(i, sched)
        bs, ms = sched.update(avg, to)
        # Constraint checks
        assert 100 <= ms <= 1920, f"ms out of range: {ms}"
        assert 1 <= bs <= 64, f"bs out of range: {bs}"
        assert bs * ms <= 9600, f"product violated: {bs}*{ms}={bs*ms}"
        if i % 5 == 0 or i == batches - 1:
            print(f"batch {i:3d}: avg_steps={avg:7.1f} timeout={to:.0%} "
                  f"-> ms={ms:4d} bs={bs:2d} corr="
                  f"{sched.correlation() if sched.correlation() is not None else float('nan'):.3f}")
    c = sched.correlation()
    c_str = "None(const ms)" if c is None else f"{c:.3f}"
    print(f"final corr={c_str}\n")
    return c


# 1) Poor policy: every episode hits the cap -> ms should expand to 1920
s1 = DynamicParamScheduler()
check("poor policy (all timeout)", s1,
      lambda i, s: (s.max_steps * 0.98, 1.0), 40)
assert s1.max_steps == 1920 and s1.batch_size == 5, (s1.max_steps, s1.batch_size)

# 2) Good policy: short episodes -> ms should fall back to 100 and bs to 64
s2 = DynamicParamScheduler(init_max_steps=1920, init_batch_size=5)
check("good policy (short episodes)", s2,
      lambda i, s: (40.0, 0.0), 20)
assert s2.max_steps == 100 and s2.batch_size == 64, (s2.max_steps, s2.batch_size)

# 3) Medium with fluctuations: avg_steps improves 300 -> 60 over time
s3 = DynamicParamScheduler(init_max_steps=100, init_batch_size=64)


def sim3(i, s):
    avg = max(60.0, 300.0 - 6.0 * i) + (10.0 if i % 2 else -10.0)
    to = 0.3 if avg > 0.8 * s.max_steps else 0.0
    return avg, to


c3 = check("improving policy (300->60 steps)", s3, sim3, 60)
assert c3 > 0.70, f"corr={c3} below threshold"

print("ALL CHECKS PASSED")
