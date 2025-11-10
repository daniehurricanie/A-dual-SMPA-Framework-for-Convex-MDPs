
import numpy as np
from cmdp_spma import TabularEstimator

def test_simple_sum_to_one():
    gamma = 0.9
    est = TabularEstimator(2, 2, gamma)
    obs = np.array([0,0,1,1], dtype=np.int64)
    acts = np.array([0,1,0,1], dtype=np.int64)
    dones = np.array([0,0,0,1], dtype=np.float32)
    est.update_from_batch(obs, acts, dones)
    d = est.value()
    s = d.sum()
    assert 0.9 <= s <= 1.1, f"sum d should be ~1, got {s}"

if __name__ == "__main__":
    test_simple_sum_to_one()
    print("ok")
