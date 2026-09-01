"""Check if attn_gym and the fla-npu example can coexist in one process."""
import sys
from pathlib import Path

import torch
import torch_npu  # noqa

# 1. does attn_gym import fla (the wheel)?
import attn_gym
import fla
print("attn_gym", attn_gym.__file__)
print("fla (wheel)", fla.__file__)

# 2. now insert FLA_ROOT and try to import the example module
FLA_ROOT = Path("/data/ljb/projects/create-ascend-titian/flash-linear-attention-npu")
sys.path.insert(0, str(FLA_ROOT))

import importlib
import examples.flash_gated_delta_rule as ex
print("example imported OK")
print("has flash_gated_delta_rule:", hasattr(ex, "flash_gated_delta_rule"))

# 3. can we still reach fla.modules after shadowing?
try:
    import fla.modules.conv.triton.ops
    print("fla.modules still reachable")
except Exception as e:
    print("fla.modules broken after example import:", repr(e)[:200])

# 4. attn_gym chunk_gdn still works?
from attn_gym.linear.gdn import chunk_gdn
print("chunk_gdn importable:", callable(chunk_gdn))
