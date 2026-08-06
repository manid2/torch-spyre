# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Probe the actual DL16_TO_FP32 stagger permutation on real hardware.

Fills a 1×64 fp16 tensor with unique sentinel values (0, 1, …, 63) — exactly
one fp16 stick — then upcasts to fp32 on device.  Copying the result verbatim
to host reveals where each logical element physically landed inside the two
fp32 output sticks.

The printed mapping is what ``unstagger_dl16_to_fp32`` in
``fp32_element_arrangement.py`` must implement.

Run with:
    python docs/source/user_guide/examples/scratchpad/probe_dl16_to_fp32_permutation.py
"""

import torch
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout

DEVICE = "spyre"

# Exactly one fp16 stick (64 elements × 2 B = 128 B).
N = 64
x_cpu = torch.arange(N, dtype=torch.float16).unsqueeze(0)  # shape [1, 64]
x_dev = x_cpu.to(DEVICE)


@torch.compile
def upcast(t):
    return t.to(torch.float32)


out_dev = upcast(x_dev)
layout = get_spyre_tensor_layout(out_dev)
print(f"Output EA : {layout.element_arrangement}")
assert layout.element_arrangement == ElementArrangement.DL16_TO_FP32, (
    f"Expected DL16_TO_FP32, got {layout.element_arrangement}"
)

# Copy verbatim — values are numerically correct, positions may be staggered.
out_host = out_dev.cpu().squeeze(0)  # shape [64], fp32
slots = out_host.to(torch.int32).tolist()

print()
print("Physical slot → logical value (what the hardware placed at each position):")
for i, v in enumerate(slots):
    stick = "A" if i < 32 else "B"
    print(f"  slot[{i:2d}] (stick {stick}, pos {i % 32:2d}) = logical e{v}")

print()
print("Compact view:")
print("  stick A (slots  0-31):", slots[:32])
print("  stick B (slots 32-63):", slots[32:])

print()
print("Inverse permutation (logical index → physical slot):")
inv = [0] * N
for phys, log in enumerate(slots):
    inv[log] = phys
print(" ", inv)
