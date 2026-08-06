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

"""FP32 Element Arrangement (EA) demo.

Spyre packs tensors into 128-byte **sticks** (64 fp16 elements / 32 fp32
elements per stick).  When a 16-bit tensor is widened to fp32 the hardware
leaves elements in-place rather than reshuffling them across sticks.  This
produces a *staggered* layout: within each pair of consecutive output sticks
the even-indexed logical elements land in stick A and the odd-indexed elements
land in stick B.

  16-bit STANDARD  stick: [ e0  e1  e2  e3  e4  e5  e6  e7 ... e62 e63 ]
                             upcast fp16 → fp32
  fp32 DL16_TO_FP32 sticks:
    stick A: [ e0  e2  e4  e6 ... e62 ]   <- even logical indices
    stick B: [ e1  e3  e5  e7 ... e63 ]   <- odd  logical indices

Every value is present; only within-stick position is scrambled.

This script demonstrates:
  1. That a fp16→fp32 upcast on Spyre produces ``DL16_TO_FP32`` EA.
  2. That copying the staggered fp32 tensor verbatim to host and reversing the
     permutation in Python reproduces the CPU result — the host-side debug
     aid described in the RFC.
  3. That the round-trip fp16→fp32→fp16 restores ``STANDARD`` EA and matches
     the CPU reference numerically.

Run with:
    python docs/source/user_guide/examples/fp32_element_arrangement.py
"""

import torch
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("spyre")
# Stick geometry: a 128-byte stick holds 64 fp16 (2 B) or 32 fp32 (4 B) elems.
ELEMS_PER_FP16_STICK = 64
ELEMS_PER_FP32_STICK = 32

torch.manual_seed(0xAFFE)


# ── Helper: reverse the DL16_TO_FP32 stagger permutation ──────────────────────
def unstagger_dl16_to_fp32(t: torch.Tensor) -> torch.Tensor:
    """Reverse the DL16_TO_FP32 stagger permutation on the host.

    After an fp16→fp32 widening on Spyre the elements within each pair of
    consecutive fp32 sticks are interleaved:

      stick A (fp32, 32 elems): even logical indices → e0 e2 e4 ... e62
      stick B (fp32, 32 elems):  odd logical indices → e1 e3 e5 ... e63

    To recover STANDARD order flatten the last logical dimension into pairs of
    fp32 sticks (i.e. groups of 64 logical elements) and interleave:
      out[2*k]   = stickA[k]   (even position)
      out[2*k+1] = stickB[k]   (odd  position)

    Args:
        t: A CPU fp32 tensor whose last dimension is a multiple of
           ``ELEMS_PER_FP16_STICK`` (64), verbatim-copied from a Spyre tensor
           with ``DL16_TO_FP32`` element arrangement.

    Returns:
        A new tensor with the same shape and dtype as ``t`` in STANDARD
        (sequential logical) order, suitable for numerical comparison with a
        CPU reference.

    Note:
        This function hard-codes the Spyre hardware permutation and is
        **debug-only**.  It must not be used in production code.
    """
    assert t.device.type == "cpu", "Input must already be on CPU"
    assert t.dtype == torch.float32, "Input must be fp32"
    last_dim = t.shape[-1]
    assert last_dim % ELEMS_PER_FP16_STICK == 0, (
        f"Last dim ({last_dim}) must be a multiple of {ELEMS_PER_FP16_STICK}"
    )

    # Work on a flat view of all dimensions except the last.
    batch = t.numel() // last_dim
    flat = t.reshape(batch, last_dim)

    # Each fp16 stick maps to two consecutive fp32 sticks (stick_A, stick_B).
    # Reshape to expose stick pairs: [batch, n_stick_pairs, 2, FP32_STICK_SIZE]
    n_pairs = last_dim // ELEMS_PER_FP16_STICK
    pairs = flat.reshape(batch, n_pairs, 2, ELEMS_PER_FP32_STICK)

    # Interleave: within each pair, elements alternate A[0], B[0], A[1], B[1]…
    # Result shape: [batch, n_pairs, ELEMS_PER_FP16_STICK]
    interleaved = pairs.permute(0, 1, 3, 2).reshape(
        batch, n_pairs, ELEMS_PER_FP16_STICK
    )

    return interleaved.reshape(t.shape)


# ── Step 1: verify EA after fp16 → fp32 upcast ────────────────────────────────
print("=" * 60)
print("Step 1: fp16 → fp32 upcast")
print("=" * 60)

# Use a shape whose last dim is a multiple of 64 (one full fp16 stick).
x_cpu = torch.rand(4, 128, dtype=torch.float16)
x_dev = x_cpu.to(DEVICE)


@torch.compile
def upcast_to_fp32(t):
    return t.to(torch.float32)


staggered_fp32 = upcast_to_fp32(x_dev)
layout = get_spyre_tensor_layout(staggered_fp32)

print(f"Input  EA : {get_spyre_tensor_layout(x_dev).element_arrangement}")
print(f"Output EA : {layout.element_arrangement}")
assert layout.element_arrangement == ElementArrangement.DL16_TO_FP32, (
    f"Expected DL16_TO_FP32, got {layout.element_arrangement}"
)
print("✓ fp16→fp32 upcast produces DL16_TO_FP32 as expected\n")


# ── Step 2: host-side reversal matches CPU ────────────────────────────────────
print("=" * 60)
print("Step 2: reverse permutation on host → compare with CPU")
print("=" * 60)

# Copy the staggered tensor verbatim to host (the stagger is preserved).
staggered_host = staggered_fp32.cpu()

# Apply the host-side un-stagger to restore STANDARD order.
unstaggered_host = unstagger_dl16_to_fp32(staggered_host)

# CPU reference: plain fp16→fp32 cast (always STANDARD).
cpu_fp32 = x_cpu.to(torch.float32)

max_delta = (unstaggered_host - cpu_fp32).abs().max().item()
print(f"Max |unstaggered_host − cpu_fp32| = {max_delta:.6e}")
assert max_delta == 0.0, f"Unstaggered host data does not match CPU: delta={max_delta}"
print("✓ host-side reversal exactly reproduces the CPU fp32 reference\n")


# ── Step 3: round-trip fp16 → fp32 → fp16 restores STANDARD EA ───────────────
print("=" * 60)
print("Step 3: fp16 → fp32 → fp16 round-trip (restores STANDARD EA)")
print("=" * 60)


@torch.compile
def roundtrip(t):
    # fp16(STANDARD) → fp32(DL16_TO_FP32) → fp16(STANDARD)
    return t.to(torch.float32).to(torch.float16)


roundtrip_dev = roundtrip(x_dev)
roundtrip_layout = get_spyre_tensor_layout(roundtrip_dev)

print(f"Round-trip output EA: {roundtrip_layout.element_arrangement}")
assert roundtrip_layout.element_arrangement == ElementArrangement.STANDARD, (
    f"Expected STANDARD after round-trip, got {roundtrip_layout.element_arrangement}"
)

# Numerical comparison with CPU.
cpu_roundtrip = x_cpu.to(torch.float32).to(torch.float16)
max_delta_rt = (roundtrip_dev.cpu() - cpu_roundtrip).abs().max().item()
print(f"Max |device_roundtrip − cpu_roundtrip| = {max_delta_rt:.6e}")
torch.testing.assert_close(roundtrip_dev.cpu(), cpu_roundtrip, rtol=1e-3, atol=1e-3)
print("✓ fp16→fp32→fp16 round-trip restores STANDARD EA and matches CPU\n")

print("All checks passed.")
