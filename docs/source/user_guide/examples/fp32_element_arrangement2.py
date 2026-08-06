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
produces a *staggered* layout named ``DL16_TO_FP32``.

The actual hardware permutation (measured by probing with sentinel values) uses
a **group-of-4** granularity.  For one fp16 stick (64 elements → 2 fp32 sticks):

  fp16 STANDARD stick (64 elems):
    [ e0  e1  e2  e3 | e4  e5  e6  e7 | e8 e9 e10 e11 | ... | e60 e61 e62 e63 ]
      ── quad 0 ──     ── quad 1 ──     ──── quad 2 ────       ──── quad 15 ───

  fp32 DL16_TO_FP32 (2 sticks, 32 elems each):
    stick A: [  e0  e1  e2  e3 |  e8  e9 e10 e11 | e16 e17 e18 e19 | ... | e56 e57 e58 e59 ]
               ── quad 0 ──      ──── quad 2 ────   ──── quad 4 ────       ──── quad 14 ───
    stick B: [  e4  e5  e6  e7 | e12 e13 e14 e15 | e20 e21 e22 e23 | ... | e60 e61 e62 e63 ]
               ── quad 1 ──      ──── quad 3 ────   ──── quad 5 ────       ──── quad 15 ───

  Rule: even-numbered quads → stick A; odd-numbered quads → stick B.

Every value is present; only the within-stick quad position is scrambled.

To **reverse** the permutation on host, flatten the last logical dimension into
groups that span two physical fp32 sticks (N//64 groups of 64 logical elements),
split each group at the stick boundary (2 × 32 elems), then interleave the
resulting quads:

  staggered flat: [ A_q0, A_q1, …, A_q7,  B_q0, B_q1, …, B_q7 ]
                    (stick A, 8 quads)        (stick B, 8 quads)

  reshape [2, 8, 4] → permute [1, 0, 2] → reshape [64]
  = [ A_q0, B_q0, A_q1, B_q1, … ]  =  STANDARD order

This script demonstrates:
  1. That a fp16→fp32 upcast on Spyre produces ``DL16_TO_FP32`` EA.
  2. That the measured hardware permutation can be reversed on the host to
     exactly reproduce the CPU reference — the debug aid described in the RFC.
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
ELEMS_PER_FP16_STICK = 64  # one fp16 stick = 64 logical elements
ELEMS_PER_FP32_STICK = 32  # one fp32 stick = 32 logical elements
# Each fp16 stick maps to exactly 2 fp32 sticks (same 128-byte region, wider).
FP32_STICKS_PER_FP16_STICK = 2
# The hardware moves elements in groups of 4 (one 128-bit SIMD lane at fp16).
QUAD = 4

torch.manual_seed(0xAFFE)


# ── Helper: visualise element arrangement ─────────────────────────────────────
def _fmt_quad(values: list, width: int = 4) -> str:
    """Format a short list of values as a fixed-width bracketed quad."""
    inner = " ".join(f"e{v:<3}" for v in values)
    return f"[{inner}]"


def visualise_arrangement(label: str, flat: list, n: int = 64) -> None:
    """Pretty-print the physical element arrangement of one fp16-stick region.

    Args:
        label: Description printed as header.
        flat:  List of *logical* values at each physical slot (length == n).
        n:     Number of elements (must equal ELEMS_PER_FP16_STICK, i.e. 64).
    """
    assert len(flat) == n == ELEMS_PER_FP16_STICK
    print(f"\n  {label}")
    quads_per_stick = ELEMS_PER_FP32_STICK // QUAD  # 8
    # Split into fp32 sticks (each 32 logical positions wide in the flat buffer)
    for s in range(FP32_STICKS_PER_FP16_STICK):
        name = chr(ord("A") + s)
        start = s * ELEMS_PER_FP32_STICK
        stick_elems = flat[start : start + ELEMS_PER_FP32_STICK]
        quads = [stick_elems[q * QUAD : (q + 1) * QUAD] for q in range(quads_per_stick)]
        row = "  ".join(_fmt_quad(q) for q in quads)
        print(f"    stick {name}: {row}")


def print_arrangement_comparison(
    x_fp16_logical: list,
    staggered_physical: list,
    unstaggered_physical: list,
) -> None:
    """Print a three-row comparison: STANDARD fp16, staggered fp32, restored fp32."""
    n = ELEMS_PER_FP16_STICK
    print("\n" + "─" * 72)
    print("  Element arrangement for one fp16 stick (64 elements)")
    print("─" * 72)

    print("\n  [1] fp16 STANDARD input (logical = physical order):")
    quads = [x_fp16_logical[q * QUAD : (q + 1) * QUAD] for q in range(n // QUAD)]
    row_a = "  ".join(_fmt_quad(q) for q in quads[:8])
    print(f"    stick: {row_a}")

    print(
        "\n  [2] fp32 DL16_TO_FP32 output (staggered — even quads → stick A, odd → stick B):"
    )
    visualise_arrangement("physical slots", staggered_physical)

    print("\n  [3] fp32 after host un-stagger (STANDARD order restored):")
    visualise_arrangement("unstaggered", unstaggered_physical)
    print()


# ── Helper: reverse the DL16_TO_FP32 stagger permutation ──────────────────────
def unstagger_dl16_to_fp32(t: torch.Tensor) -> torch.Tensor:
    """Reverse the DL16_TO_FP32 stagger permutation on the host.

    The Spyre hardware moves fp16→fp32 elements in groups of 4 (one 128-bit
    lane). For each fp16 stick (64 logical elements spanning two fp32 sticks of
    32 elements each), even-numbered quads land in stick A and odd-numbered
    quads land in stick B:

      physical flat: [ A_q0 A_q1 … A_q7 | B_q0 B_q1 … B_q7 ]
                       ── stick A (32) ──   ── stick B (32) ──

    Reversal: reshape the last dimension into groups of 64 as
    ``[2 sticks, 8 quads, 4 elems]``, interleave the sticks along the quad
    axis, then flatten back:

      reshape [..., 2, 8, 4]  →  permute [..., 1, 0, 2]  →  reshape [..., 64]

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

    # Flatten all dimensions except the last into a single batch axis.
    batch = t.numel() // last_dim
    n_groups = last_dim // ELEMS_PER_FP16_STICK  # number of fp16-stick groups
    quads_per_stick = ELEMS_PER_FP32_STICK // QUAD  # 8

    # [batch, n_groups, FP32_STICKS_PER_FP16_STICK, quads_per_stick, QUAD]
    # = [batch, n_groups, 2, 8, 4]
    shaped = t.reshape(
        batch, n_groups, FP32_STICKS_PER_FP16_STICK, quads_per_stick, QUAD
    )

    # Interleave the two fp32 sticks along the quad axis:
    #   permute → [batch, n_groups, quads_per_stick, FP32_STICKS_PER_FP16_STICK, QUAD]
    #           = [batch, n_groups, 8, 2, 4]
    interleaved = shaped.permute(0, 1, 3, 2, 4).contiguous()

    # Collapse back to [batch, n_groups * ELEMS_PER_FP16_STICK] then to original shape.
    return interleaved.reshape(t.shape)


# ── Step 1: verify EA after fp16 → fp32 upcast ────────────────────────────────
print("=" * 72)
print("Step 1: fp16 → fp32 upcast — verify DL16_TO_FP32 EA")
print("=" * 72)

# Shape whose last dim is a multiple of 64 (one full fp16 stick per row).
x_cpu = torch.rand(4, 128, dtype=torch.float16)
x_dev = x_cpu.to(DEVICE)


@torch.compile
def upcast_to_fp32(t):
    return t.to(torch.float32)


staggered_fp32 = upcast_to_fp32(x_dev)
layout = get_spyre_tensor_layout(staggered_fp32)

print(f"  Input  EA : {get_spyre_tensor_layout(x_dev).element_arrangement}")
print(f"  Output EA : {layout.element_arrangement}")
assert layout.element_arrangement == ElementArrangement.DL16_TO_FP32, (
    f"Expected DL16_TO_FP32, got {layout.element_arrangement}"
)
print("  ✓ fp16→fp32 upcast produces DL16_TO_FP32 as expected")


# ── Step 2: visualise the permutation on one sentinel stick ───────────────────
print()
print("=" * 72)
print("Step 2: visualise element arrangement (sentinel 1×64 tensor)")
print("=" * 72)

sentinel_cpu = torch.arange(ELEMS_PER_FP16_STICK, dtype=torch.float16).unsqueeze(0)
sentinel_dev = sentinel_cpu.to(DEVICE)
sentinel_fp32_dev = upcast_to_fp32(sentinel_dev)
sentinel_fp32_host = sentinel_fp32_dev.cpu().squeeze(0)  # [64] fp32, physical order

# STANDARD fp16 order is simply 0..63.
fp16_logical = list(range(ELEMS_PER_FP16_STICK))
staggered_physical = sentinel_fp32_host.to(torch.int32).tolist()
unstaggered_physical = (
    unstagger_dl16_to_fp32(sentinel_fp32_host.unsqueeze(0))
    .squeeze(0)
    .to(torch.int32)
    .tolist()
)

print_arrangement_comparison(fp16_logical, staggered_physical, unstaggered_physical)


# ── Step 3: host-side reversal matches CPU ────────────────────────────────────
print("=" * 72)
print("Step 3: reverse permutation on host → compare with CPU reference")
print("=" * 72)

# Copy the staggered tensor verbatim to host (stagger preserved).
staggered_host = staggered_fp32.cpu()

# Apply the host-side un-stagger to restore STANDARD order.
unstaggered_host = unstagger_dl16_to_fp32(staggered_host)

# CPU reference: plain fp16→fp32 cast (always STANDARD).
cpu_fp32 = x_cpu.to(torch.float32)

max_delta = (unstaggered_host - cpu_fp32).abs().max().item()
print(f"  Max |unstaggered_host − cpu_fp32| = {max_delta:.6e}")
assert max_delta == 0.0, f"Unstaggered host data does not match CPU: delta={max_delta}"
print("  ✓ host-side reversal exactly reproduces the CPU fp32 reference")


# ── Step 4: round-trip fp16 → fp32 → fp16 restores STANDARD EA ───────────────
print()
print("=" * 72)
print("Step 4: fp16 → fp32 → fp16 round-trip (restores STANDARD EA)")
print("=" * 72)


@torch.compile
def roundtrip(t):
    # fp16(STANDARD) → fp32(DL16_TO_FP32) → fp16(STANDARD)
    return t.to(torch.float32).to(torch.float16)


roundtrip_dev = roundtrip(x_dev)
roundtrip_layout = get_spyre_tensor_layout(roundtrip_dev)

print(f"  Round-trip output EA : {roundtrip_layout.element_arrangement}")
assert roundtrip_layout.element_arrangement == ElementArrangement.STANDARD, (
    f"Expected STANDARD after round-trip, got {roundtrip_layout.element_arrangement}"
)

cpu_roundtrip = x_cpu.to(torch.float32).to(torch.float16)
max_delta_rt = (roundtrip_dev.cpu() - cpu_roundtrip).abs().max().item()
print(f"  Max |device_roundtrip − cpu_roundtrip| = {max_delta_rt:.6e}")
torch.testing.assert_close(roundtrip_dev.cpu(), cpu_roundtrip, rtol=1e-3, atol=1e-3)
print("  ✓ fp16→fp32→fp16 round-trip restores STANDARD EA and matches CPU")

print()
print("All checks passed.")
