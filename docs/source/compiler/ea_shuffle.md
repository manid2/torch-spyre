# Element-Arrangement Shuffle

The element-arrangement shuffle (`spyre::ea_shuffle`) is a compiler primitive
that restores or applies the within-stick element ordering introduced by
fp16 ↔ fp32 type conversions, without changing a tensor's dtype or logical
shape. It maps to the deeptools `shuffle` opfunc and relies on the
sub-stick reshuffle capability added in deeptools PR #4651.

---

## Background: staggered element arrangement

On-device widening (fp16/bf16 → fp32) cannot keep elements both *in place*
and *in sequential order* within a 128-byte stick. The hardware leaves them
**staggered** — every value is correct but within-stick position is
scrambled. The `ElementArrangement` (EA) enum on `SpyreTensorLayout` tracks
this per tensor:

| Conversion | Input EA | Output EA |
|---|---|---|
| fp16/bf16 → fp32 (upcast) | `STANDARD` | `DL16_TO_FP32` |
| fp32 → fp16/bf16 (downcast) | `STANDARD` | `FP32_TO_DL16` |
| fp32 → fp16/bf16 (restore from upcast) | `DL16_TO_FP32` | `STANDARD` |
| fp16/bf16 → fp32 (restore from downcast) | `FP32_TO_DL16` | `STANDARD` |

An op whose operands carry **different staggered EAs** is illegal (mixed EA).
Previously this raised `Unsupported`. With `ea_shuffle`, arrangement
incompatibilities can be resolved by explicitly converting one operand's EA
before the op.

---

## The four shuffle cases

```
Case 1  fp32(DL16_TO_FP32)  ──ea_shuffle──►  fp32(STANDARD)
        undo stagger after an upcast          (2-byte source elements)

Case 2  fp16(FP32_TO_DL16)  ──ea_shuffle──►  fp16(STANDARD)
        undo stagger after a downcast         (4-byte source elements)

Case 3  fp32(STANDARD)      ──ea_shuffle──►  fp32(FP32_TO_DL16)
        apply downcast stagger                (prepares for a later downcast)

Case 4  fp16(STANDARD)      ──ea_shuffle──►  fp16(DL16_TO_FP32)
        apply upcast stagger                  (prepares for a later upcast)
```

In all four cases the dtype and logical shape are unchanged; only the
within-stick element order (the EA tag) differs.

---

## Implementation

### Custom op (`customops.py`)

```python
@torch.library.custom_op("spyre::ea_shuffle", mutates_args=(), device_types="spyre")
def ea_shuffle(src: torch.Tensor, dst_arrangement: int) -> torch.Tensor: ...
```

`dst_arrangement` is the integer value of the target `ElementArrangement`
enum member. On CPU, `ea_shuffle_cpu` returns a plain clone — arrangement
is a device-internal concept with no host-side meaning.

### Lowering (`lowering.py`)

`lower_spyre_ea_shuffle` materialises an identity `Pointwise` with
`origin_node=V.get_current_node()` so the `spyre.ea_shuffle` FX node is
visible to later passes via `data.origins`. The `dst_arrangement` integer is
stashed directly on the realized `ComputedBuffer`:

```python
pw.realize()
pw.data.data.ea_shuffle_dst_arrangement = dst_arrangement
```

This avoids traversing the FX graph at layout-propagation time.

### Layout propagation (`propagate_layouts.py`)

The `ea_shuffle` case in `_single_arg_op_layout` is reached when
`aten_op == spyreop.ea_shuffle.default`. It:

1. Reads `op.ea_shuffle_dst_arrangement` (set by the lowering).
2. Constructs an output `SpyreTensorLayout` with the same `device_size`,
   `stride_map`, and `device_dtype` as the input, but with the new
   `ElementArrangement`.
3. Assigns `AnyInNode.from_args()` as the restickify cost function — the
   layout geometry is unchanged, so no restickify is needed.

```python
case spyreop.ea_shuffle.default:
    dst_ea = ElementArrangement(op.ea_shuffle_dst_arrangement)
    out_stl = SpyreTensorLayout(
        stl.device_size, stl.stride_map, stl.device_dtype, dst_ea
    )
    op.restick_cost_fn = AnyInNode.from_args()
    return [out_stl]
```

### SDSC codegen (`superdsc.py`, `compute_ops.py`)

**Op mapping.** `parse_op_spec` maps `EA_SHUFFLE_OP` → opfunc `"shuffle"`,
the same opfunc used by LX-relayout identity copies.

**Padding.** `EA_SHUFFLE_OP` uses the same "pad both args" path as
`RESTICKIFY_OP`, ensuring that both the input and output stick boundaries are
correctly aligned in the iteration space.

**Per-tensor arrangement field.** `SDSCArgs` gains an `element_arrangement`
field, populated from `TensorArg.element_arrangement` when building SDSC args.
In `primaryDsInfo_` (inside `generate_sdsc`), a non-`STANDARD` arrangement
emits `"elemArrangement_"` into the per-layout JSON entry:

```python
**(
    {"elemArrangement_": STAGGERED_EA_TO_SDSC_NAME[ea]}
    if ea in STAGGERED_EA_TO_SDSC_NAME
    else {}
),
```

The deeptools backend (PR #4651) reads this field and splits the innermost
loop to perform the within-stick permutation via 16-byte or 2-byte splat
accesses.

### Constants (`constants.py`)

| Name | Value | Purpose |
|---|---|---|
| `EA_SHUFFLE_OP` | `"ea_shuffle"` | Op name string used in op-spec dispatch |
| `STAGGERED_EA_TO_SDSC_NAME` | `{DL16_TO_FP32: "DL16_TO_FP32", FP32_TO_DL16: "FP32_TO_DL16"}` | Maps EA enum values to deeptools JSON field strings |

`EA_SHUFFLE_OP` is also registered in `SPECIAL_OPS` in
`op_spec_validation.py` to prevent spurious "unknown op" warnings.

---

## SDSC JSON example

For case 1 (undo upcast stagger, fp32 tensor, 2×64 shape):

```json
{
  "shuffle": {
    "numCoresUsed_": 1,
    "N_": { "mb_": 2, "out_": 64 },
    "primaryDsInfo_": {
      "INPUT": {
        "layoutDimOrder_": ["mb", "out"],
        "stickDimOrder_": ["out"],
        "stickSize_": [32],
        "elemArrangement_": "DL16_TO_FP32"
      },
      "OUTPUT": {
        "layoutDimOrder_": ["mb", "out"],
        "stickDimOrder_": ["out"],
        "stickSize_": [32]
      }
    }
  }
}
```

`OUTPUT` has no `elemArrangement_` key — its absence signals `STANDARD`
ordering. The deeptools backend splits the loop at the stick boundary and
places the permutation entirely in the two address streams; the SFP copy
instruction is unchanged.

---

## Data flow example

```
x_fp16[2, 64]
    │
    ▼ dl16tofp32
x_fp32[2, 64]  (EA = DL16_TO_FP32, staggered)
    │
    ▼ ea_shuffle(dst_arrangement = STANDARD)
x_fp32[2, 64]  (EA = STANDARD, sequential)
    │
    ▼ add(x_fp32_standard, bias_fp32_standard)   ← no mixed-EA error
result_fp32[2, 64]
```

Without `ea_shuffle`, the `add` above would raise `Unsupported` because one
operand is staggered `DL16_TO_FP32` and the other is `STANDARD`.

---

## Files changed

| File | Change |
|---|---|
| `torch_spyre/_inductor/constants.py` | Add `EA_SHUFFLE_OP`, `STAGGERED_EA_TO_SDSC_NAME` |
| `torch_spyre/_inductor/customops.py` | Define `spyre::ea_shuffle` custom op |
| `torch_spyre/_inductor/lowering.py` | Lower `spyre.ea_shuffle` as tagged Pointwise |
| `torch_spyre/_inductor/propagate_layouts.py` | Propagate EA tag in `_single_arg_op_layout` |
| `torch_spyre/_inductor/codegen/superdsc.py` | Add `element_arrangement` to `SDSCArgs`; map op → `"shuffle"` |
| `torch_spyre/_inductor/codegen/compute_ops.py` | Emit `elemArrangement_` in `primaryDsInfo_` |
| `torch_spyre/_inductor/op_spec_validation.py` | Register `EA_SHUFFLE_OP` in `SPECIAL_OPS` |

---

## Related documentation

- [Layout Optimization](layout_optimization.md) — layout propagation passes
  where EA propagation and `_single_arg_op_layout` live
- [FP32 Element Arrangement (RFC 2971)](../rfcs/index.rst) — the EA enum,
  legality rules, and the ephemeral FP32 bracket design
- [Adding Operations](adding_operations.md) — general guide for new Spyre ops
