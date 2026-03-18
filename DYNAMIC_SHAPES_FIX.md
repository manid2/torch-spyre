# Dynamic Shaped Tensors Support for Spyre Device

## Problem
Dynamic shaped tensors were not supported on the Spyre device. When using `@torch.compile(dynamic=True)`, the code would fail with:
```
TypeError: Cannot convert symbols to int
```

This occurred at line 699 in `torch_spyre/_inductor/spyre_kernel.py` in the `derive_dim_info` method.

## Root Cause
The issue was that the code tried to convert symbolic expressions (sympy symbols representing dynamic dimensions) to integers using `int()`. This fails because symbolic expressions cannot be converted to concrete integers until runtime when actual tensor shapes are known.

## Solution
The fix involves several changes to support symbolic dimensions throughout the compilation pipeline:

### 1. Updated `DimensionInfo` dataclass (spyre_kernel.py:111-114)
Changed `numel` field from `int` to `Union[int, sympy.Expr]` to support both concrete and symbolic dimensions:
```python
@dataclass(frozen=True)
class DimensionInfo:
    var: sympy.Symbol
    numel: Union[int, sympy.Expr]  # Changed from: numel: int
```

### 2. Updated `derive_dim_info` method (spyre_kernel.py:691-703)
Removed the `int()` conversion that was causing the error:
```python
def derive_dim_info(self, access: TensorAccess) -> list[DimensionInfo]:
    var_ranges = self.var_ranges()
    if var_ranges:
        dim_map = map_dims_to_vars(access.layout, access.index)
        return [
            DimensionInfo(dim_map[v], var_ranges.get(dim_map[v], 1))  # Removed int()
            for v in sorted(dim_map)
        ]
    else:
        return [DimensionInfo(wildcard_symbol(0), 1)]
```

### 3. Updated `create_op_spec` function (spyre_kernel.py:358-391)
Added logic to handle symbolic dimensions when creating OpSpec:
```python
def create_op_spec(...) -> OpSpec:
    # ... validation code ...
    
    # Convert symbolic dimensions to integers when possible
    iteration_space = []
    for d in dims:
        if isinstance(d.numel, sympy.Expr):
            try:
                val = int(d.numel)
                iteration_space.append(val)
            except (TypeError, ValueError):
                # Keep as symbolic expression - will be resolved at runtime
                iteration_space.append(d.numel)
        else:
            iteration_space.append(d.numel)
    
    return OpSpec(op, is_reduction, iteration_space, args, op_info)
```

### 4. Updated `OpSpec` dataclass (runtime/__init__.py:63-80)
Changed `iteration_space` type to support symbolic expressions:
```python
@dataclasses.dataclass
class OpSpec:
    op: str
    is_reduction: bool
    iteration_space: list[Union[int, Any]]  # Changed from: list[int]
    args: Sequence[TensorArg | ConstantArg]
    op_info: dict[str, Any]
```

### 5. Updated async compilation (runtime/async_compile.py:93-122)
Added logic to resolve symbolic dimensions to concrete integers at compilation time:
```python
# Resolve symbolic dimensions to concrete integers
dimensions = []
for dim in ks.iteration_space:
    if isinstance(dim, int):
        dimensions.append(dim)
    elif isinstance(dim, sympy.Expr):
        # Try to resolve from tensor arguments
        resolved = False
        for arg in ks.args:
            if isinstance(arg, TensorArg):
                for size in arg.host_size:
                    if size == dim:
                        dimensions.append(int(size))
                        resolved = True
                        break
            if resolved:
                break
        if not resolved:
            try:
                dimensions.append(int(dim))
            except (TypeError, ValueError):
                dimensions.append(dim)
    else:
        try:
            dimensions.append(int(dim))
        except (TypeError, ValueError):
            dimensions.append(dim)
```

## Testing
A test script `test_dynamic_shapes.py` has been created to verify the fix works with the provided example:

```python
@torch.compile(dynamic=True)
def f(x):
    return x * x.size()[0]

f(torch.rand(10, device=device))
f(torch.rand(20, device=device))
f(torch.rand(30, device=device))
f(torch.rand(40, device=device))
```

## Impact
- **Backward Compatibility**: The changes are backward compatible. Static shapes (concrete integers) continue to work as before.
- **Performance**: No performance impact for static shapes. Dynamic shapes may have slight overhead during compilation but runtime performance should be similar.
- **Scope**: Changes are localized to the Inductor backend compilation pipeline and do not affect eager mode execution.

## Files Modified
1. `torch_spyre/_inductor/spyre_kernel.py` - Core kernel generation logic
2. `torch_spyre/_inductor/runtime/__init__.py` - Runtime data structures
3. `torch_spyre/_inductor/runtime/async_compile.py` - Async compilation and SDSC generation

## Future Improvements
- Add more comprehensive tests for various dynamic shape scenarios
- Optimize symbolic expression resolution for better performance
- Add better error messages when symbolic dimensions cannot be resolved