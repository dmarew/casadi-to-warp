# CasADi to Warp

Transpiles CasADi SX functions to NVIDIA Warp kernels for high-performance GPU execution.

## Installation

This project is managed with `uv`.

```bash
uv sync
```

## Usage

### Library

You can use the transpiler in your own scripts:

```python
import casadi as ca
from casadi_to_warp import CasadiToWarp

# Define your CasADi function
x = ca.SX.sym('x', 2)
y = ca.norm_2(x)
func = ca.Function('my_func', [x], [y])

# Transpile to Warp
transpiler = CasadiToWarp(func, function_name="my_kernel")
kernel = transpiler.load_kernel()

# Use with Warp
import warp as wp
# ... setup warp arrays ...
wp.launch(kernel=kernel, dim=1024, inputs=[inputs, outputs])
```

### Examples

Run the example script to see a benchmark of Forward Kinematics and Jacobian calculation:

```bash
uv run examples/staccatoe_motor_map.py
```
