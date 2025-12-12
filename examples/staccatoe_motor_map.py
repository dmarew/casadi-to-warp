import casadi as ca
import warp as wp
import numpy as np
import os
import time
import sys

from casadi_to_warp import CasadiToWarp

# --- CONFIGURATION ---
BATCH_SIZE = 4096 * 4
USE_FLOAT64 = True

# Global types for Warp
FloatT = wp.float64 if USE_FLOAT64 else wp.float32
np_dtype = np.float64 if USE_FLOAT64 else np.float32

def main():
    wp.init()
    np.random.seed(42)
    
    print(f"--- Configuration ---\nBatch Size: {BATCH_SIZE}\nPrecision:  {'Float64' if USE_FLOAT64 else 'Float32'}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fk_path = os.path.join(script_dir, "../casadi_functions/staccatoe_fk_func.casadi")
    
    try:        
        fk_func = ca.Function.load(fk_path)
    except Exception as e:
        print(f"Error loading FK function: {e}")
        return
    print("Loaded FK Function.")

    print("Generating Symbolic Jacobian in CasADi...")
    
    x_sym = fk_func.sx_in(0)
    out_sym = fk_func(x_sym)
    jac_sym = ca.jacobian(out_sym, x_sym)
    
    jac_flat = ca.densify(jac_sym) # Make sure it's dense
    jac_flat = ca.reshape(jac_flat, -1, 1) 
    
    jac_func = ca.Function('jac_func', [x_sym], [jac_flat])
    print("Jacobian Function Created.")

    # 3. Transpile Both Functions
    print("\n--- Transpiling ---")
    
    output_dir = "generated_kernels"
    
    fk_transpiler = CasadiToWarp(fk_func, function_name="fk_kernel", use_float64=USE_FLOAT64, output_dir=output_dir)
    fk_kernel = fk_transpiler.load_kernel()
    print("FK Kernel Ready.")
    
    jac_transpiler = CasadiToWarp(jac_func, function_name="jac_kernel", use_float64=USE_FLOAT64, output_dir=output_dir)
    jac_kernel = jac_transpiler.load_kernel()
    print("Jacobian Kernel Ready.")

    in_dim = fk_func.size1_in(0)
    out_dim_fk = fk_func.size1_out(0)
    out_dim_jac = jac_func.size1_out(0) # This is OutDim * InDim
    
    print(f"Input Dim: {in_dim}")
    print(f"FK Output Dim: {out_dim_fk}")
    print(f"Jac Output Dim (flattened): {out_dim_jac}")

    joint_limits = np.array(
        [
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [0, 2.66],
            [-1.3, 0.3],
            [-0.3, 0.3],
            [-0.9, 0.3],
        ]
    )
    
    x_np = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1], (BATCH_SIZE, in_dim)).astype(np_dtype)
    x_wp = wp.from_numpy(x_np, dtype=FloatT)
    
    # Create Outputs
    out_fk_wp = wp.zeros((BATCH_SIZE, out_dim_fk), dtype=FloatT)
    out_jac_wp = wp.zeros((BATCH_SIZE, out_dim_jac), dtype=FloatT)

    args_fk = [x_wp, out_fk_wp]
    args_jac = [x_wp, out_jac_wp]

    print("\n--- Benchmarking Forward Kinematics ---")
    
    # Warmup
    wp.launch(kernel=fk_kernel, dim=BATCH_SIZE, inputs=args_fk)
    wp.synchronize()
    
    t0 = time.time()
    wp.launch(kernel=fk_kernel, dim=BATCH_SIZE, inputs=args_fk)
    wp.synchronize()
    t_fk_warp = time.time() - t0
    
    fk_map = fk_func.map(BATCH_SIZE)
    x_casadi = x_np.T.astype(np.float64) # CasADi native double
    t0 = time.time()
    res_fk_casadi = fk_map(x_casadi)
    t_fk_casadi = time.time() - t0
    
    print(f"Warp FK:   {t_fk_warp*1000:.4f} ms")
    print(f"CasADi FK: {t_fk_casadi*1000:.4f} ms")
    print(f"Speedup:   {t_fk_casadi/t_fk_warp:.2f}x")

    print("\n--- Benchmarking Jacobian ---")
    
    # Warmup
    wp.launch(kernel=jac_kernel, dim=BATCH_SIZE, inputs=args_jac)
    wp.synchronize()
    
    t0 = time.time()
    wp.launch(kernel=jac_kernel, dim=BATCH_SIZE, inputs=args_jac)
    wp.synchronize()
    t_jac_warp = time.time() - t0
    
    # CasADi CPU
    jac_map = jac_func.map(BATCH_SIZE)
    t0 = time.time()
    res_jac_casadi = jac_map(x_casadi)
    t_jac_casadi = time.time() - t0
    
    print(f"Warp Jac:   {t_jac_warp*1000:.4f} ms")
    print(f"CasADi Jac: {t_jac_casadi*1000:.4f} ms")
    print(f"Speedup:    {t_jac_casadi/t_jac_warp:.2f}x")

    # 7. Validation
    print("\n--- Validation ---")
    
    # Validate FK
    fk_warp_np = out_fk_wp.numpy()
    fk_casadi_np = np.array(res_fk_casadi).T
    if not USE_FLOAT64: fk_casadi_np = fk_casadi_np.astype(np.float32)
    
    diff_fk = np.max(np.abs(fk_warp_np - fk_casadi_np))
    
    # Validate Jacobian
    jac_warp_np = out_jac_wp.numpy()
    jac_casadi_np = np.array(res_jac_casadi).T
    if not USE_FLOAT64: jac_casadi_np = jac_casadi_np.astype(np.float32)
    
    diff_jac = np.max(np.abs(jac_warp_np - jac_casadi_np))
    
    print(f"Max Diff FK:  {diff_fk:.8f}")
    print(f"Max Diff Jac: {diff_jac:.8f}")
    
    if diff_jac < 1e-3:
        print("SUCCESS: Both kernels match CasADi baseline.")
    else:
        print("MISMATCH DETECTED.")

if __name__ == "__main__":
    main()