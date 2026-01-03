import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(script_dir, "../src")
sys.path.append(package_dir)

import time

import casadi as ca
import numpy as np
import warp as wp
from casadi_to_warp import CasadiToWarp

# --- CONFIGURATION ---
BATCH_SIZE = 4096
USE_FLOAT64 = False
FIXED_BASE = False

# Global types for Warp
FloatT = wp.float64 if USE_FLOAT64 else wp.float32
np_dtype = np.float64 if USE_FLOAT64 else np.float32


def main():
    wp.init()
    np.random.seed(42)
    np.set_printoptions(precision=3)

    print(
        f"--- Configuration ---\nBatch Size: {BATCH_SIZE}\nPrecision:  {'Float64' if USE_FLOAT64 else 'Float32'}\nFixed Base: {FIXED_BASE}"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(script_dir, "../src")
    sys.path.append(package_dir)
    print(f"DEBUG: script_dir={script_dir}")
    print(f"DEBUG: package_dir={package_dir}, exists={os.path.exists(package_dir)}")
    print(f"DEBUG: sys.path={sys.path}")

    fk_path = os.path.join(script_dir, "../casadi_functions/staccatoe_fk.casadi")

    try:
        fk_func = ca.Function.load(fk_path)
    except Exception as e:
        print(f"Error loading FK function: {e}")
        return
    print("Loaded FK Function.")

    print(
        f"Generating Symbolic Jacobian in CasADi for {'FIXED' if FIXED_BASE else 'FLOATING'} BASE..."
    )
    joint_limits = np.array(
        [
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [0, 2.66],
            [-1.3, 0.5],
            [-0.3, 0.3],
            [-1.1345, 0.1745],
        ]
    )
    # joint_pos_sym = fk_func.sx_in(0)

    if FIXED_BASE:
        nq = 6
        nv = 6
        nu = 6

        q_sym = ca.SX.sym("q", nq)
        v_sym = ca.SX.sym("v", nv)
        I_rotor_sym = ca.SX.sym("I", nu, 1)

        lbq = joint_limits[:, 0]
        ubq = joint_limits[:, 1]

        # 1. Define the clamped expression for position evaluation
        q_clamped = ca.fmax(lbq, ca.fmin(ubq, q_sym))
        motor_pos_sym = fk_func(q_clamped)

        # 2. Fix: Compute Jacobian w.r.t raw symbols, then substitute clamped values
        jac_raw = ca.jacobian(fk_func(q_sym), q_sym)
        jac_sym = ca.substitute(jac_raw, q_sym, q_clamped)

        jac_flat = ca.densify(jac_sym).T  # Make sure it's dense
        jac_flat = ca.reshape(jac_flat, -1, 1)

        jac_func = ca.Function("jac_func", [q_sym], [jac_flat])
        print("Jacobian Function Created.")

        # Armature calculation (Reflected Inertia)
        armature_sym = jac_sym.T @ ca.diag(I_rotor_sym) @ jac_sym

        # For fixed base, we don't need to pad
        armature_sym_ext = armature_sym

        armature_flat = ca.densify(armature_sym_ext).T
        armature_flat = ca.reshape(armature_flat, -1, 1)

        armature_func = ca.Function(
            "armature_func", [q_sym, I_rotor_sym], [armature_flat]
        ).expand()

        motor_vel_sym = jac_sym @ v_sym

        kernel_name = "state_armature_jac_fixed_kernel"
        state_armature_jac_func = ca.Function(
            kernel_name,
            [q_sym, v_sym, I_rotor_sym],
            [motor_pos_sym, motor_vel_sym, jac_flat, armature_flat],
        ).expand()
    else:

        nq = 13
        nv = 12
        nu = 6

        q_sym = ca.SX.sym("q", nq)
        v_sym = ca.SX.sym("v", nv)
        I_rotor_sym = ca.SX.sym("I", nu, 1)

        joint_pos_sym = q_sym[7:]
        joint_vel_sym = v_sym[6:]

        lbq = joint_limits[:, 0]
        ubq = joint_limits[:, 1]

        # 1. Define the clamped expression for position evaluation
        joint_pos_clamped = ca.fmax(lbq, ca.fmin(ubq, joint_pos_sym))
        motor_pos_sym = fk_func(joint_pos_clamped)

        # 2. Fix: Compute Jacobian w.r.t raw symbols, then substitute clamped values
        jac_raw = ca.jacobian(fk_func(joint_pos_sym), joint_pos_sym)
        jac_sym = ca.substitute(jac_raw, joint_pos_sym, joint_pos_clamped)

        jac_flat = ca.densify(jac_sym).T  # Make sure it's dense
        jac_flat = ca.reshape(jac_flat, -1, 1)

        jac_func = ca.Function("jac_func", [joint_pos_sym], [jac_flat])
        print("Jacobian Function Created.")

        armature_sym = jac_sym.T @ ca.diag(I_rotor_sym) @ jac_sym

        armature_sym_ext = ca.SX.zeros(nv, nv)
        # [0|0]
        # -----
        # [0|armature]
        armature_sym_ext[6:, 6:] = armature_sym

        armature_flat = ca.densify(armature_sym_ext).T
        armature_flat = ca.reshape(armature_flat, -1, 1)

        armature_func = ca.Function(
            "armature_func", [joint_pos_sym, I_rotor_sym], [armature_flat]
        ).expand()

        motor_vel_sym = jac_sym @ joint_vel_sym

        kernel_name = "state_armature_jac_kernel"
        state_armature_jac_func = ca.Function(
            kernel_name,
            [q_sym, v_sym, I_rotor_sym],
            [motor_pos_sym, motor_vel_sym, jac_flat, armature_flat],
        ).expand()

    # 3. Transpile Both Functions
    print("\n--- Transpiling ---")

    output_dir = os.path.join(script_dir, "../generated_kernels")

    fk_transpiler = CasadiToWarp(
        fk_func, function_name="fk_kernel", output_dir=output_dir
    )
    fk_kernel = fk_transpiler.load_kernel()
    print("FK Kernel Ready.")

    jac_transpiler = CasadiToWarp(
        jac_func, function_name="jac_kernel", output_dir=output_dir
    )
    jac_kernel = jac_transpiler.load_kernel()
    print("Jacobian Kernel Ready.")

    armature_transpiler = CasadiToWarp(
        armature_func, function_name="armature_kernel", output_dir=output_dir
    )
    armature_kernel = armature_transpiler.load_kernel()

    state_armature_jac_transpiler = CasadiToWarp(
        state_armature_jac_func,
        function_name=kernel_name,
        output_dir=output_dir,
        safe_math=False,
    )

    state_armature_jac_kernel = state_armature_jac_transpiler.load_kernel()

    print(f"{kernel_name} Ready.")

    in_dim = fk_func.size1_in(0)
    out_dim_fk = fk_func.size1_out(0)
    out_dim_jac = jac_func.size1_out(0)  # This is OutDim * InDim

    print(f"Input Dim: {in_dim}")
    print(f"FK Output Dim: {out_dim_fk}")
    print(f"Jac Output Dim (flattened): {out_dim_jac}")

    x_np = np.random.uniform(
        joint_limits[:, 0], joint_limits[:, 1], (BATCH_SIZE, in_dim)
    ).astype(np_dtype)
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
    x_casadi = x_np.T.astype(np.float64)  # CasADi native double
    t0 = time.time()
    res_fk_casadi = fk_map(x_casadi)
    t_fk_casadi = time.time() - t0

    print(f"Warp FK:   {t_fk_warp * 1000:.4f} ms")
    print(f"CasADi FK: {t_fk_casadi * 1000:.4f} ms")
    print(f"Speedup:   {t_fk_casadi / t_fk_warp:.2f}x")

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

    print(f"Warp Jac:   {t_jac_warp * 1000:.4f} ms")
    print(f"CasADi Jac: {t_jac_casadi * 1000:.4f} ms")
    print(f"Speedup:    {t_jac_casadi / t_jac_warp:.2f}x")

    # 7. Validation
    print("\n--- Validation ---")

    # Validate FK
    fk_warp_np = out_fk_wp.numpy()
    fk_casadi_np = np.array(res_fk_casadi).T
    if not USE_FLOAT64:
        fk_casadi_np = fk_casadi_np.astype(np.float32)

    diff_fk = np.max(np.abs(fk_warp_np - fk_casadi_np))

    # Validate Jacobian
    jac_warp_np = out_jac_wp.numpy()
    jac_casadi_np = np.array(res_jac_casadi).T
    if not USE_FLOAT64:
        jac_casadi_np = jac_casadi_np.astype(np.float32)

    diff_array = np.abs(jac_warp_np - jac_casadi_np)
    diff_jac = np.max(diff_array)
    max_diff_idx = np.unravel_index(np.argmax(diff_array), diff_array.shape)
    print(f"Index of Max Diff Jac: {max_diff_idx}")
    print(f"input at max {x_np[max_diff_idx[0]]}")
    print(f"jac at warp max:\n{jac_warp_np[max_diff_idx[0]].reshape(6, 6)}")
    print(f"jac at casadi max:\n{jac_casadi_np[max_diff_idx[0]].reshape(6, 6)}")

    print(f"Max Diff FK:  {diff_fk:.8f}")
    print(f"Max Diff Jac: {diff_jac:.8f}")

    if diff_jac < 1e-3:
        print("SUCCESS: Both kernels match CasADi baseline.")
    else:
        print("MISMATCH DETECTED.")

    print("\n--- Validating State Armature Jacobian Kernel ---")

    # Generate inputs for state_armature_jac_kernel
    # q: (BATCH_SIZE, 13) or (BATCH_SIZE, 6)
    # v: (BATCH_SIZE, 12) or (BATCH_SIZE, 6)
    # I_rotor: (BATCH_SIZE, 6)

    if FIXED_BASE:
        q_np = np.random.uniform(
            joint_limits[:, 0], joint_limits[:, 1], (BATCH_SIZE, 6)
        ).astype(np_dtype)
        v_np = np.random.randn(BATCH_SIZE, 6).astype(np_dtype)
    else:
        q_np = np.zeros((BATCH_SIZE, 13), dtype=np_dtype)
        # Randomize joints (last 6)
        q_np[:, 7:] = np.random.uniform(
            joint_limits[:, 0], joint_limits[:, 1], (BATCH_SIZE, 6)
        ).astype(np_dtype)
        # Randomize base (first 7) - just for completeness, though kernel might not use them all
        q_np[:, :3] = np.random.randn(BATCH_SIZE, 3).astype(np_dtype)  # pos
        q_np[:, 3:7] = np.random.randn(BATCH_SIZE, 4).astype(
            np_dtype
        )  # quat (unnormalized is fine for test if not used for rotation logic that assumes unit)
        # Normalize quat just in case
        q_norm = np.linalg.norm(q_np[:, 3:7], axis=1, keepdims=True)
        q_np[:, 3:7] /= q_norm

        v_np = np.random.randn(BATCH_SIZE, 12).astype(np_dtype)
    I_rotor_np = np.random.uniform(1e-6, 1e-4, (BATCH_SIZE, 6)).astype(np_dtype)

    q_wp = wp.from_numpy(q_np, dtype=FloatT)
    v_wp = wp.from_numpy(v_np, dtype=FloatT)
    I_rotor_wp = wp.from_numpy(I_rotor_np, dtype=FloatT)

    # Outputs
    out_motor_pos_wp = wp.zeros((BATCH_SIZE, nu), dtype=FloatT)
    out_motor_vel_wp = wp.zeros((BATCH_SIZE, nu), dtype=FloatT)
    out_jac_dense_wp = wp.zeros((BATCH_SIZE, nu * nu), dtype=FloatT)
    out_armature_wp = wp.zeros((BATCH_SIZE, nv * nv), dtype=FloatT)  # 12x12 flattened

    # Launch Warp Kernel
    wp.launch(
        kernel=state_armature_jac_kernel,
        dim=BATCH_SIZE,
        inputs=[q_wp, v_wp, I_rotor_wp],
        outputs=[out_motor_pos_wp, out_motor_vel_wp, out_jac_dense_wp, out_armature_wp],
    )
    wp.synchronize()

    # Run CasADi Baseline
    state_armature_jac_map = state_armature_jac_func.map(BATCH_SIZE)
    res_casadi = state_armature_jac_map(q_np.T, v_np.T, I_rotor_np.T)

    motor_pos_casadi = np.array(res_casadi[0]).T
    motor_vel_casadi = np.array(res_casadi[1]).T
    jac_dense_casadi = np.array(res_casadi[2]).T
    armature_casadi = np.array(res_casadi[3]).T

    # Compare
    diff_motor_pos = np.max(np.abs(out_motor_pos_wp.numpy() - motor_pos_casadi))
    diff_motor_vel = np.max(np.abs(out_motor_vel_wp.numpy() - motor_vel_casadi))
    diff_jac_dense = np.max(np.abs(out_jac_dense_wp.numpy() - jac_dense_casadi))
    diff_armature = np.max(np.abs(out_armature_wp.numpy() - armature_casadi))

    print(f"Max Diff Motor Pos: {diff_motor_pos:.8f}")
    print(f"Max Diff Motor Vel: {diff_motor_vel:.8f}")
    print(f"Max Diff Jac Dense: {diff_jac_dense:.8f}")
    print(f"Max Diff Armature:  {diff_armature:.8f}")

    if max(diff_motor_pos, diff_motor_vel, diff_jac_dense, diff_armature) < 1e-3:
        print("SUCCESS: State Armature Jacobian Kernel matches CasADi baseline.")
    else:
        print("MISMATCH DETECTED in State Armature Jacobian Kernel.")


if __name__ == "__main__":
    main()
