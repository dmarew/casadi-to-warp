import casadi as ca
import re
import os
import sys
import importlib.util

class CasadiToWarp:
    """
    Transpiles CasADi SX functions to NVIDIA Warp kernels.
    """
    def __init__(self, casadi_func: ca.Function, function_name="casadi_kernel_gen", use_float64=False, output_dir="."):
        self.func = casadi_func
        self.name = function_name
        self.dtype_str = "wp.float64" if use_float64 else "wp.float32"
        self.source_code = ""
        self.output_dir = output_dir
        
        # Regex Math Map
        self.math_map = {
            r'\bcasadi_sq\(': 'sq(', r'\bsq\(': 'sq(',
            r'\bcasadi_sqrt\(': 'wp.sqrt(', r'\bsqrt\(': 'wp.sqrt(',
            r'\bcasadi_pow\(': 'wp.pow(', r'\bpow\(': 'wp.pow(',
            r'\bexp\(': 'wp.exp(', r'\blog\(': 'wp.log(', r'\blog10\(': 'wp.log10(',
            r'\bsin\(': 'wp.sin(', r'\bcos\(': 'wp.cos(', r'\btan\(': 'wp.tan(',
            r'\basin\(': 'wp.asin(', r'\bacos\(': 'wp.acos(', r'\batan\(': 'wp.atan(',
            r'\batan2\(': 'wp.atan2(',
            r'\bsinh\(': 'wp.sinh(', r'\bcosh\(': 'wp.cosh(', r'\btanh\(': 'wp.tanh(',
            r'\basinh\(': 'wp.asinh(', r'\bacosh\(': 'wp.acosh(', r'\batanh\(': 'wp.atanh(',
            r'\bcasadi_fabs\(': 'wp.abs(', r'\bfabs\(': 'wp.abs(', r'\babs\(': 'wp.abs(',
            r'\bfloor\(': 'wp.floor(', r'\bceil\(': 'wp.ceil(',
            r'\bcasadi_sign\(': 'wp.sign(', r'\bsign\(': 'wp.sign(',
            # Custom Helpers
            r'\bcopysign\(':     'c_copysign(', 
            r'\bcasadi_fmod\(':  'c_fmod(', 
            r'\bfmod\(':         'c_fmod(',
            r'\bcasadi_fmin\(':  'wp.min(', r'\bfmin\(': 'wp.min(',
            r'\bcasadi_fmax\(':  'wp.max(', r'\bfmax\(': 'wp.max(',
            r'\bcasadi_erf\(':   'wp.erf(', r'\berf\(': 'wp.erf(',
        }
        self.logic_map = { ' && ': ' and ', ' || ': ' or ', ' !': ' not ' }

    def _generate_c_source(self):
        filename = f"{self.name}_tmp.c"
        opts = {'main': False, 'mex': False, 'with_header': False}
        self.func.generate(filename, opts)
        with open(filename, 'r') as f: c_code = f.read()
        if os.path.exists(filename): os.remove(filename)
        return c_code

    def _parse_body(self, c_code):
        # Extract function body
        func_pattern = re.compile(r"(?:static\s+)?(?:int|void)\s+\w+\s*\(\s*const\s+casadi_real\*\*?\s*arg,\s*casadi_real\*\*?\s*res.*?\)\s*\{(.*?)\}", re.DOTALL)
        match = func_pattern.search(c_code)
        if not match:
            func_pattern_simple = re.compile(r"void \w+\(const casadi_real\*\* arg, casadi_real\*\* res\)\s*\{(.*?)\}", re.DOTALL)
            match = func_pattern_simple.search(c_code)
            if not match: raise ValueError("Could not parse CasADi C output.")
        
        body = match.group(1)
        lines = body.split('\n')
        parsed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith(("//", "#", "return", "if (sz_", "if (arg")): continue
            if line.startswith("casadi_real ") and "=" not in line: continue 
            
            # Fix Ternary Null Checks
            line = re.sub(r'arg\[(\d+)\]\s*\?\s*arg\[\1\]\[(\d+)\]\s*:\s*0', r'arg[\1][\2]', line)

            # Logic & Generic Ternary
            for c_op, py_op in self.logic_map.items():
                line = line.replace(c_op, py_op)
            if '?' in line and ':' in line:
                line = re.sub(r'=\s*(.*?)\s*\?\s*(.*?)\s*:\s*(.*?);', r'= \2 if (\1) else \3;', line)

            # Output Checks
            if "if (res" in line:
                assign_match = re.search(r'res\[\d+\]\[\d+\]\s*=[^;]+;', line)
                if assign_match: line = assign_match.group(0)

            # Variable Names
            line = re.sub(r'arg\[(\d+)\]', r'inputs_\1', line)
            line = re.sub(r'res\[(\d+)\]', r'outputs_\1', line)
            
            # Apply Math Map
            for pattern, replacement in self.math_map.items():
                line = re.sub(pattern, replacement, line)
            
            # Fix Array Indexing (inputs_0[0] -> inputs_0[tid, 0])
            # Use a safe placeholder so casting doesn't destroy the index
            def index_replacer(m):
                return f"{m.group(1)}{m.group(2)}[tid, __IDX_{m.group(3)}__]"
            line = re.sub(r'(inputs_|outputs_)(\d+)\[(\d+)\]', index_replacer, line)
            
            # Cast Numeric Literals
            numeric_pattern = r'(?<![\w])(\d+(\.\d*)?([eE][+-]?\d+)?)'
            line = re.sub(numeric_pattern, r'FloatT(\1)', line)
            
            # Restore Indices
            line = re.sub(r'__IDX_(\d+)__', r'\1', line)

            if line.endswith(";"):
                line = line[:-1]
            parsed_lines.append(line)
        return parsed_lines

    def transpile(self):
        c_code = self._generate_c_source()
        body_lines = self._parse_body(c_code)
        
        indent = "    "
        python_src = []
        python_src.append("import warp as wp")
        python_src.append("")
        python_src.append(f"FloatT = {self.dtype_str}")
        python_src.append("")

        # Helpers
        python_src.append("@wp.func")
        python_src.append("def sq(x: FloatT):")
        python_src.append(f"{indent}return x * x")
        python_src.append("")
        python_src.append("@wp.func")
        python_src.append("def c_copysign(x: FloatT, y: FloatT):")
        python_src.append(f"{indent}return wp.select(y >= 0.0, -wp.abs(x), wp.abs(x))")
        python_src.append("")
        python_src.append("@wp.func")
        python_src.append("def c_fmod(x: FloatT, y: FloatT):")
        python_src.append(f"{indent}return x - wp.trunc(x / y) * y")
        python_src.append("")
        
        n_in = self.func.n_in()
        n_out = self.func.n_out()
        
        args = []
        for i in range(n_in):
            args.append(f"inputs_{i}: wp.array(dtype={self.dtype_str}, ndim=2)")
        for i in range(n_out):
            args.append(f"outputs_{i}: wp.array(dtype={self.dtype_str}, ndim=2)")
            
        arg_str = ", ".join(args)
        python_src.append("@wp.kernel")
        python_src.append(f"def {self.name}({arg_str}):")
        python_src.append(f"{indent}tid = wp.tid()")
        
        for line in body_lines:
            python_src.append(f"{indent}{line}")
            
        self.source_code = "\n".join(python_src)
        return self.source_code

    def load_kernel(self):
        if not self.source_code: self.transpile()
        
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        module_filename = os.path.join(self.output_dir, f"{self.name}.py")
        with open(module_filename, "w") as f: f.write(self.source_code)
        
        spec = importlib.util.spec_from_file_location(self.name, module_filename)
        if spec is None: raise ImportError(f"Could not load generated kernel from {module_filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.name] = module
        spec.loader.exec_module(module)
        return getattr(module, self.name)
