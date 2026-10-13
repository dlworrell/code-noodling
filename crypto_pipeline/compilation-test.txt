import subprocess
import os

def test_compilation_pipeline():
    print("=== Launching Automated C-Wrapper Integrity Compilation ===")
    
    # Check for file existence
    if not os.path.exists("crypto_multiplier.c"):
        print("[ERROR] 'crypto_multiplier.c' source file missing.")
        return False
        
    # Standard strict compiler execution sequence mimicking the Makefile flags
    compile_cmd = [
        "gcc", "-O2", "-Wall", "-Wextra", "-std=c11",
        "-fno-strict-aliasing", "-fwrapv", "-fno-aggressive-loop-optimizations",
        "-c", "crypto_multiplier.c", "-o", "crypto_multiplier.o"
    ]
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        print("[SUCCESS] C Module compiled to object file with ZERO errors.")
        if result.stderr:
            print(f"[COMPILER WARNINGS]:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[COMPILE FAILURE] Architecture error detected:\n{e.stderr}")
        return False

# Execute simulation block
test_compilation_pipeline()
