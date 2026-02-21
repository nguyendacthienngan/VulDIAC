import os
import glob
import argparse
import subprocess

# ==============================
# CONFIG
# ==============================
JOERN_HOME = os.path.expanduser("~/joern-2.0.72")
JOERN_BIN = os.path.join(JOERN_HOME, "joern")

# NEW batch script
SCRIPT_PATH = "/home/ngan/Downloads/VulDiac/archive/VulDIAC/scripts/export_batch.sc"

# JVM heap
os.environ["_JAVA_OPTIONS"] = "-Xmx10g"


# ==============================
# Args
# ==============================
def parse_options():
    parser = argparse.ArgumentParser(
        description="Joern Batch Export (Single Session)"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Directory containing .bin CPGs"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory"
    )

    return parser.parse_args()


# ==============================
# Batch Export (ONE JOERN RUN)
# ==============================
def run_batch_export(bin_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    print("\n[+] Starting SINGLE Joern session batch export")
    print(f"[+] Input bins : {bin_dir}")
    print(f"[+] Output dir : {out_dir}")

    cmd = [
        JOERN_BIN,
        "--script", SCRIPT_PATH,
        "--param", f"cpgDir={bin_dir}",
        "--param", f"outRoot={out_dir}",
    ]

    subprocess.run(cmd, check=True)


# ==============================
# MAIN
# ==============================
def main():
    args = parse_options()

    bin_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)

    run_batch_export(bin_dir, out_dir)

    print("\n✅ Batch export finished.")


if __name__ == "__main__":
    main()