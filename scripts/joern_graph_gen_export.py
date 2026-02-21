import os
import glob
import argparse
import subprocess
from multiprocessing import Pool
import shutil
from functools import partial

# ===== CONFIG =====
JOERN_HOME = os.path.expanduser("~/joern-2.0.72")
JOERN_PATH = JOERN_HOME
JOERN_EXPORT = os.path.join(JOERN_HOME, "joern-export")
SCRIPT_PATH = "/home/ngan/Downloads/VulDiac/archive/VulDIAC/scripts/export_per_func.sc"
# JVM heap (máy bạn 14GB free → 8-10GB là đẹp)
os.environ["_JAVA_OPTIONS"] = "-Xmx10g"


def parse_options():
    parser = argparse.ArgumentParser(description="Joern Graph Generator (Joern 2.x)")
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-t", "--type", choices=["parse", "export"], required=True)
    parser.add_argument("--pool", type=int, default=2)
    return parser.parse_args()


# ------------------------------------
# Parse C files → CPG .bin
# ------------------------------------
def joern_parse(file, outdir):
    name = os.path.basename(file).replace(".c", "")
    out_bin = os.path.join(outdir, name + ".bin")

    if os.path.exists(out_bin):
        print(f"[Skip parse] {name}")
        return

    print(f"[Parse] {name}")

    cmd = [
        os.path.join(JOERN_PATH, "joern-parse"),
        file,
        "--language", "c",
        "-o", out_bin
    ]

    subprocess.run(cmd)


# ------------------------------------
# Export per-function DOT
# ------------------------------------
def joern_export(bin_file, outdir):
    name = os.path.basename(bin_file).replace(".bin", "")
    func_outdir = os.path.join(outdir, name)

    if os.path.exists(func_outdir):
        print(f"[Skip export] {name}")
        return

    os.makedirs(func_outdir, exist_ok=True)

    print(f"[Export] {name}")

    # Joern version 1.x
    # cmd = [
    #     os.path.join(JOERN_PATH, "joern"),
    #     "--script", SCRIPT_PATH,
    #     "--params",
    #     f"cpgFile={bin_file}",
    #     f"outDir={func_outdir}"
    # ]

    # Joern version 2.x
    cmd = [
        os.path.join(JOERN_PATH, "joern"),
        "--script", SCRIPT_PATH,
        "--param", f"cpgFile={bin_file}",
        "--param", f"outDir={func_outdir}"
    ]

    subprocess.run(cmd)


# ------------------------------------
# Main
# ------------------------------------
def main():
    args = parse_options()

    input_path = args.input.rstrip("/") + "/"
    output_path = args.output.rstrip("/") + "/"

    os.makedirs(output_path, exist_ok=True)

    pool = Pool(args.pool)

    if args.type == "parse":
        files = glob.glob(input_path + "*.c")
        pool.map(partial(joern_parse, outdir=output_path), files)

    elif args.type == "export":
        bins = glob.glob(input_path + "*.bin")
        pool.map(partial(joern_export, outdir=output_path), bins)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()