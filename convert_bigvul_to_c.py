import csv
import os
import sys
from multiprocessing import Pool, cpu_count
import argparse

# =====================================================
# FIX BIG CSV FIELD LIMIT (BigVul requirement)
# =====================================================
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

# =====================================================
# CONFIG
# =====================================================

CODE_COLUMN = "func_before"   # change if needed
ID_COLUMN = "_auto_id"
LABEL_COLUMN = "vul"

BATCH_SIZE = 5000
FILES_PER_FOLDER = 1000
N_WORKERS = max(cpu_count() - 1, 1)

# =====================================================
# CLEAN + JOERN-STABLE CODE
# =====================================================

def clean_code(code: str, idx: int, label: str) -> str:
    """
    Prepare code for Joern parsing.
    Adds stable metadata header.
    """

    if not code:
        code = ""

    # remove null bytes (Joern crash cause)
    code = code.replace("\x00", "")

    header = f"""// BIGVUL_ID: {idx}
// VUL_LABEL: {label}

"""

    if not code.endswith("\n"):
        code += "\n"

    return header + code


# =====================================================
# WRITE BATCH (runs in each worker)
# =====================================================

def write_batch(rows, output_dir):

    created_dirs = set()  # cache directories per worker

    written = 0

    for row in rows:
        try:
            idx = int(row[ID_COLUMN])
            label = row.get(LABEL_COLUMN, "0")

            code = clean_code(
                row.get(CODE_COLUMN, ""),
                idx,
                label
            )

            # folder batching
            folder_id = idx // FILES_PER_FOLDER
            folder = os.path.join(output_dir, f"{folder_id:04d}")

            if folder not in created_dirs:
                os.makedirs(folder, exist_ok=True)
                created_dirs.add(folder)

            # Joern-stable filename
            filepath = os.path.join(
                folder,
                f"bigvul_{idx:07d}.c"
            )

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)

            written += 1

        except Exception:
            # skip corrupted rows silently
            continue

    return written


# =====================================================
# STREAM CSV + MULTIPROCESS PIPELINE
# =====================================================

def process_csv(csv_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    pool = Pool(N_WORKERS)
    jobs = []
    batch = []

    print(f"Using {N_WORKERS} workers")

    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for auto_id, row in enumerate(reader):
            row["_auto_id"] = auto_id
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                jobs.append(
                    pool.apply_async(write_batch, (batch, output_dir))
                )
                batch = []

        if batch:
            jobs.append(
                pool.apply_async(write_batch, (batch, output_dir))
            )

    total = 0
    for j in jobs:
        total += j.get()

    pool.close()
    pool.join()

    print(f"\n✅ Finished writing {total} Joern-ready C files")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert BigVul CSV into Joern-ready C files"
    )

    parser.add_argument(
        "csv_path",
        help="Path to BigVul CSV file"
    )

    parser.add_argument(
        "output_dir",
        help="Directory to store generated .c files"
    )

    args = parser.parse_args()

    process_csv(args.csv_path, args.output_dir)