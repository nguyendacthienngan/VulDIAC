import csv
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

# ==============================
# CONFIG
# ==============================

CODE_COLUMN = "func_before"   # change if needed
ID_COLUMN = "id"
LABEL_COLUMN = "vul"          # optional

BATCH_SIZE = 5000             # rows per worker batch
FILES_PER_FOLDER = 1000       # avoid millions in one folder
N_WORKERS = max(cpu_count() - 1, 1)

# ==============================
# CLEAN CODE FOR JOERN
# ==============================

def clean_code(code: str) -> str:
    """
    Make code Joern-safe
    """
    if not code:
        return ""

    # remove null bytes (Joern killer)
    code = code.replace("\x00", "")

    # ensure newline at end
    if not code.endswith("\n"):
        code += "\n"

    return code


# ==============================
# WRITE BATCH FUNCTION
# ==============================

def write_batch(rows, output_dir):
    for row in rows:
        try:
            idx = int(row[ID_COLUMN])
            code = clean_code(row[CODE_COLUMN])

            # optional filtering
            # if row[LABEL_COLUMN] != "1":
            #     continue

            folder_id = idx // FILES_PER_FOLDER
            folder = os.path.join(output_dir, f"{folder_id:04d}")
            os.makedirs(folder, exist_ok=True)

            filepath = os.path.join(folder, f"{idx}.c")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)

        except Exception:
            # skip broken rows silently
            continue

    return len(rows)


# ==============================
# STREAM + PARALLEL PIPELINE
# ==============================

def process_csv(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pool = Pool(N_WORKERS)
    batch = []
    jobs = []

    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row in reader:
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                jobs.append(pool.apply_async(write_batch, (batch,output_dir,)))
                batch = []

        # remaining rows
        if batch:
            jobs.append(pool.apply_async(write_batch, (batch,output_dir,)))

    # wait for completion
    total = 0
    for j in jobs:
        total += j.get()

    pool.close()
    pool.join()

    print(f"\n✅ Finished writing {total} files")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to dataset CSV")
    parser.add_argument("output_dir", help="Directory to write .c files")
    args = parser.parse_args()
    process_csv(args.csv_path, args.output_dir)