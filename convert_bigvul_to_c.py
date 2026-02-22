import csv
import os
from multiprocessing import Pool, cpu_count
from functools import partial

# ==============================
# CONFIG
# ==============================

CSV_PATH = "bigvul.csv"
OUTPUT_ROOT = "joern_project/src"

CODE_COLUMN = "func_before"   # change if needed
ID_COLUMN = "id"
LABEL_COLUMN = "vul"          # optional

BATCH_SIZE = 5000             # rows per worker batch
FILES_PER_FOLDER = 1000       # avoid millions in one folder
N_WORKERS = max(cpu_count() - 1, 1)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


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

def write_batch(rows):
    for row in rows:
        try:
            idx = int(row[ID_COLUMN])
            code = clean_code(row[CODE_COLUMN])

            # optional filtering
            # if row[LABEL_COLUMN] != "1":
            #     continue

            folder_id = idx // FILES_PER_FOLDER
            folder = os.path.join(OUTPUT_ROOT, f"{folder_id:04d}")
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

def process_csv():

    pool = Pool(N_WORKERS)
    batch = []
    jobs = []

    with open(CSV_PATH, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row in reader:
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                jobs.append(pool.apply_async(write_batch, (batch,)))
                batch = []

        # remaining rows
        if batch:
            jobs.append(pool.apply_async(write_batch, (batch,)))

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
    process_csv()