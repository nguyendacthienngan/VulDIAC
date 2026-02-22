import json
import os
import argparse
from tqdm import tqdm


def extract_code_and_label(entry):
    """
    Try common dataset formats.
    Modify here if your JSON structure differs.
    """

    # BigVul style
    if "func" in entry:
        code = entry["func"]
        label = entry.get("target", entry.get("label", 0))

    # generic
    elif "code" in entry:
        code = entry["code"]
        label = entry.get("target", entry.get("label", 0))

    else:
        return None, None

    return code, int(label)


def split_dataset(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Some datasets wrap list inside dict
    if isinstance(data, dict):
        for key in ["data", "functions", "items"]:
            if key in data:
                data = data[key]
                break

    print(f"Total samples: {len(data)}")

    written = 0

    for i, entry in enumerate(tqdm(data)):

        try:
            code, label = extract_code_and_label(entry)

            if not code or len(code.strip()) == 0:
                continue

            filename = f"sample_{i}_{label}.c"
            path = os.path.join(output_dir, filename)

            with open(path, "w", encoding="utf-8") as f:
                f.write(code.strip() + "\n")

            written += 1

        except Exception as e:
            print(f"Skipped sample {i}: {e}")

    print("\nDone.")
    print("Files written:", written)
    print("Output folder:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to dataset JSON")
    parser.add_argument("output_dir", help="Directory to write .c files")
    args = parser.parse_args()

    split_dataset(args.json_path, args.output_dir)
