import os
import re

SRC_DIR = "src"
DATASET_DIR = "dataset"

# Match anything ending in .csv inside quotes
CSV_PATTERN = re.compile(r'["\']([^"\']+\.csv)["\']')

# Collect all actual dataset files
dataset_files = set(os.listdir(DATASET_DIR))

missing = []
found = []

for root, _, files in os.walk(SRC_DIR):
    for file in files:
        if file.endswith(".py"):
            py_path = os.path.join(root, file)

            with open(py_path, "r", encoding="utf-8") as f:
                content = f.read()

            matches = CSV_PATTERN.findall(content)

            for ref in matches:
                filename = os.path.basename(ref)  # ignore path, just filename

                if filename in dataset_files:
                    found.append((py_path, filename))
                else:
                    missing.append((py_path, ref))

# Deduplicate
found = list(set(found))
missing = list(set(missing))

# Output
print("\n✅ Valid CSV references:")
for py_file, fname in found:
    print(f"{py_file} -> {fname}")

print("\n❌ Missing CSV references:")
for py_file, ref in missing:
    print(f"{py_file} -> {ref}")