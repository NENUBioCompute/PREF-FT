"""
Aggregate per_class_metrics.csv (class_id=1) across all methods under results/.

For each subfolder in results/, reads:
    results/{method}/external_tests/GSE104942/per_class_metrics.csv

Extracts the row where class_id == 1, prepends the method name, and writes a
single summary CSV to:
    results/aggregated_class1_metrics.csv
"""

import csv
from pathlib import Path


def main():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    output_path = results_dir / "aggregated_class1_metrics.csv"

    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}")
        return

    rows = []
    headers = None
    missing = []
    skipped = []

    # Iterate over subdirectories (one per method), sorted for reproducibility
    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        csv_path = method_dir / "external_tests" / "GSE104942" / "per_class_metrics.csv"

        if not csv_path.is_file():
            missing.append(method_name)
            continue

        # Read the CSV
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            lines = list(reader)

        # Strip trailing empty rows
        while lines and all(cell.strip() == "" for cell in lines[-1]):
            lines.pop()

        if len(lines) < 2:
            skipped.append((method_name, "too few lines"))
            continue

        # First line is header
        file_headers = [h.strip() for h in lines[0]]
        if headers is None:
            headers = ["method"] + file_headers
        elif file_headers != headers[1:]:
            print(f"WARNING: header mismatch for {method_name}, using its own headers")
            print(f"  Expected: {headers[1:]}")
            print(f"  Got:      {file_headers}")

        # Find the row with class_id == 1
        found = False
        for row in lines[1:]:
            # Skip empty rows
            if all(cell.strip() == "" for cell in row):
                continue
            if row[0].strip() == "1":
                rows.append([method_name] + [cell.strip() for cell in row])
                found = True
                break

        if not found:
            skipped.append((method_name, "no class_id=1 row"))

    # Write output
    if headers is None:
        print("ERROR: no valid data found.")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Done! Aggregated {len(rows)} methods → {output_path}")

    if missing:
        print(f"\nMissing file ({len(missing)}): {', '.join(missing)}")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(f'{m}({r})' for m, r in skipped)}")


if __name__ == "__main__":
    main()
