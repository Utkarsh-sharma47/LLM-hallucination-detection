"""Consolidate calibration shard files into unified features and dataset."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SHARDS_DIR = ROOT / "calibration-batch-001-qwen0.5b" / "calibration-batch-001-qwen0.5b"
DECISIONS_DIR = (
    ROOT
    / "artifacts"
    / "reviews"
    / "task7-halueval-generation-2400-001-ai-v1"
    / "calibration-batch-001-qwen0.5b-ai-v1"
)
OUTPUT_DIR = ROOT / "artifacts" / "calibration-data"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading shards from: {SHARDS_DIR}")
    decisions = []
    for p in sorted(DECISIONS_DIR.glob("decisions-*.json")):
        decisions.extend(json.loads(p.read_text(encoding="utf-8")))
    decisions.sort(key=lambda d: d[0])

    all_rows = []
    vector_bytes_list = []
    qid_list = []

    for shard_idx in range(8):
        json_path = SHARDS_DIR / f"shard-{shard_idx:03d}.json"
        npz_path = SHARDS_DIR / f"shard-{shard_idx:03d}.npz"

        shard_data = json.loads(json_path.read_text(encoding="utf-8"))
        rows = shard_data["rows"]
        all_rows.extend(rows)

        with zipfile.ZipFile(npz_path) as zf:
            with zf.open("vectors.npy") as vf:
                _magic = vf.read(8)
                header_len = int.from_bytes(vf.read(2), "little")
                _header = ast.literal_eval(vf.read(header_len).decode("latin1").strip())
                # Read raw float32 bytes (100 * 896 * 4 bytes)
                raw_bytes = vf.read()
                vector_bytes_list.append(raw_bytes)

            with zf.open("question_id.npy") as qf:
                _magic = qf.read(8)
                q_header_len = int.from_bytes(qf.read(2), "little")
                _q_header = ast.literal_eval(qf.read(q_header_len).decode("latin1").strip())
                q_raw = qf.read()
                # Extract ASCII string array entries
                stride = len(q_raw) // len(rows)
                for i in range(len(rows)):
                    qid = q_raw[i * stride : (i + 1) * stride].decode("ascii", errors="ignore").strip("\x00")
                    qid_list.append(qid)

    print(f"Total rows collected: {len(all_rows)}")
    print(f"Total decisions matched: {len(decisions)}")

    # Merge rows with reviewed decisions
    merged_rows = []
    binary_rows = []
    for row, dec, qid in zip(all_rows, decisions, qid_list, strict=True):
        idx, label, quote, note, flags = dec
        merged = dict(row)
        merged["reviewed_label"] = label
        merged["evidence_quote"] = quote
        merged["evidence_note"] = note
        merged["flags"] = flags
        merged["question_id"] = qid
        merged_rows.append(merged)

        if label in {"FAITHFUL", "HALLUCINATED"}:
            binary_rows.append(
                {
                    "source_index": idx,
                    "question_id": qid,
                    "group_id": row["group_id"],
                    "split": row["split"],
                    "reviewed_label": label,
                    "target": 0 if label == "FAITHFUL" else 1,
                }
            )

    # Save consolidated dataset JSON
    dataset_file = OUTPUT_DIR / "calibration_reviewed_dataset.json"
    dataset_file.write_text(json.dumps(merged_rows, indent=2), encoding="utf-8")

    # Save binary index JSONL
    binary_index_file = OUTPUT_DIR / "calibration_binary_index.jsonl"
    with binary_index_file.open("w", encoding="utf-8") as f:
        for b in binary_rows:
            f.write(json.dumps(b) + "\n")

    # Save consolidated npz containing the combined 800 vectors
    total_vectors_bytes = b"".join(vector_bytes_list)
    output_npz = OUTPUT_DIR / "calibration_features.npz"
    with zipfile.ZipFile(output_npz, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        # Create valid .npy header for (800, 896) float32
        hdr_dict = {"descr": "<f4", "fortran_order": False, "shape": (len(all_rows), 896)}
        hdr_str = str(hdr_dict) + "   \n"
        hdr_len = len(hdr_str)
        pad = (16 - ((10 + hdr_len) % 16)) % 16
        hdr_str = str(hdr_dict) + " " * (pad + 3) + "\n"
        hdr_bytes = hdr_str.encode("latin1")
        npy_header = b"\x93NUMPY\x01\x00" + len(hdr_bytes).to_bytes(2, "little") + hdr_bytes
        zout.writestr("vectors.npy", npy_header + total_vectors_bytes)

    print(f"\nSuccessfully assembled:")
    print(f"  Dataset:      {dataset_file} ({len(merged_rows)} rows)")
    print(f"  Binary Index: {binary_index_file} ({len(binary_rows)} candidates)")
    print(f"  Vectors:      {output_npz} ({len(all_rows)} vectors of dim 896)")


if __name__ == "__main__":
    main()

