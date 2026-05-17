"""Audit a candidate PEFT adapter against the 0.86 kien reference structure.

Reads only the safetensors header (cheap, no GPU, no model load). Checks the
properties that historically distinguish a valid submission from a silently
broken one:

  * adapter_config.json: rank, alpha, dropout, target_modules, peft_type
  * adapter_model.safetensors: tensor count, MoE-untying (128 per-expert copies),
    Mamba in_proj presence, lm_head presence, key prefix conventions

Use against any candidate before spending a Kaggle daily submission slot.

Example
-------
    python verify_adapter.py --adapter ./adapters/huikang-v27-peft
    python verify_adapter.py --adapter ./adapters/huikang-v27-peft \\
        --reference ./adapters/kien-tinker
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_FILES = {"adapter_config.json", "adapter_model.safetensors"}

# Expected structural fingerprint of the 0.86 kien tinker-adapter, derived
# from header inspection. Counts that diverge dramatically signal a broken
# conversion before the daily-submission slot is spent.
KIEN_FINGERPRINT = {
    "tensor_count": 12010,
    "experts_dot_count": 11776,   # = 23 MoE layers * 128 experts * 2 weight types * 2 (A+B)
    "in_proj_count": 46,          # 23 Mamba layers * 2 (A+B), no gate_proj / x_proj
    "lm_head_count": 2,
    "size_gib_min": 3.0,
    "size_gib_max": 4.0,
}


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return header, header_len


def check_config(adapter_dir: Path) -> dict:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    expected = {
        "peft_type": "LORA",
        "r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
    }
    findings = {}
    for k, v in expected.items():
        actual = cfg.get(k)
        ok = actual == v
        findings[k] = (ok, actual, v)
    return findings


def fingerprint_safetensors(adapter_dir: Path) -> dict:
    st_path = adapter_dir / "adapter_model.safetensors"
    hdr, hdr_len = read_header(st_path)
    tensors = [k for k in hdr if k != "__metadata__"]
    size_bytes = st_path.stat().st_size

    markers = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj", "gate_up_proj",
               "in_proj", "out_proj", "x_proj",
               "lm_head", "experts.", ".experts.",
               ".w1.", ".w2.", ".w3.",
               "backbone", "base_model.model.model"]
    marker_counts = {m: sum(1 for k in tensors if m in k) for m in markers}

    # Per-expert untying: count unique expert indices
    expert_indices: Counter = Counter()
    for k in tensors:
        if ".experts." not in k:
            continue
        parts = k.split(".experts.")
        if len(parts) < 2:
            continue
        idx_str = parts[1].split(".")[0]
        if idx_str.isdigit():
            expert_indices[idx_str] += 1

    return {
        "size_bytes": size_bytes,
        "size_gib": size_bytes / 1024**3,
        "header_len": hdr_len,
        "tensor_count": len(tensors),
        "marker_counts": marker_counts,
        "n_unique_expert_indices": len(expert_indices),
        "first_tensors": [(k, hdr[k].get("dtype"), hdr[k].get("shape"))
                          for k in tensors[:6]],
    }


def compare_to_reference(candidate: dict, reference: dict) -> list[str]:
    notes: list[str] = []
    c_count = candidate["tensor_count"]
    r_count = reference["tensor_count"]
    notes.append(f"tensor count: candidate={c_count} reference={r_count} "
                 f"(delta={c_count - r_count:+d})")

    c_markers = candidate["marker_counts"]
    r_markers = reference["marker_counts"]
    for m in sorted(set(c_markers) | set(r_markers)):
        cv = c_markers.get(m, 0)
        rv = r_markers.get(m, 0)
        if cv != rv:
            notes.append(f"marker '{m}': candidate={cv} reference={rv} (delta={cv-rv:+d})")

    ce = candidate["n_unique_expert_indices"]
    re_ = reference["n_unique_expert_indices"]
    if ce != re_:
        notes.append(f"unique expert indices: candidate={ce} reference={re_}")
    return notes


def check_against_fingerprint(candidate: dict) -> list[str]:
    notes: list[str] = []
    fp = KIEN_FINGERPRINT
    if candidate["tensor_count"] != fp["tensor_count"]:
        notes.append(f"FAIL tensor_count: got {candidate['tensor_count']}, "
                     f"expected {fp['tensor_count']}")
    if candidate["marker_counts"].get(".experts.", 0) != fp["experts_dot_count"]:
        notes.append(f"FAIL .experts. count: got {candidate['marker_counts'].get('.experts.', 0)}, "
                     f"expected {fp['experts_dot_count']} "
                     f"(suggests MoE wasn't untied to 128 per-expert copies)")
    if candidate["marker_counts"].get("in_proj", 0) != fp["in_proj_count"]:
        notes.append(f"WARN in_proj count: got {candidate['marker_counts'].get('in_proj', 0)}, "
                     f"expected {fp['in_proj_count']} "
                     f"(Mamba gate_proj/x_proj should be merged into in_proj)")
    if candidate["marker_counts"].get("lm_head", 0) != fp["lm_head_count"]:
        notes.append(f"WARN lm_head LoRA tensors: got {candidate['marker_counts'].get('lm_head', 0)}, "
                     f"expected {fp['lm_head_count']} (Unsloth drops lm_head for MoE — manual re-add needed)")
    if candidate["marker_counts"].get(".w1.", 0) > 0:
        notes.append(f"FAIL raw Tinker w1/w2/w3 keys present — conversion was not applied")
    sz = candidate["size_gib"]
    if not (fp["size_gib_min"] <= sz <= fp["size_gib_max"]):
        notes.append(f"WARN safetensors size {sz:.2f} GiB outside expected "
                     f"[{fp['size_gib_min']}, {fp['size_gib_max']}] GiB")
    return notes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True,
                   help="Path to candidate adapter folder (must contain adapter_config.json + safetensors).")
    p.add_argument("--reference", default=None,
                   help="Optional reference adapter folder (e.g. kien tinker-adapter) for diff.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on any FAIL/WARN.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter = Path(args.adapter).absolute()
    missing = REQUIRED_FILES - {p.name for p in adapter.iterdir() if p.is_file()}
    if missing:
        raise SystemExit(f"Adapter dir missing: {sorted(missing)}")

    print(f"=== {adapter} ===")
    print("[adapter_config.json]")
    for k, (ok, actual, expected) in check_config(adapter).items():
        flag = "ok " if ok else "!! "
        print(f"  {flag}{k:18s} = {actual!r}  (expected {expected!r})")

    print("\n[adapter_model.safetensors fingerprint]")
    cand_fp = fingerprint_safetensors(adapter)
    print(f"  size              : {cand_fp['size_gib']:.3f} GiB ({cand_fp['size_bytes']:,} bytes)")
    print(f"  header length     : {cand_fp['header_len']:,} bytes")
    print(f"  tensor count      : {cand_fp['tensor_count']:,}")
    print(f"  unique expert idx : {cand_fp['n_unique_expert_indices']}")
    print(f"  first 6 tensors:")
    for name, dtype, shape in cand_fp["first_tensors"]:
        print(f"    {name}  {dtype} {shape}")

    print("\n  marker counts (non-zero):")
    for m, c in sorted(cand_fp["marker_counts"].items()):
        if c > 0:
            print(f"    {m:24s} {c}")

    print("\n[checks vs kien 0.86 fingerprint]")
    findings = check_against_fingerprint(cand_fp)
    if not findings:
        print("  all fingerprint checks pass — structurally equivalent to kien tinker-adapter")
    else:
        for f in findings:
            print(f"  {f}")

    if args.reference:
        ref = Path(args.reference).absolute()
        print(f"\n[diff vs reference {ref}]")
        ref_fp = fingerprint_safetensors(ref)
        for note in compare_to_reference(cand_fp, ref_fp):
            print(f"  {note}")
        if cand_fp["tensor_count"] == ref_fp["tensor_count"] and \
           cand_fp["marker_counts"] == ref_fp["marker_counts"]:
            print("  candidate structure matches reference exactly")

    if args.strict and findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
