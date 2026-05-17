"""Convert a raw Tinker-format Nemotron LoRA adapter into PEFT/Kaggle format.

Replicates the asalhi/tinker-adapter-to-ready-to-submit-adapter notebook locally:
calls ``tinker_cookbook.weights.build_lora_adapter`` with the asalhi patch that
SVD-compresses fused projections back to rank 32 (the competition cap).

Outputs ``adapter_config.json`` + ``adapter_model.safetensors`` and a zipped
``submission.zip`` ready for upload.

Example
-------
    python convert_tinker_adapter.py \\
        --base-model ./models/nemotron-3-nano-30b-a3b-bf16 \\
        --adapter-path ./adapters/huikang-v27 \\
        --output-dir ./adapters/huikang-v27-peft
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


FORCED_FUSED_RANK = 32


def _compress_lora_pair_to_rank(B, A_mat, rank: int):
    """SVD-compress a LoRA pair (B @ A) down to ``rank``.

    Lifted verbatim from asalhi's notebook. Keeps the rank-32 contract after
    Tinker's fused projections are un-fused into per-module LoRAs.
    """
    import torch

    delta = B.float() @ A_mat.float()
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]
    sroot = torch.sqrt(S)
    B_new = U * sroot.unsqueeze(0)
    A_new = sroot.unsqueeze(1) * Vh
    return B_new.to(B.dtype).contiguous(), A_new.to(A_mat.dtype).contiguous()


def patched_merge_fused_projections(
    fused_model_key: str,
    adapter_layer_prefix: str,
    components,
    model_state_shapes,
    peft_weights,
    target_modules,
    profile,
) -> int:
    """asalhi's monkey-patch: merge fused projections, enforce rank-32 ceiling.

    The stock tinker-cookbook implementation lets effective rank exceed 32 when
    Q/K/V (or gate/up) projections are fused. The Kaggle scorer rejects that.
    This version applies the SVD compression in :func:`_compress_lora_pair_to_rank`
    after merging.
    """
    import torch

    import tinker_cookbook.weights._adapter as A

    fused_out_dim = model_state_shapes[fused_model_key][0]
    fused_target_name = fused_model_key.removesuffix(".weight").rsplit(".", 1)[-1]

    component_order = None
    for target, comps in profile.fused_projection_map:
        if target == fused_target_name:
            component_order = comps
            break
    assert component_order is not None, f"No fused projection map for {fused_target_name!r}"

    comp_by_name = {name: (lora_A, lora_B) for name, lora_A, lora_B in components}

    lora_A_parts: list = []
    comp_slices: list[tuple[int, int, int]] = []
    merged_rank = 0
    row_offset = 0

    for comp_name in component_order:
        if comp_name not in comp_by_name:
            raise RuntimeError(
                f"Missing component {comp_name!r} for fused target {fused_model_key!r}"
            )
        lora_A, lora_B = comp_by_name[comp_name]
        r = lora_A.shape[0]
        out_dim = lora_B.shape[0]

        lora_A_parts.append(lora_A)
        comp_slices.append((row_offset, row_offset + out_dim, r))
        row_offset += out_dim
        merged_rank += r

    merged_lora_A = torch.cat(lora_A_parts, dim=0)
    merged_lora_B = torch.zeros(
        fused_out_dim, merged_rank, dtype=merged_lora_A.dtype, device=merged_lora_A.device
    )

    rank_offset = 0
    for i, (row_start, row_end, r) in enumerate(comp_slices):
        _, lora_B = comp_by_name[component_order[i]]
        merged_lora_B[row_start:row_end, rank_offset : rank_offset + r] = lora_B
        rank_offset += r

    final_rank = merged_rank
    if merged_rank > FORCED_FUSED_RANK:
        merged_lora_B, merged_lora_A = _compress_lora_pair_to_rank(
            merged_lora_B, merged_lora_A, FORCED_FUSED_RANK
        )
        final_rank = FORCED_FUSED_RANK

    peft_target_key = f"{adapter_layer_prefix}.{fused_target_name}.weight"
    A._add_peft_weight(peft_target_key, merged_lora_A, merged_lora_B, peft_weights, target_modules)
    return final_rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True,
                   help="HF model id OR local path to nemotron-3-nano-30b-a3b-bf16 (needed for shape lookups).")
    p.add_argument("--adapter-path", required=True,
                   help="Path to a raw Tinker-format adapter folder (e.g. huikang/nemotron-adapter/v27).")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the PEFT-format adapter + submission.zip.")
    p.add_argument("--skip-zip", action="store_true",
                   help="Skip writing submission.zip (write only the adapter files).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model:     {args.base_model}")
    print(f"Source adapter: {args.adapter_path}")
    print(f"Output dir:     {output_dir}")
    print(f"FORCED_FUSED_RANK = {FORCED_FUSED_RANK}")

    # Install monkey-patch BEFORE importing the high-level weights API so
    # tinker-cookbook picks up the rank-32 enforcement on its first call.
    import tinker_cookbook.weights._adapter as A
    A._merge_fused_projections = patched_merge_fused_projections
    print(f"Patched tinker_cookbook.weights._adapter._merge_fused_projections "
          f"(SVD compress -> rank {FORCED_FUSED_RANK})")

    from tinker_cookbook import weights

    weights.build_lora_adapter(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_path=str(output_dir),
    )
    print(f"Wrote PEFT adapter to {output_dir}")

    required = {"adapter_config.json", "adapter_model.safetensors"}
    present = {p.name for p in output_dir.iterdir() if p.is_file()}
    missing = required - present
    if missing:
        raise RuntimeError(f"Conversion incomplete; missing: {sorted(missing)}")

    if args.skip_zip:
        print("--skip-zip set; not packaging submission.zip.")
        return

    zip_path = output_dir / "submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name in sorted(required):
            zf.write(output_dir / name, arcname=name)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        sizes = {n: zf.getinfo(n).file_size for n in names}
    print(f"\nWrote {zip_path}")
    for n in names:
        print(f"  {n}: {sizes[n] / 1024**3:.3f} GiB")


if __name__ == "__main__":
    main()
