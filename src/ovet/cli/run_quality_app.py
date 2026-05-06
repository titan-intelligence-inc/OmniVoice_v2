"""Focused listening app for the quality_test_* runs.

Layout: 4 variants (baseline / phase4 / accent_strong / accent_quality)
× 3 reps audio matrix, plus a metric table. A cell selector at the top
lets the user switch between (lang, emotion) datasets that exist on disk.

Designed to make A/B/C/D comparison effortless.

Example:
    python -m ovet.cli.run_quality_app --share
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import gradio as gr


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def _resolve(p: str | Path, root: Path) -> str:
    pp = Path(p)
    return str(pp if pp.is_absolute() else (root / pp).resolve())


def discover_cells(outputs_dir: Path) -> list[Path]:
    """Find every ``quality_test_*/result.json`` in the outputs tree."""
    return sorted(p.parent for p in outputs_dir.glob("quality_test_*/result.json"))


def load_cell(result_json: Path) -> dict:
    with open(result_json, encoding="utf-8") as f:
        data = json.load(f)
    by_variant = {row["name"]: row for row in data["rows"]}
    return {
        "lang": data["lang"],
        "emotion": data["emotion"],
        "reps": int(data["reps"]),
        "by_variant": by_variant,
        "variants": [r["name"] for r in data["rows"]],
        "result_path": str(result_json),
    }


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

_VARIANT_DESCRIPTIONS = {
    "baseline":              "OmniVoice native (alpha=0)",
    "phase4":                "α=1.0, accent off",
    "phase4_accent_strong":  "anchor: α=1.0, accent=1.0, layers [8,12,16], all 32 steps",
    "phase4_accent_quality": "α=1.0, accent=0.5, layer=[12], step_window=(0,16), norm_clip=0.3",
    "strong_step_half":      "anchor + step_window=(0,16) (first half of 32 steps)",
    "strong_step_quarter":   "anchor + step_window=(0,8) (first quarter)",
    "strong_clip_05":        "anchor + norm_clip=0.5 (didn't bite — same as anchor)",
    "strong_step_clip":      "anchor + step_window=(0,16) + norm_clip=0.5",
    # Round 3 — gentle continuous push instead of strong pulsed push
    "strong_clip_03":        "anchor + norm_clip=0.3 (each step's delta capped at 30 % of |h|)",
    "strong_clip_02":        "anchor + norm_clip=0.2 (cap 20 % — should bite hardest)",
    "strong_a15_clip03":     "overdrive: accent=1.5 + norm_clip=0.3",
    "strong_a20_clip02":     "overdrive: accent=2.0 + norm_clip=0.2",
    # Round 4 — localized + overdriven push
    "strong_qtr_a15":        "step=(0,8) + accent=1.5  (overdrive in 8-step window)",
    "strong_qtr_a20":        "step=(0,8) + accent=2.0",
    "strong_qtr_a30":        "step=(0,8) + accent=3.0  (heavy — broke output)",
    "strong_step_8th_a15":   "step=(0,4) + accent=1.5  (4-step window + slight overdrive)",
    "strong_step_8th_a20":   "step=(0,4) + accent=2.0  ← ROUND 4 WINNER on metrics",
    "strong_step_8th_a25":   "step=(0,4) + accent=2.5  (edge of stability)",
    # Round 5 — position_mask (apply only at audio token positions, not text)
    "strong_8th_a20_pmask":  "round-4 winner + position_mask (audio tokens only)",
    "strong_pmask":          "anchor + position_mask",
    "strong_8th_a30_pmask":  "step=(0,4) + accent=3.0 + position_mask (heavy)",
}


def _fmt_metric_table(by_variant: dict, variants: list[str]) -> str:
    """Markdown table comparing variants on key metrics."""
    out = ["| variant | vad_dist | V_d | E_ratio | spk_sim | CER med |",
           "|---|---|---|---|---|---|"]
    for name in variants:
        row = by_variant.get(name)
        if not row:
            continue
        m = row["metrics"]
        out.append(
            f"| `{name}` | {m['vad_dist']['mean']:.3f}±{m['vad_dist']['std']:.3f} | "
            f"{m['valence_diff']['mean']:.3f} | "
            f"{m['energy_std_ratio']['mean']:.3f} | "
            f"{m['speaker_sim']['mean']:.3f} | "
            f"{m['content_error']['median']:.3f} |"
        )
    return "\n".join(out)


def _delta_table(by_variant: dict, variants: list[str]) -> str:
    """Show Δ vs phase4_accent_strong (the one user found unnatural)."""
    ref = by_variant.get("phase4_accent_strong")
    if not ref:
        return ""
    rm = ref["metrics"]
    rows = ["**Δ vs phase4_accent_strong** (negative for vad/V_d/CER and "
            "|1−E_r| are improvements):",
            "",
            "| variant | Δvad | ΔE_r toward 1 | Δspk |",
            "|---|---|---|---|"]
    for name in variants:
        if name == "phase4_accent_strong":
            continue
        row = by_variant.get(name)
        if not row:
            continue
        m = row["metrics"]
        d_vad = m["vad_dist"]["mean"]    - rm["vad_dist"]["mean"]
        # E_r distance from 1.0 (smaller |1-E_r| = closer to natural ratio)
        d_er  = abs(1 - m["energy_std_ratio"]["mean"]) - abs(1 - rm["energy_std_ratio"]["mean"])
        d_spk = m["speaker_sim"]["mean"] - rm["speaker_sim"]["mean"]
        rows.append(f"| `{name}` | {d_vad:+.3f} | {d_er:+.3f} | {d_spk:+.3f} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------

def make_app(outputs_dir: Path, project_root: Path):
    cell_dirs = discover_cells(outputs_dir)
    if not cell_dirs:
        raise FileNotFoundError(f"No quality_test_* dirs under {outputs_dir}")

    # Load all cells once
    cells_by_key: dict[str, dict] = {}
    for d in cell_dirs:
        c = load_cell(d / "result.json")
        key = f"{c['lang']}/{c['emotion']}"
        cells_by_key[key] = c

    cell_keys = sorted(cells_by_key.keys())
    print(f"[ovet] discovered cells: {cell_keys}", flush=True)

    # Auto-discover variants from the data, in stable order across cells.
    # Preferred display order: baseline first, then anchor (strong), then quality variants.
    PREFERRED_ORDER = [
        "baseline", "phase4",
        "phase4_accent_strong",                # anchor
        # Round 2 (step-window family)
        "strong_step_half", "strong_step_quarter",
        # Round 4 (localized overdrive)
        "strong_step_8th_a20",
        # Round 5 (position mask — TODAY'S CANDIDATES)
        "strong_8th_a20_pmask", "strong_pmask", "strong_8th_a30_pmask",
        # Other round 4 / 5 variants
        "strong_step_8th_a15", "strong_step_8th_a25",
        "strong_qtr_a15", "strong_qtr_a20", "strong_qtr_a30",
        # Round 3 (clip family — push every step but capped)
        "strong_clip_05", "strong_clip_03", "strong_clip_02",
        "strong_a15_clip03", "strong_a20_clip02",
        # Round 1 (single-layer + low alpha — accent fully lost)
        "strong_step_clip", "phase4_accent_quality",
    ]
    seen_variants: list[str] = []
    for c in cells_by_key.values():
        for v in c["variants"]:
            if v not in seen_variants:
                seen_variants.append(v)
    DISPLAY_VARIANTS = (
        [v for v in PREFERRED_ORDER if v in seen_variants]
        + [v for v in seen_variants if v not in PREFERRED_ORDER]
    )
    print(f"[ovet] display variants: {DISPLAY_VARIANTS}", flush=True)
    MAX_REPS = max(c["reps"] for c in cells_by_key.values())

    def update(cell_key: str):
        c = cells_by_key.get(cell_key)
        if not c:
            return [None] * (1 + len(DISPLAY_VARIANTS) * MAX_REPS) + ["", "", ""]
        ref = _resolve(
            f"baseline/jvnv_samples/jvnv_F1_{c['emotion']}.wav", project_root,
        )

        # Audio paths: 4 variants × MAX_REPS players
        audio_paths = []
        for v in DISPLAY_VARIANTS:
            row = c["by_variant"].get(v)
            for r in range(MAX_REPS):
                if row and r < c["reps"]:
                    wav = row["per_rep"][r].get("wav")
                    audio_paths.append(_resolve(wav, project_root) if wav else None)
                else:
                    audio_paths.append(None)

        info_md = (
            f"### {cell_key}\n\n"
            f"Loaded from: `{c['result_path']}`\n\n"
            f"reps per variant: {c['reps']}"
        )
        metric_md = _fmt_metric_table(c["by_variant"], DISPLAY_VARIANTS)
        delta_md = _delta_table(c["by_variant"], DISPLAY_VARIANTS)

        return [ref] + audio_paths + [info_md, metric_md, delta_md]

    with gr.Blocks(title="ovet — quality A/B listening") as app:
        gr.Markdown(
            "# ovet — accent-quality listening comparison\n"
            "Compare 4 variants × 3 reps for a single (lang, emotion) cell. "
            "The reference player on top is the JVNV F1 source. The variants "
            "differ in steering parameters as labelled."
        )

        cell_dd = gr.Dropdown(
            choices=cell_keys,
            value=cell_keys[0] if cell_keys else None,
            label="Cell (language / emotion)",
        )

        info_md = gr.Markdown()

        gr.Markdown("## Reference (JVNV F1)")
        ref_player = gr.Audio(label="reference", type="filepath", interactive=False)

        # Variant columns × rep rows
        all_players: list[gr.Audio] = []
        with gr.Row():
            for v in DISPLAY_VARIANTS:
                with gr.Column():
                    desc = _VARIANT_DESCRIPTIONS.get(v, v)
                    gr.Markdown(f"### `{v}`\n_{desc}_")
                    for r in range(MAX_REPS):
                        p = gr.Audio(type="filepath", interactive=False,
                                     label=f"rep {r}")
                        all_players.append(p)

        gr.Markdown("## Metrics")
        metrics_md = gr.Markdown()
        delta_md   = gr.Markdown()

        outputs = [ref_player] + all_players + [info_md, metrics_md, delta_md]
        cell_dd.change(update, [cell_dd], outputs)
        app.load(update, [cell_dd], outputs)

    return app


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Quality A/B listening Gradio app")
    ap.add_argument("--outputs-dir", default=Path("outputs"), type=Path,
                    help="Where to discover quality_test_* directories")
    ap.add_argument("--project-root", default=None, type=Path,
                    help="Project root (default: cwd)")
    ap.add_argument("--port",        type=int, default=7860)
    ap.add_argument("--server-name", default="0.0.0.0")
    ap.add_argument("--share",       action="store_true")
    ap.add_argument("--hf-home",     default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    project_root = (args.project_root or Path.cwd()).resolve()
    print(f"[ovet] project root: {project_root}", flush=True)
    print(f"[ovet] discovering quality_test_* in {args.outputs_dir} ...", flush=True)

    app = make_app(args.outputs_dir.resolve(), project_root)

    print(f"[ovet] launching on {args.server_name}:{args.port} (share={args.share})",
          flush=True)
    _, local_url, share_url = app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(project_root)],
        prevent_thread_lock=True,
    )
    print("=" * 60, flush=True)
    print(f"[ovet] LOCAL URL: {local_url}", flush=True)
    if share_url:
        print(f"[ovet] SHARE URL: {share_url}", flush=True)
    print("=" * 60, flush=True)
    Path("outputs/quality_app_urls.txt").write_text(
        f"local: {local_url}\nshare: {share_url or '(disabled)'}\n",
        encoding="utf-8",
    )
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
