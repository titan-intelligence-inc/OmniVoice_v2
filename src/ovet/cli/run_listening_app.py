"""Gradio listening app for the multi-language sweep.

A side-by-side player: pick (language, emotion); the app shows the
reference recording, the OmniVoice baseline output, the phase4 (steered
+ language-projected) output, plus per-cell metrics and the Δ between
the two strategies.

Designed for RunPod / headless servers — binds 0.0.0.0 by default and
supports ``--share`` for a public gradio.live URL.

Example:
    python -m ovet.cli.run_listening_app \
        --result-json outputs/multilang_F1_anger_sad_fear_happy_surprise_disgust/result.json \
        --share
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
    """Make a path absolute if it isn't already."""
    pp = Path(p)
    return str(pp if pp.is_absolute() else (root / pp).resolve())


def load_data(result_json_path: Path):
    with open(result_json_path, encoding="utf-8") as f:
        data = json.load(f)
    project_root = result_json_path.resolve().parent.parent.parent
    if not (project_root / "src" / "ovet").exists():
        # Fallback: cwd is the project root
        project_root = Path.cwd()

    cells = data["cells"]
    by_key = {(c["lang_code"], c["emotion"], c["strategy"]): c for c in cells}

    cfg            = data.get("config", {})
    lang_order     = cfg.get("language_order", [])
    lang_full_map  = cfg.get("language_full_name", {})
    emotions       = cfg.get("emotions", [])
    target_texts   = cfg.get("target_text", {})
    ref_audio_dir  = Path(cfg.get("ref_audio_dir", "baseline/jvnv_samples"))
    ref_speaker    = cfg.get("ref_speaker", "F1")
    ref_texts      = data.get("ref_texts", {})

    # Discover strategies in display order: prefer config order, fall back to
    # whatever appears in cells.
    strat_cfg = cfg.get("strategies") or []
    strat_in_cells = []
    for c in cells:
        if c["strategy"] not in strat_in_cells:
            strat_in_cells.append(c["strategy"])
    if strat_cfg:
        strategy_names = [s["name"] for s in strat_cfg if s["name"] in strat_in_cells]
        # append any orphans
        for s in strat_in_cells:
            if s not in strategy_names:
                strategy_names.append(s)
    else:
        strategy_names = strat_in_cells

    return {
        "by_key": by_key, "lang_order": lang_order,
        "lang_full_map": lang_full_map, "emotions": emotions,
        "target_texts": target_texts, "ref_texts": ref_texts,
        "ref_audio_dir": ref_audio_dir, "ref_speaker": ref_speaker,
        "project_root": project_root, "raw": data,
        "strategies": strategy_names,
    }


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def _fmt_metrics(m: dict, label: str) -> str:
    if not m:
        return f"### {label}\n_(missing)_"
    return (
        f"### {label}\n\n"
        f"| metric | mean ± std |\n|---|---|\n"
        f"| vad_dist | {m['vad_dist']['mean']:.3f} ± {m['vad_dist']['std']:.3f} |\n"
        f"| valence_diff | {m['valence_diff']['mean']:.3f} ± {m['valence_diff']['std']:.3f} |\n"
        f"| arousal_diff | {m['arousal_diff']['mean']:.3f} ± {m['arousal_diff']['std']:.3f} |\n"
        f"| f0_std_ratio | {m['f0_std_ratio']['mean']:.3f} |\n"
        f"| energy_std_ratio | {m['energy_std_ratio']['mean']:.3f} |\n"
        f"| e2v_cos | {m['e2v_cos']['mean']:.3f} |\n"
        f"| speaker_sim | {m['speaker_sim']['mean']:.3f} |\n"
        f"| content_error (CER) | {m['content_error']['mean']:.3f} |\n"
    )


def _fmt_delta(b: dict, p: dict) -> str:
    if not (b and p):
        return ""
    def _delta(k):
        return p[k]["mean"] - b[k]["mean"]
    arrow = lambda v, lower_better: ("✅" if (v < 0) == lower_better else "⚠️") if abs(v) >= 0.02 else "≈"
    rows = [
        ("vad_dist",       _delta("vad_dist"),       True),
        ("valence_diff",   _delta("valence_diff"),   True),
        ("energy_std_ratio→1", abs(1 - p["energy_std_ratio"]["mean"]) - abs(1 - b["energy_std_ratio"]["mean"]), True),
        ("e2v_cos",        _delta("e2v_cos"),        False),
        ("speaker_sim",    _delta("speaker_sim"),    False),
        ("content_error",  _delta("content_error"),  True),
    ]
    body = "\n".join(f"| {n} | {v:+.3f} | {arrow(v, lb)} |" for n, v, lb in rows)
    return f"### Δ phase4 − baseline\n\n| metric | Δ | verdict |\n|---|---|---|\n{body}\n"


# ---------------------------------------------------------------------
# Highlight presets — quick links to interesting cells
# ---------------------------------------------------------------------

def compute_highlights(by_key, lang_order, emotions, focus_strategy: str = "phase4"):
    """Find top improvements / regressions / broken cells.

    Compares ``focus_strategy`` against ``baseline``. Falls back to whichever
    non-baseline strategy is present if ``focus_strategy`` is missing.
    """
    cells = []
    for lang in lang_order:
        for emo in emotions:
            b = by_key.get((lang, emo, "baseline"))
            p = by_key.get((lang, emo, focus_strategy))
            if not (b and p):
                continue
            d_vad = p["agg_metrics"]["vad_dist"]["mean"] - b["agg_metrics"]["vad_dist"]["mean"]
            # Use median CER (robust to Whisper hallucination outliers).
            cer_p = p["agg_metrics"]["content_error"].get("median",
                       p["agg_metrics"]["content_error"]["mean"])
            cells.append({"lang": lang, "emo": emo, "d_vad": d_vad, "cer_p": cer_p})

    improved = sorted(cells, key=lambda c: c["d_vad"])[:5]
    regressed = sorted(cells, key=lambda c: -c["d_vad"])[:5]
    broken    = [c for c in cells if c["cer_p"] > 0.30]
    broken.sort(key=lambda c: -c["cer_p"])
    broken    = broken[:5]
    return improved, regressed, broken


def _highlight_choices(highlights):
    out = []
    for c in highlights:
        label = f"{c['lang']}/{c['emo']}  Δvad={c['d_vad']:+.3f}  CER={c['cer_p']:.2f}"
        out.append((label, f"{c['lang']}|{c['emo']}"))
    return out


# ---------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------

_STRATEGY_DESCRIPTIONS = {
    "baseline":      "OmniVoice native, alpha=0",
    "phase4":        "alpha=1.0 + v_lang projection",
    "phase4_accent": "phase4 + accent removal (language_alpha=1.0)",
}


def make_app(result_json_path: Path):
    state = load_data(result_json_path)
    project_root = state["project_root"]
    strategies   = state["strategies"]

    # Pick the "focus" strategy for highlights (the most-interventional one).
    focus = strategies[-1] if strategies else "phase4"
    improved, regressed, broken = compute_highlights(
        state["by_key"], state["lang_order"], state["emotions"], focus_strategy=focus
    )

    def update(lang_code: str, emotion: str):
        ref_path = _resolve(
            state["ref_audio_dir"] / f"jvnv_{state['ref_speaker']}_{emotion}.wav",
            project_root,
        )

        # Per-strategy paths and metrics
        strat_paths = []
        strat_metric_mds = []
        for s in strategies:
            c = state["by_key"].get((lang_code, emotion, s))
            path = _resolve(c["best_wav"], project_root) if c else None
            metrics = c["agg_metrics"] if c else {}
            label = f"{s} ({_STRATEGY_DESCRIPTIONS.get(s, s)})"
            strat_paths.append(path)
            strat_metric_mds.append(_fmt_metrics(metrics, label))

        # Δ rows: baseline vs each non-baseline strategy
        b = state["by_key"].get((lang_code, emotion, "baseline"))
        b_metrics = b["agg_metrics"] if b else {}
        delta_blocks = []
        for s in strategies:
            if s == "baseline":
                continue
            c = state["by_key"].get((lang_code, emotion, s))
            if not (c and b):
                continue
            d = _fmt_delta(b_metrics, c["agg_metrics"])
            if d:
                delta_blocks.append(f"### Δ {s} − baseline\n\n" + d.split("\n", 1)[1])
        delta_md = "\n\n".join(delta_blocks) if delta_blocks else ""

        text_md = (
            f"**Reference text** ({emotion}): {state['ref_texts'].get(emotion, '')}\n\n"
            f"**Target text** ({state['lang_full_map'].get(lang_code, lang_code)}): "
            f"{state['target_texts'].get(lang_code, '')}"
        )
        return (ref_path, text_md, *strat_paths, *strat_metric_mds, delta_md)

    def jump(combo: str):
        if not combo:
            return gr.update(), gr.update()
        try:
            lang, emo = combo.split("|", 1)
        except ValueError:
            return gr.update(), gr.update()
        return gr.update(value=lang), gr.update(value=emo)

    with gr.Blocks(title="ovet — listening comparison") as app:
        gr.Markdown(
            "# OmniVoice cross-lingual emotion preservation — listening comparison\n"
            "Pick a (language, emotion) and compare strategies side-by-side. "
            f"Strategies in this run: {', '.join('**'+s+'**' for s in strategies)}. "
            "Reference audio on top is the JVNV F1 source v_emo was built from."
        )

        with gr.Row():
            with gr.Column(scale=2):
                lang_dd = gr.Dropdown(
                    choices=[(f"{state['lang_full_map'].get(c, c)} ({c})", c)
                             for c in state["lang_order"]],
                    value=state["lang_order"][0] if state["lang_order"] else None,
                    label="Language",
                )
            with gr.Column(scale=2):
                emo_dd = gr.Dropdown(
                    choices=state["emotions"],
                    value=state["emotions"][0] if state["emotions"] else None,
                    label="Emotion",
                )

        with gr.Accordion(f"⚡ Quick jumps (sorted by Δvad vs baseline, focus={focus})",
                          open=True):
            with gr.Row():
                imp_dd = gr.Dropdown(
                    choices=_highlight_choices(improved), value=None,
                    label=f"Top 5 improvements ({focus} better than baseline)",
                )
                reg_dd = gr.Dropdown(
                    choices=_highlight_choices(regressed), value=None,
                    label=f"Top 5 regressions ({focus} worse than baseline)",
                )
                brk_dd = gr.Dropdown(
                    choices=_highlight_choices(broken), value=None,
                    label=f"Cells with high CER (>0.30, median)",
                )

        text_md = gr.Markdown()

        gr.Markdown("## Reference audio (JVNV F1)")
        ref_player = gr.Audio(label="reference (the source v_emo was built from)",
                              type="filepath", interactive=False)

        # Per-strategy column row
        strat_players = []
        strat_metric_widgets = []
        with gr.Row():
            for s in strategies:
                with gr.Column():
                    label = _STRATEGY_DESCRIPTIONS.get(s, s)
                    gr.Markdown(f"## {s}\n_{label}_")
                    p = gr.Audio(type="filepath", interactive=False, label=f"{s} output")
                    m = gr.Markdown()
                    strat_players.append(p)
                    strat_metric_widgets.append(m)

        delta_md = gr.Markdown()

        outputs = [ref_player, text_md, *strat_players, *strat_metric_widgets, delta_md]
        lang_dd.change(update, [lang_dd, emo_dd], outputs)
        emo_dd.change(update,  [lang_dd, emo_dd], outputs)
        imp_dd.change(jump,    [imp_dd], [lang_dd, emo_dd])
        reg_dd.change(jump,    [reg_dd], [lang_dd, emo_dd])
        brk_dd.change(jump,    [brk_dd], [lang_dd, emo_dd])

        app.load(update, [lang_dd, emo_dd], outputs)

    return app


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------

DEFAULT_RESULT = Path("outputs/multilang_F1_anger_sad_fear_happy_surprise_disgust/result.json")


def main():
    ap = argparse.ArgumentParser(description="Gradio listening app for ovet sweep")
    ap.add_argument("--result-json", default=DEFAULT_RESULT, type=Path)
    ap.add_argument("--port",        type=int, default=7860)
    ap.add_argument("--server-name", default="0.0.0.0",
                    help="Bind address (0.0.0.0 lets RunPod expose it).")
    ap.add_argument("--share",       action="store_true",
                    help="Create a public gradio.live URL (works through RunPod NAT).")
    args = ap.parse_args()

    if not args.result_json.exists():
        raise FileNotFoundError(args.result_json)

    print(f"[ovet] Loading: {args.result_json}", flush=True)
    app = make_app(args.result_json)

    # Allow gradio to serve any file inside the project tree.
    project_root = Path.cwd().resolve()
    print(f"[ovet] Serving from project root: {project_root}", flush=True)
    print(f"[ovet] Launching on {args.server_name}:{args.port} "
          f"(share={'on' if args.share else 'off'})", flush=True)

    # gradio.launch returns (app, local_url, share_url) when prevent_thread_lock=True
    _, local_url, share_url = app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(project_root)],
        prevent_thread_lock=True,   # so we can print URLs ourselves
    )
    print("=" * 60, flush=True)
    print(f"[ovet] LOCAL URL: {local_url}", flush=True)
    if share_url:
        print(f"[ovet] SHARE URL: {share_url}", flush=True)
    print("=" * 60, flush=True)
    # Persist URLs to a sidecar file the operator can also cat.
    url_file = Path(args.result_json).parent / "gradio_urls.txt"
    url_file.write_text(
        f"local: {local_url}\nshare: {share_url or '(disabled)'}\n",
        encoding="utf-8",
    )
    print(f"[ovet] URLs written to: {url_file}", flush=True)
    # Block forever so the server keeps serving.
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
