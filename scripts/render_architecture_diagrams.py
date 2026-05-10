"""Render architecture diagrams as PNG for the docs.

Two figures:
  docs/figures/plain_omnivoice.png    — baseline (1-pass OmniVoice).
  docs/figures/current_architecture.png — Phase-6 / ε-3 + emotion
                                          conditioning pipeline.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
from matplotlib import font_manager
font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _box(ax, x, y, w, h, text, *, fc="#ffffff", ec="#444", lw=1.6,
         fontsize=10, bold=False, italic=False, fontcolor="#000"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    style = "italic" if italic else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, weight=weight, style=style,
            color=fontcolor,
            zorder=3, linespacing=1.4)


def _arrow(ax, p1, p2, *, color="#444", ls="-", lw=1.5,
           connectionstyle="arc3,rad=0", arrowstyle="-|>",
           mutation_scale=14, label=None, label_offset=(0.0, 0.15)):
    a = FancyArrowPatch(
        p1, p2,
        arrowstyle=arrowstyle, mutation_scale=mutation_scale,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=connectionstyle, zorder=4,
    )
    ax.add_patch(a)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=8.5, color=color,
                ha="center", va="center", style="italic", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="round,pad=0.2"))


def _data_chip(ax, x, y, w, h, text, *, fc="#fffbe6", fontsize=9.5,
               ec="#a98c1f"):
    _box(ax, x, y, w, h, text, fc=fc, ec=ec, lw=1.0, fontsize=fontsize)


# ----------------------------------------------------------------------
# Figure 1: plain OmniVoice
# ----------------------------------------------------------------------

def render_plain():
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    ax.set_title("Plain OmniVoice (baseline)",
                 fontsize=15, weight="bold", pad=14)

    _data_chip(ax, 0.6, 4.6, 3.4, 0.85,
               "target_text\n(例: 明天的天气预报…)", fontsize=10)
    _data_chip(ax, 6.0, 4.6, 3.4, 0.85,
               "F1 reference audio\n(例: jvnv_F1_sad.wav)", fontsize=10)

    _box(ax, 2.2, 2.4, 5.6, 1.5,
         "OmniVoice\n(Qwen3 LLM + Higgs Audio V2 tokenizer)",
         fc="#e6f0ff", ec="#1f4f99", bold=True, fontsize=11.5)

    _box(ax, 3.4, 0.55, 3.2, 1.05, "audio @ 24 kHz",
         fc="#e6f7e6", ec="#2a7a2a", bold=True, fontsize=11.5)

    _arrow(ax, (2.3, 4.6), (3.7, 3.95))
    _arrow(ax, (7.7, 4.6), (6.3, 3.95))
    _arrow(ax, (5.0, 2.4), (5.0, 1.6))

    ax.text(5.0, 0.18,
            "観測: zh で cer_med ≈ 1.000 — JP 訛りで Whisper が漢字を取り違える",
            ha="center", va="center", fontsize=10,
            style="italic", color="#a02a2a")

    fig.savefig(OUT_DIR / "plain_omnivoice.png", dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'plain_omnivoice.png'}")


# ----------------------------------------------------------------------
# Figure 2: current architecture — clean single-pass top→bottom layout
#
# Three vertical lanes:
#   Left lane:    target_text → OmniVoice ×8 (native ref) → content_donors
#   Middle lane:  best content → Approach 1 (F0-stats) → emotion source
#                                         → SeedVC (huge box)
#   Right lane:   F1 emotional ref → emotion paths (F0 stats μ,σ;
#                 emotion2vec embedding; campplus speaker style)
# ----------------------------------------------------------------------

def render_current():
    # 16 wide × 14.5 tall. Extra height for explicit output box.
    fig, ax = plt.subplots(figsize=(15.5, 14.5))
    ax.set_xlim(0, 16); ax.set_ylim(-1.0, 14); ax.axis("off")

    ax.set_title("現行 (Phase-6 ε-3 + emotion conditioning)",
                 fontsize=16, weight="bold", pad=16)

    # ------------------------------------------------------------------
    # ROW 1 (y≈12.6): inputs
    # ------------------------------------------------------------------
    _data_chip(ax, 0.4, 12.7, 4.4, 0.85,
               "target_text\n(例: 明天的天气预报是多云转晴。)",
               fontsize=10.5)
    _data_chip(ax, 5.6, 12.7, 4.4, 0.85,
               "native zh ref (× M)\nFLEURS dev サンプル",
               fontsize=10.5)
    _data_chip(ax, 11.0, 12.7, 4.4, 0.85,
               "F1 emotional ref\njvnv_F1_{emo}.wav",
               fontsize=10.5)

    # ------------------------------------------------------------------
    # ROW 2 (y≈10.7): OmniVoice generation (×N candidates)
    # ------------------------------------------------------------------
    _box(ax, 0.4, 10.7, 4.4, 1.25,
         "OmniVoice ×8\n(native ref + target_text)",
         fc="#e6f0ff", ec="#1f4f99", bold=True, fontsize=11)

    # input arrows to OmniVoice ×8
    _arrow(ax, (2.6, 12.7), (2.6, 11.95))                 # target_text → OV×8
    _arrow(ax, (5.8, 12.7), (4.6, 11.95),                 # native ref → OV×8
           connectionstyle="arc3,rad=-0.10")

    # ------------------------------------------------------------------
    # ROW 3 (y≈9.0): content_donors stream
    # ------------------------------------------------------------------
    _data_chip(ax, 0.4, 8.95, 4.4, 1.0,
               "content_donors (×8)\n= ネイティブ話者の声色 + 正しい zh\n(中性 prosody)",
               fontsize=9.5, fc="#f3e8ff", ec="#6a4a9a")
    _arrow(ax, (2.6, 10.7), (2.6, 9.95))

    # ------------------------------------------------------------------
    # ROW 4 (y≈7.4): ASR-rank → best content
    # ------------------------------------------------------------------
    _box(ax, 0.6, 7.40, 4.0, 0.95,
         "ASR-rank (Whisper CER)\n→ best_content_donor",
         fc="#fff4dc", ec="#c08b00", fontsize=10)
    _arrow(ax, (2.6, 8.95), (2.6, 8.35))

    # ------------------------------------------------------------------
    # ROW 5 (y≈5.4): Approach 1 — F0-stats transfer
    # ------------------------------------------------------------------
    _box(ax, 5.0, 5.40, 7.4, 2.0,
         "Approach 1 — F0-stats transfer (WORLD)\n\n"
         "(f0, sp, ap) = WORLD.decompose(content_donor)\n"
         "f0_new[voiced] = (f0 − μ_c)/σ_c · σ_emo + μ_emo\n"
         "audio = WORLD.synthesize(f0_new, sp, ap)",
         fc="#e0f5e0", ec="#2a7a2a", fontsize=10.2)

    _arrow(ax, (4.6, 7.85), (5.5, 7.40),                 # best content → A1 (left side in)
           connectionstyle="arc3,rad=-0.10",
           color="#1d5d1d", lw=1.7)
    _arrow(ax, (13.2, 9.30), (12.4, 7.40),               # split node → A1 (μ, σ)
           connectionstyle="arc3,rad=-0.10",
           color="#aa6622", ls="--",
           label="F0 mean, std",
           label_offset=(0.4, 0.2))

    # ------------------------------------------------------------------
    # ROW 6 (y≈4.0): emotion-conditioned source (output of A1)
    # ------------------------------------------------------------------
    _box(ax, 5.6, 4.05, 6.2, 1.05,
         "emotion-conditioned source\n(content phonetics + F1 emotional pitch)",
         fc="#cde8cd", ec="#1d5d1d", bold=True, fontsize=10.5)
    _arrow(ax, (8.7, 5.40), (8.7, 5.10), color="#1d5d1d")

    # ------------------------------------------------------------------
    # SeedVC big container box (rows 7-9, y 0.55..3.85)
    # ------------------------------------------------------------------
    _box(ax, 1.0, 0.55, 14.0, 3.30, "",
         fc="#eef3ff", ec="#1f4f99", lw=2.2)
    ax.text(8.0, 3.55, "SeedVC (DiT + flow-matching)",
            ha="center", va="center", fontsize=12.5, weight="bold",
            color="#1f4f99", zorder=5)

    # Inside SeedVC: 3 sub-blocks for inputs, then cfm.inference
    _box(ax, 1.4, 2.2, 4.0, 0.95,
         "Whisper content encoder\n(source → S_alt content tokens)",
         fc="#ffffff", ec="#88a", fontsize=9.5)
    _box(ax, 6.0, 2.2, 4.0, 0.95,
         "campplus(F1 ref)\n→ style2_speaker (192-d)",
         fc="#ffffff", ec="#88a", fontsize=9.5)
    _box(ax, 10.6, 2.2, 4.0, 0.95,
         "emotion2vec(F1 emo) → 768-d\nrandom orth proj → style2_emo (192-d)",
         fc="#fff2d9", ec="#c08b00", fontsize=9.5)

    # blending equation under sub-blocks
    ax.text(8.0, 1.65,
            r"style2 = α · style2_speaker + β · style2_emo   ←  Approach 2",
            ha="center", va="center", fontsize=10.2,
            color="#a05a00", weight="bold", zorder=5,
            bbox=dict(facecolor="#fff7e0", edgecolor="#c08b00",
                      boxstyle="round,pad=0.4", linewidth=1.0))

    _box(ax, 3.0, 0.70, 10.0, 0.75,
         "cfm.inference(content, mel2_F1, style2_blended,  "
         "diffusion_steps=100, cfg=0.7) → BigVGAN vocoder",
         fc="#ffffff", ec="#88a", fontsize=9.7)

    # F1 emotional ref splits into a "right rail" that fans out to:
    #   (a) Approach-1 F0 stats  (already drawn above with --)
    #   (b) campplus speaker style  (--; speaker path)
    #   (c) emotion2vec embedding   (--, emotion path)
    # Drop a small "F1 ref signal" node mid-right to clean up routing.
    _data_chip(ax, 12.9, 9.3, 2.7, 0.85,
               "F1 ref signal\n(spectral + prosody)",
               fontsize=9.5, fc="#ffe9d3", ec="#aa6622")
    _arrow(ax, (13.2, 12.7), (14.25, 10.15),             # F1 ref → split node
           color="#aa6622", ls="--", lw=1.5)
    # split → A1 (curving left into F0-stats label area)
    # — already covered by the F0 stats arrow above; keep that intact.
    # split → campplus (to middle SeedVC sub-block)
    _arrow(ax, (14.0, 9.30), (8.0, 3.15),
           connectionstyle="arc3,rad=0.30",
           color="#666", lw=1.2,
           label="speaker style (campplus)", label_offset=(1.0, -0.5))
    # split → emotion2vec (right SeedVC sub-block)
    _arrow(ax, (14.25, 9.30), (12.6, 3.15),
           connectionstyle="arc3,rad=0.10",
           color="#aa6622", ls="--", lw=1.5,
           label="emotion2vec emb.", label_offset=(0.9, 0.4))

    # source → Whisper content encoder (left in the SeedVC box)
    _arrow(ax, (4.7, 4.05), (3.4, 3.15),
           connectionstyle="arc3,rad=0.05",
           color="#1d5d1d", lw=1.7)

    # arrows from sub-blocks into cfm
    _arrow(ax, (3.4, 2.2), (5.5, 1.45), color="#666",
           lw=1.1, mutation_scale=11)
    _arrow(ax, (8.0, 2.2), (8.0, 1.45), color="#666",
           lw=1.1, mutation_scale=11)
    _arrow(ax, (12.6, 2.2), (10.5, 1.45), color="#666",
           lw=1.1, mutation_scale=11)

    # ------------------------------------------------------------------
    # Output (below SeedVC) — explicit box
    # ------------------------------------------------------------------
    _box(ax, 4.5, -0.85, 7.0, 1.10,
         "audio @ 22.05 kHz → resample → 24 kHz\n"
         "(best-of-8: lowest ASR-CER candidate)",
         fc="#e6f7e6", ec="#2a7a2a", bold=True, fontsize=10.5)
    _arrow(ax, (8.0, 0.55), (8.0, 0.25), color="#2a7a2a", lw=1.7)

    # ------------------------------------------------------------------
    # Legend (bottom-left, outside main flow)
    # ------------------------------------------------------------------
    legend_elems = [
        Line2D([0], [0], color="#1d5d1d", lw=2, label="content path"),
        Line2D([0], [0], color="#aa6622", lw=2, ls="--",
               label="emotion path (F0 stats / emotion2vec)"),
        Line2D([0], [0], color="#666", lw=1.5,
               label="speaker path"),
    ]
    ax.legend(handles=legend_elems, loc="upper left",
              bbox_to_anchor=(0.0, 0.04), frameon=False, fontsize=10)

    fig.savefig(OUT_DIR / "current_architecture.png", dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'current_architecture.png'}")


if __name__ == "__main__":
    render_plain()
    render_current()
