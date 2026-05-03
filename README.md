# OmniVoice Cross-Lingual Emotion Preservation (ovet)

Training-free pipeline that preserves emotion in OmniVoice voice cloning when
the target text language differs from the reference audio language.

See `docs/omnivoice_emotion_transfer_requirements_design.md` for the full
requirements/design document.

## Phase 1 minimal usage

```bash
python -m ovet.cli.run_emotion_clone \
    --text "Thank you so much for coming today." \
    --language English \
    --ref-audio baseline/jvnv_samples/jvnv_F1_anger.wav \
    --emotion-hint anger \
    --output-dir outputs/test_anger_en
```
