# OmniVoice 感情転写強化パイプライン 要件定義書・設計書

## 1. 概要

本ドキュメントは、OmniVoice をベースに、追加学習データを用いずに感情転写を強化するための推論時制御パイプラインの要件定義および設計をまとめたものである。

本プロジェクトの主目的は、感情参照音声から抽出した感情・プロソディ特徴を OmniVoice の生成音声に反映しつつ、target speaker prompt の話者性崩壊を抑えることである。

本設計では、OmniVoice 本体の大規模再学習は行わない。代わりに、OSS の感情認識モデル、speaker encoder、DSP特徴量抽出器、hidden activation steering、候補生成・評価・選別ロジックを組み合わせる。

---

## 2. 背景

OmniVoice は instruction tokens と acoustic prompt tokens を用いて音声生成を行うため、参照音声や自然言語指示による表現制御と相性が良い。

一方で、感情参照音声をそのまま利用すると、以下の問題が起きやすい。

- 感情だけでなく、参照話者の声質まで転写される
- target speaker の話者性が崩れる
- 感情を強めるほど音質や内容忠実性が低下する
- 参照音声の録音条件やノイズが出力に混入する
- 感情ラベルだけでは、抑揚・間・息っぽさ・声の張りなどが十分に反映されない

そのため、本設計では「感情を強める処理」と「話者性を守る処理」を分離して設計する。

---

## 3. 目的

### 3.1 主目的

追加学習データなしで、OmniVoice の出力に感情参照音声の感情傾向を反映する。

### 3.2 副目的

- target speaker prompt の話者性を維持する
- 感情強度をスライダー的に制御できるようにする
- 感情参照音声由来の話者性漏れを抑える
- 生成候補を自動評価し、最良候補を選べるようにする
- 将来的な LoRA / Adapter / 再学習に拡張可能な構成にする

---

## 4. 非目的

本プロジェクトでは、初期段階では以下を行わない。

- OmniVoice のフルファインチューニング
- 感情ペアデータセットの新規収集
- GRL / contrastive loss / MI minimization などを用いた本格的な disentanglement 学習
- 完全な frame-level emotional style transfer
- 参照音声の泣き・震え・息遣いの完全再現
- 商用ライセンス確認を含む法務判断の自動化

---

## 5. 用語定義

| 用語 | 意味 |
|---|---|
| target speaker | 最終出力で維持したい話者 |
| emotion reference | 感情を抽出するための参照音声 |
| speaker leakage | emotion reference 側の話者性が生成音声に漏れる現象 |
| activation steering | モデル内部 hidden state に方向ベクトルを加算して出力傾向を変える手法 |
| emotion steering vector | neutral と emotional の hidden state 差分から作る感情方向ベクトル |
| speaker direction vector | 話者性変化を表す方向ベクトル |
| projection removal | 感情方向ベクトルから話者方向成分を射影除去する処理 |
| inference-time disentanglement | 学習ではなく推論時の制約・探索・幾何処理によって疑似的に分離する手法 |

---

## 6. 全体アーキテクチャ

```text
[Text]
  │
  ├──────────────────────────────────────────────┐
  │                                              │
[Target Speaker Prompt Audio]                    │
  │                                              │
  ├─ Speaker Encoder ───── target speaker emb    │
  │                                              │
[Emotion Reference Audio]                        │
  │                                              │
  ├─ Emotion Encoder ───── emotion emb/label     │
  ├─ DSP Extractor ─────── prosody features      │
  └─ Speaker Encoder ───── emotion-ref speaker emb
                                                 │
[Prompt Composer]
  │
  ├─ instruction text
  ├─ prosody description
  └─ negative transfer instruction
                                                 │
[OmniVoice Generator]
  │
  ├─ optional hidden hook
  ├─ emotion steering
  ├─ speaker projection removal
  └─ multi-alpha candidate generation
                                                 │
[Candidate Evaluator]
  │
  ├─ emotion similarity
  ├─ target speaker similarity
  ├─ emotion-ref speaker similarity penalty
  ├─ content accuracy / ASR check
  └─ audio quality score
                                                 │
[Best Candidate Selector]
  │
[Final WAV]
```

---

## 7. 実装方針の対応関係

本設計は、過去に検討した案1〜5、および学習なしの speaker-disentanglement 案F〜Jを統合する。

### 7.1 感情転写強化案

| 案 | 内容 | 本設計での扱い |
|---|---|---|
| 案1 | Emotion2Vec → instruction 自動生成 | 採用 |
| 案2 | Emotion2Vec + DSP特徴 → prosody 指示 | 採用 |
| 案3 | Activation Steering 自動化 | 採用。ただし段階的実装 |
| 案4 | OmniVoice → OpenVoice/VC 後段変換 | オプション |
| 案5 | 感情TTSモデルを教師にした探索 | オプション評価用 |

### 7.2 学習なし disentanglement 案

| 案 | 内容 | 本設計での扱い |
|---|---|---|
| 案F | Speaker Similarity Guard | 採用 |
| 案G | 感情方向から話者方向成分を射影除去 | 採用 |
| 案H | 複数参照平均による共通感情成分抽出 | オプション |
| 案I | ブラックボックス探索 | 採用 |
| 案J | promptによる明示分離 | 採用 |

---

## 8. 機能要件

## 8.1 入力

システムは以下を入力として受け取る。

| 入力 | 必須 | 説明 |
|---|---:|---|
| text | 必須 | 読み上げ対象テキスト |
| target_speaker_audio | 必須 | 維持したい話者性の参照音声 |
| emotion_reference_audio | 必須 | 転写したい感情を含む参照音声 |
| neutral_reference_audio | 任意 | activation steering 用の中立音声 |
| emotion_label_hint | 任意 | happy/sad/angry 等のユーザー指定 |
| emotion_intensity | 任意 | 0.0〜1.0 の感情強度 |
| output_dir | 任意 | 出力保存先 |
| num_candidates | 任意 | 候補生成数 |
| alpha_grid | 任意 | steering強度の探索値 |

---

## 8.2 出力

システムは以下を出力する。

| 出力 | 説明 |
|---|---|
| final.wav | 最良候補の音声 |
| candidates/*.wav | 生成候補群 |
| result.json | 各候補の評価スコア |
| prompt.txt | 実際にOmniVoiceへ渡した instruction |
| debug/*.npy | 必要に応じた hidden state / embedding |
| report.md | 実験結果の簡易レポート |

---

## 8.3 感情抽出

### 要件

- emotion_reference_audio から感情ラベル、感情強度、emotion embedding を抽出する
- emotion_label_hint がある場合は、それを優先または補助情報として扱う
- emotion embedding は後続の類似度評価にも使用する

### 推奨OSS候補

- emotion2vec
- Emotion2Vec-S
- WavLM / HuBERT 系 SER モデル
- SpeechBrain 系 emotion recognition モデル

### インターフェース

```python
class EmotionAnalyzer:
    def analyze(self, wav_path: str) -> EmotionAnalysis:
        ...
```

```python
@dataclass
class EmotionAnalysis:
    label: str
    confidence: float
    intensity: float
    embedding: np.ndarray
    logits: dict[str, float]
```

---

## 8.4 DSP特徴量抽出

### 要件

emotion_reference_audio から以下の音響特徴を抽出する。

| 特徴量 | 用途 |
|---|---|
| F0 mean | 声の高さ傾向 |
| F0 std / range | 抑揚の大きさ |
| energy mean | 声の強さ |
| energy std | 強弱変化 |
| speech rate | 話速 |
| pause ratio | 間の多さ |
| pause duration mean | 間の長さ |
| voiced ratio | 有声音比率 |
| breathiness proxy | 息っぽさ |
| creaky/pressed proxy | 声質変化の近似 |

### 推奨ライブラリ

- pyworld
- librosa
- openSMILE
- silero-vad
- webrtcvad
- reaper

### インターフェース

```python
class ProsodyAnalyzer:
    def analyze(self, wav_path: str) -> ProsodyFeatures:
        ...
```

```python
@dataclass
class ProsodyFeatures:
    f0_mean: float
    f0_std: float
    f0_range: float
    energy_mean: float
    energy_std: float
    speech_rate: float
    pause_ratio: float
    pause_duration_mean: float
    voiced_ratio: float
    breathiness_proxy: float | None = None
    voice_quality_proxy: dict[str, float] | None = None
```

---

## 8.5 Instruction生成

### 要件

EmotionAnalysis と ProsodyFeatures から、OmniVoice に渡す自然言語 instruction を生成する。

### 基本方針

悪い指示：

```text
この参照音声のように読んでください。
```

これは話者性・録音条件・ノイズまで混ざって転写される可能性が高い。

良い指示：

```text
target speaker prompt の声質、年齢感、性別感、声の太さは維持してください。
emotion reference audio からは感情、抑揚、話速、ポーズ、息遣いのみを反映してください。
emotion reference audio の話者性、声質、録音環境、ノイズ、マイク距離感は模倣しないでください。
```

### 指示テンプレート

```text
[Speaker]
Maintain the speaker identity, timbre, age impression, gender impression, and vocal thickness of the target speaker prompt.

[Emotion]
Express {emotion_label} with intensity {emotion_intensity}.
Reflect only the emotional prosody, intonation, pause pattern, speaking rate, and voice energy from the emotion reference.

[Prosody]
Use {pitch_description}, {energy_description}, {speed_description}, and {pause_description}.

[Do Not Transfer]
Do not imitate the speaker identity, timbre, recording condition, microphone distance, background noise, accent, or identity-specific habits of the emotion reference speaker.

[Text]
{text}
```

### インターフェース

```python
class PromptComposer:
    def compose(
        self,
        text: str,
        emotion: EmotionAnalysis,
        prosody: ProsodyFeatures,
        options: PromptOptions,
    ) -> str:
        ...
```

---

## 8.6 OmniVoice生成

### 要件

- text / instruction / target_speaker_audio を入力として音声生成する
- activation steering なしでも動作する
- activation steering を有効化できる
- 複数 alpha 候補を生成できる

### インターフェース

```python
class OmniVoiceWrapper:
    def generate(
        self,
        text: str,
        instruction: str,
        target_speaker_audio: str,
        steering: SteeringConfig | None = None,
        output_path: str | None = None,
    ) -> str:
        ...
```

```python
@dataclass
class SteeringConfig:
    enabled: bool
    alpha: float
    layer_ids: list[int]
    emotion_vector: dict[int, np.ndarray] | None = None
    speaker_vector: dict[int, np.ndarray] | None = None
    projection_removal: bool = True
```

---

## 8.7 Activation Steering

### 要件

OmniVoice の中間 hidden state に対して、感情方向ベクトルを加算する。

```text
h_l' = h_l + alpha * v_emo_clean_l
```

ここで、

```text
v_emo_clean_l = v_emo_l - projection(v_emo_l, v_spk_l)
```

とする。

### 感情方向ベクトル

neutral_reference_audio がある場合：

```text
v_emo_l = hidden_l(emotional_ref) - hidden_l(neutral_ref)
```

neutral_reference_audio がない場合：

```text
v_emo_l = hidden_l(emotional_ref) - hidden_l(target_speaker_prompt)
```

または、

```text
v_emo_l = hidden_l(emotional_ref) - hidden_l(generic_neutral_prompt)
```

### 話者方向ベクトル

候補1：

```text
v_spk_l = hidden_l(emotion_reference_audio) - hidden_l(target_speaker_audio)
```

候補2：

```text
v_spk_l = hidden_l(target_speaker_audio) - hidden_l(generic_neutral_prompt)
```

候補3：

```text
複数話者の中立音声から PCA / 平均差分で speaker direction を近似
```

### 射影除去

```python
def remove_projection(v_emo: np.ndarray, v_spk: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    coef = np.sum(v_emo * v_spk) / (np.sum(v_spk * v_spk) + eps)
    return v_emo - coef * v_spk
```

### 複数層対応

```python
clean_vectors = {}

for layer_id in layer_ids:
    v_emo = emotion_vectors[layer_id]
    v_spk = speaker_vectors[layer_id]

    if projection_removal:
        clean_vectors[layer_id] = remove_projection(v_emo, v_spk)
    else:
        clean_vectors[layer_id] = v_emo
```

### 実装上の注意

- hidden shape は `[batch, time, dim]` または `[time, dim]` の可能性がある
- 差分を取る前に時間方向平均を取るか、token alignment を行う
- **whole-utterance の単純時間平均は最後の手段**。優先順位は以下:
  1. 同一テキスト読み上げ対 (neutral_ref と emotional_ref が同一発話内容) で frame-level 差分 → 平均
  2. VAD で有声フレームのみに限定した平均
  3. whole-utterance 単純平均 (sanity check 用途)
- layerごとに norm が異なるため、L2 normalize または RMS normalize を行う
- alpha は小さめから探索する
- emotion vector / speaker vector の生成パスは差分定義 (どの音声を引いたか) をメタデータとして保存し、再現性を確保する

---

## 8.8 Candidate生成

### 要件

以下の組み合わせで複数候補を生成する。

- instruction variation
- alpha
- layer set
- projection removal on/off
- prosody prompt strength

### alpha grid 例

```python
alpha_grid = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
```

### layer set 例

```python
layer_sets = [
    [],
    [6],
    [10],
    [14],
    [10, 14],
    [8, 12, 16],
]
```

### インターフェース

```python
class CandidateGenerator:
    def generate_candidates(
        self,
        request: GenerationRequest,
        alpha_grid: list[float],
        layer_sets: list[list[int]],
    ) -> list[Candidate]:
        ...
```

---

## 8.9 Candidate評価

### 要件

各候補について以下のスコアを算出する。

| スコア | 目的 |
|---|---|
| emotion_similarity | 感情参照音声にどれだけ近いか |
| emotion_label_score | 目標感情として分類されるか |
| target_speaker_similarity | target speaker に似ているか |
| emotion_ref_speaker_similarity | emotion reference speaker に似すぎていないか |
| prosody_similarity | F0/energy/pause が近いか |
| content_accuracy | 読み上げ内容が崩れていないか |
| audio_quality | 音質が崩れていないか |

### スコア式

```text
score =
  w_emo * emotion_similarity
+ w_emo_label * emotion_label_score
+ w_target_spk * target_speaker_similarity
- w_ref_spk * emotion_ref_speaker_similarity
+ w_prosody * prosody_similarity
- w_content * content_error
+ w_quality * audio_quality
```

初期値：

```python
weights = {
    "emotion_similarity": 1.0,
    "emotion_label_score": 0.8,
    "target_speaker_similarity": 1.0,
    "emotion_ref_speaker_similarity": 0.7,
    "prosody_similarity": 0.4,
    "content_error": 0.8,
    "audio_quality": 0.5,
}
```

### hard constraint

候補選択時には以下を満たす候補のみを採用対象とする。

```text
target_speaker_similarity >= speaker_threshold
content_error <= content_error_threshold
audio_quality >= quality_threshold
```

初期値：

```python
speaker_threshold = 0.80
content_error_threshold = 0.20
quality_threshold = 0.50
```

### 評価器信頼性の前提

これらの自動指標と閾値は、**評価器が日本語音声に対して妥当に動くこと**が前提。Phase 0 の spot check (10〜20 サンプルで主観評価との相関を見る) を実施した上で、初期値を補正する。具体的には:

- SER スコアと主観の感情ラベル一致率が低い場合は `emotion_label_score` の重みを下げる
- speaker encoder が同一話者の感情変化で大きく振れる場合は `speaker_threshold` を緩める、または感情強度ごとに閾値を分ける
- Whisper CER が neutral 発話でも高めに出る場合は `content_error_threshold` を緩める

評価器の信頼性検証なしに Phase 5 の grid search を回すと、artifact を最適化することになるため、Phase 5 着手の前提条件として明示する。

---

## 8.10 Best Candidate Selector

### 要件

- hard constraint を満たす候補をフィルタリングする
- 残った候補の中で総合スコア最大のものを選ぶ
- 候補が1つも残らない場合は、alpha が低い候補から最も安全なものを選ぶ
- result.json に全候補のスコアを保存する

### 疑似コード

```python
def select_best(candidates: list[Candidate], thresholds: Thresholds) -> Candidate:
    valid = []

    for c in candidates:
        if c.target_speaker_similarity < thresholds.speaker:
            continue
        if c.content_error > thresholds.content_error:
            continue
        if c.audio_quality < thresholds.quality:
            continue
        valid.append(c)

    if valid:
        return max(valid, key=lambda c: c.total_score)

    # fallback: 話者性と内容崩壊を優先して選ぶ
    return max(
        candidates,
        key=lambda c: (
            c.target_speaker_similarity,
            -c.content_error,
            c.audio_quality,
            c.emotion_similarity,
        )
    )
```

---

## 9. CLI設計

### 9.1 最小実行

```bash
python run_emotion_transfer.py \
  --text "今日は来てくれて、本当にありがとう。" \
  --target-speaker target.wav \
  --emotion-ref sad_ref.wav \
  --output-dir outputs/test01
```

### 9.2 steering有効

```bash
python run_emotion_transfer.py \
  --text "今日は来てくれて、本当にありがとう。" \
  --target-speaker target.wav \
  --emotion-ref sad_ref.wav \
  --neutral-ref neutral.wav \
  --enable-steering \
  --alpha-grid 0.0,0.2,0.4,0.6,0.8 \
  --layers 8,12,16 \
  --projection-removal \
  --output-dir outputs/test02
```

### 9.3 prompt制御のみ

```bash
python run_emotion_transfer.py \
  --text "今日は来てくれて、本当にありがとう。" \
  --target-speaker target.wav \
  --emotion-ref happy_ref.wav \
  --disable-steering \
  --prompt-only \
  --output-dir outputs/prompt_only
```

### 9.4 評価のみ

```bash
python evaluate_candidates.py \
  --candidate-dir outputs/test02/candidates \
  --target-speaker target.wav \
  --emotion-ref sad_ref.wav \
  --text "今日は来てくれて、本当にありがとう。"
```

---

## 10. ディレクトリ構成案

```text
omnivoice-emotion-transfer/
  README.md
  requirements.txt
  pyproject.toml

  configs/
    default.yaml
    scoring.yaml
    prompt_templates.yaml

  scripts/
    run_emotion_transfer.py
    evaluate_candidates.py
    extract_hidden.py
    debug_prompt_only.py

  src/
    ovet/
      __init__.py

      analyzers/
        emotion_analyzer.py
        prosody_analyzer.py
        speaker_analyzer.py
        asr_analyzer.py
        quality_analyzer.py

      prompts/
        prompt_composer.py
        templates.py

      omnivoice/
        wrapper.py
        hidden_hooks.py
        steering.py

      generation/
        candidate_generator.py
        request.py
        result.py

      evaluation/
        evaluator.py
        scoring.py
        selector.py

      utils/
        audio.py
        io.py
        normalization.py
        logging.py

  outputs/
    .gitkeep

  tests/
    test_projection.py
    test_scoring.py
    test_prompt_composer.py
```

---

## 11. データクラス設計

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class GenerationRequest:
    text: str
    target_speaker_audio: Path
    emotion_reference_audio: Path
    neutral_reference_audio: Path | None
    output_dir: Path
    emotion_intensity: float | None = None
    emotion_label_hint: str | None = None

@dataclass
class Candidate:
    wav_path: Path
    instruction: str
    alpha: float
    layer_ids: list[int]
    projection_removal: bool
    scores: dict[str, float]
    total_score: float

@dataclass
class Thresholds:
    speaker: float = 0.80
    content_error: float = 0.20
    quality: float = 0.50
```

---

## 12. 実装ステップ

各フェーズには gate (前段の完了条件) があり、gate を満たさない場合は次フェーズに進まない方針とする。

```text
Phase 0 (環境 + アーキテクチャ調査 + 評価器 spot check)
  ↓ gate: hook 候補 layer 特定済み + 評価器信頼性レンジ把握済み
Phase 1 (prompt-only) ← MVP の最小到達点
  ↓ gate: prompt-only の天井観測 + 不足要素の言語化
Phase 2 (候補評価)
  ↓ gate: 自動スコアと主観の相関確認 + 閾値補正完了
  ↓ ここで要件達成なら Phase 3 以降は不要
Phase 3 (Activation Steering)
  ↓ gate: alpha=0 で通常生成と一致 + alpha 増加で感情スコアが変化
Phase 4 (Projection Removal)
  ├─ 4a: Geometric Validation Spike (線形分離可能性検証)
  │   ↓ gate: 線形分離が機能する見通し
  └─ 4b: 本実装
  ↓ gate: speaker similarity の低下が抑制される
Phase 5 (Black-box Optimization)
  ↓ gate: 評価器が validated で artifact 最適化のリスクが低い
Phase 6 (Optional VC / Teacher 比較) -- オプション
```


## Phase 0: 環境構築・前提検証

- [ ] OmniVoice をローカルで通常実行できるようにする
- [ ] target speaker prompt で通常TTS生成できることを確認
- [ ] emotion2vec または代替SERモデルを導入する
- [ ] speaker encoder を導入する
- [ ] pyworld / librosa / VAD を導入する
- [ ] 出力wavの保存・再生確認を行う
- [ ] **OmniVoice アーキテクチャ調査**
  - LLM 部 / audio decoder 部の境界を特定する
  - instruction tokens / acoustic prompt tokens の入り口と流れを把握する
  - hidden state を取得・上書き可能な layer 候補を列挙する (Phase 3 の hook 設計に直結)
- [ ] **評価器の日本語信頼性 spot check (10〜20 サンプル)**
  - SER モデル (emotion2vec 等) のラベル/embedding と人手評価の相関を確認
  - speaker encoder が同一話者の感情変化に対してどれだけ embedding が動くかを測る
  - Whisper の CER が感情強度でどれだけ悪化するかを把握
  - 結果に応じて Phase 2 の閾値・重みの初期値を決める

完了条件：

- 通常 TTS 生成が再現可能
- hook 可能な layer 候補が文書化されている
- 評価器ごとの信頼性レンジが把握されており、後続の閾値設計の根拠になっている

---

## Phase 1: Prompt制御のみ (Phase 3-4 必要性の判定 gate)

目的：OmniVoice本体を改造せず、感情参照音声からinstructionを生成して感情制御を試す。
本フェーズは同時に **prompt-only でどこまで感情転写できるかの天井を観測し、Phase 3-4 (steering) に進む必要があるかを判定する gate** として位置付ける。

- [ ] EmotionAnalyzer を実装
- [ ] ProsodyAnalyzer を実装
- [ ] PromptComposer を実装
- [ ] OmniVoiceWrapper を実装
- [ ] prompt-only 生成スクリプトを実装
- [ ] 出力音声を主観評価
- [ ] emotion2vecで出力感情を自動評価
- [ ] **天井観測**: 感情強度・prosody 反映度・話者性保持の 3軸で prompt-only の到達点を定量化
- [ ] **不足要素の特定**: 何が足りないか (感情強度? 抑揚? 息遣い?) を言語化し、Phase 3-4 で何を補うべきかを明文化

完了条件：

- target speaker prompt の声質を大きく崩さず、感情ラベルに沿った出力が得られる
- prompt.txt と result.json が保存される
- prompt-only の限界点と、Phase 3-4 着手の要否判断が文書化されている

---

## Phase 2: Candidate生成・評価

目的：複数候補を生成し、感情と話者性のバランスが良いものを自動選択する。

- [ ] SpeakerAnalyzer を実装
- [ ] CandidateEvaluator を実装
- [ ] BestCandidateSelector を実装
- [ ] alphaなしのprompt variation生成を実装
- [ ] result.json 保存を実装
- [ ] report.md 保存を実装

完了条件：

- 複数候補から最良候補を選択できる
- target speaker similarity が閾値未満の候補を棄却できる

---

## Phase 3: Activation Steering

目的：OmniVoice内部hiddenに感情方向を足して、感情の出方を強める。

**前提**: Phase 0 のアーキテクチャ調査で hook 可能な layer 候補が特定されていること。Phase 1 の天井観測で「prompt-only では届かない不足要素」が言語化されていること。これらが揃わないうちは本フェーズに着手しない。

- [ ] OmniVoice の Transformer block に hook を入れる
- [ ] 指定 layer の hidden state を取得する
- [ ] emotional / neutral の hidden 差分を計算する
  - **時間平均は whole-utterance 単純平均を避け、有声フレーム限定平均を採用する**
  - 可能なら同一テキスト読み上げ対 (neutral_ref と emotional_ref を同一テキスト) で差分を取る
  - 単純平均しか取れない場合でも、layer ごとに L2 / RMS normalize する
- [ ] steering vector を保存・読み込みできるようにする
- [ ] 推論時に hidden state へ alpha * vector を加算する
- [ ] alpha_grid による候補生成を実装する

完了条件：

- alphaを上げると感情スコアが変化する
- alpha=0.0 のとき通常生成と一致または近似する

---

## Phase 4: Projection Removal

目的：感情方向ベクトルから話者方向成分を除去し、話者性崩壊を抑える。

### Phase 4a: Geometric Validation Spike (本実装の前)

`v_emo - proj(v_emo, v_spk)` は「感情と話者性が hidden 空間で線形に分離可能」という強い仮定の上に乗っている。本実装に進む前に、この仮定を最低限のコストで検証する。

- [ ] 多話者 × 多感情 (最低 3話者 × 3感情) で hidden を抽出する
- [ ] PCA / linear probe で「話者軸」と「感情軸」の直交性を見る
- [ ] [:404-421](#) の speaker direction 3候補のうちどれが最も話者情報を捕捉するかを linear probe で比較し、採用候補を1つに絞る
- [ ] projection removal 適用前後の hidden を speaker classifier に通し、話者情報が落ちることを確認する
- [ ] 結果が芳しくない場合は Phase 4 本実装を見送り、Phase 1-3 で着地させる判断材料とする

### Phase 4b: 本実装

- [ ] speaker direction vector を Phase 4a で選定した定義で実装する
- [ ] remove_projection を実装する
- [ ] layerごとに projection removal を適用する
- [ ] projection removal on/off の比較評価を行う
- [ ] speaker similarity の低下量を比較する

完了条件：

- Phase 4a の geometric validation で線形分離の有効性が確認されている
- projection removal により target speaker similarity の低下が抑えられる
- emotion similarity が極端に低下しない

---

## Phase 5: Black-box Optimization

目的：prompt・alpha・layer・projection設定を探索し、感情と話者性のPareto最適候補を選ぶ。

- [ ] grid search を実装
- [ ] optionalでOptuna等の探索に対応
- [ ] score weights を設定ファイル化
- [ ] Pareto front を report.md に出力
- [ ] 各候補のメタデータを保存する

完了条件：

- speaker thresholdを満たす範囲でemotion similarity最大の候補を選べる
- 探索結果が再現可能である

---

## Phase 6: Optional VC / Teacher比較

目的：必要に応じて、OpenVoice / StyleTTS2 / XTTS等を比較用または後段変換として接続する。

- [ ] teacher emotional speech を生成または入力できるようにする
- [ ] teacher と OmniVoice候補の emotion embedding 距離を測る
- [ ] 後段VCあり/なしを比較する
- [ ] 音質劣化と話者性崩壊を評価する

完了条件：

- 後段VCの有効性を定量・主観の両方で比較できる

---

## 13. 評価指標

### 13.1 自動評価

| 指標 | ツール候補 | 目的 |
|---|---|---|
| emotion label accuracy | emotion2vec / SER | 目標感情として認識されるか |
| emotion embedding cosine | emotion2vec | 感情参照に近いか |
| speaker similarity | ECAPA-TDNN / SpeechBrain / pyannote | target speakerを維持しているか |
| ref speaker leakage | 同上 | emotion reference speakerに似すぎていないか |
| F0 distance | pyworld | 抑揚傾向の近さ |
| energy distance | librosa | 声の強さの近さ |
| pause similarity | VAD | 間の近さ |
| CER/WER | Whisper等 | 内容が崩れていないか |
| UTMOS / DNSMOS | 音質評価モデル | 音質劣化の確認 |

### 13.2 主観評価

最低限、以下の5段階評価を行う。

| 項目 | 評価 |
|---|---|
| 感情が伝わるか | 1〜5 |
| target speakerらしさ | 1〜5 |
| emotion reference speakerに似てしまっていないか | 1〜5、低いほど良い |
| 内容の聞き取りやすさ | 1〜5 |
| 音質 | 1〜5 |
| 総合自然性 | 1〜5 |

---

## 14. 設定ファイル例

```yaml
# configs/default.yaml

generation:
  # MVP では反復速度を確保するため候補数を絞る
  # Phase 5 (black-box optimization) で広げる
  num_candidates: 8
  alpha_grid: [0.0, 0.4, 0.8]
  layer_sets:
    - []
    - [12]
    - [8, 12, 16]
  projection_removal: true

thresholds:
  speaker: 0.80
  content_error: 0.20
  quality: 0.50

scoring:
  emotion_similarity: 1.0
  emotion_label_score: 0.8
  target_speaker_similarity: 1.0
  emotion_ref_speaker_similarity: 0.7
  prosody_similarity: 0.4
  content_error: 0.8
  audio_quality: 0.5

prompt:
  language: "ja"
  negative_transfer_instruction: true
  prosody_detail_level: "high"
```

---

## 15. 主要アルゴリズム

## 15.1 Training-free Disentangled Emotion Transfer

```python
def run_pipeline(request: GenerationRequest):
    emotion = emotion_analyzer.analyze(request.emotion_reference_audio)
    prosody = prosody_analyzer.analyze(request.emotion_reference_audio)

    instruction = prompt_composer.compose(
        text=request.text,
        emotion=emotion,
        prosody=prosody,
        options=PromptOptions()
    )

    steering_vectors = None

    if config.generation.enable_steering:
        emotion_vectors = hidden_extractor.compute_emotion_vectors(
            emotional_audio=request.emotion_reference_audio,
            neutral_audio=request.neutral_reference_audio,
            target_speaker_audio=request.target_speaker_audio,
            layer_ids=config.generation.layer_ids,
        )

        speaker_vectors = hidden_extractor.compute_speaker_vectors(
            emotion_ref_audio=request.emotion_reference_audio,
            target_speaker_audio=request.target_speaker_audio,
            layer_ids=config.generation.layer_ids,
        )

        steering_vectors = make_clean_vectors(
            emotion_vectors=emotion_vectors,
            speaker_vectors=speaker_vectors,
            projection_removal=True,
        )

    candidates = candidate_generator.generate(
        text=request.text,
        instruction=instruction,
        target_speaker_audio=request.target_speaker_audio,
        steering_vectors=steering_vectors,
        alpha_grid=config.generation.alpha_grid,
        layer_sets=config.generation.layer_sets,
    )

    evaluated = evaluator.evaluate_all(
        candidates=candidates,
        target_speaker_audio=request.target_speaker_audio,
        emotion_reference_audio=request.emotion_reference_audio,
        text=request.text,
    )

    best = selector.select(evaluated)

    save_results(best, evaluated, request.output_dir)

    return best
```

---

## 15.2 Projection Removal

```python
def remove_projection(v_emo, v_spk, eps=1e-8):
    # v_emo から v_spk 方向成分を除去する。
    coef = np.sum(v_emo * v_spk) / (np.sum(v_spk * v_spk) + eps)
    return v_emo - coef * v_spk
```

---

## 15.3 Candidate Scoring

```python
def compute_total_score(scores, weights):
    return (
        weights["emotion_similarity"] * scores["emotion_similarity"]
        + weights["emotion_label_score"] * scores["emotion_label_score"]
        + weights["target_speaker_similarity"] * scores["target_speaker_similarity"]
        - weights["emotion_ref_speaker_similarity"] * scores["emotion_ref_speaker_similarity"]
        + weights["prosody_similarity"] * scores["prosody_similarity"]
        - weights["content_error"] * scores["content_error"]
        + weights["audio_quality"] * scores["audio_quality"]
    )
```

---

## 16. 実装上のリスク

| リスク | 内容 | 対策 |
|---|---|---|
| OmniVoiceがinstructionに弱い | promptだけでは感情が出ない | steeringを導入 |
| steeringで音質崩壊 | hidden加算が過剰 | alpha探索、layer限定、norm制御 |
| 話者性崩壊 | 感情参照話者の声質が漏れる | projection removal、speaker guard |
| emotion2vec評価の偏り | SERモデルの誤判定 | 主観評価も併用 |
| speaker encoderの偏り | 声質変化を過大/過小評価 | 複数encoderで比較 |
| 日本語感情表現の弱さ | SERモデルが英語寄り | 日本語音声で主観評価を重視 |
| 内容崩壊 | 感情を強めると発音が崩れる | ASR/CER制約 |
| 生成時間増大 | 候補生成数が多い | alpha/layer探索を段階化 |
| 評価器の日本語信頼性不足 | 自動スコアが主観と乖離し探索が artifact を最適化する | Phase 0 で spot check を実施し閾値・重みを補正 |
| hidden 抽出の粒度不足 | whole-utterance 単純平均で content/prosody/speaker が混入 | 同一テキスト対差分・有声フレーム平均を優先 |
| 線形分離仮定の破綻 | 感情と話者性が hidden 上で線形に分離できない | Phase 4a の geometric validation で事前検証、不可なら Phase 4 を見送る |
| speaker direction 定義の不確定性 | 3 候補で結果が大きく変わる | Phase 4a で linear probe による比較・選定 |
| hook 可能性の不明 | OmniVoice の特定 layer が hook できない | Phase 0 のアーキテクチャ調査で事前確認、不可なら Phase 3 を見送る |
| Phase 1 で十分なケースの見落とし | prompt-only で要件達成しているのに steering を実装してしまう | Phase 1 の天井観測を gate として明示 |

---

## 17. 推奨初期MVP

最初に作るべき最小構成は以下 (Phase 0 + Phase 1 + Phase 2 のスコープ)。

```text
EmotionAnalyzer
ProsodyAnalyzer
PromptComposer
OmniVoiceWrapper
SpeakerAnalyzer
CandidateEvaluator
BestCandidateSelector
```

MVPでは activation steering は入れない。Phase 0 のアーキテクチャ調査と評価器 spot check を実施した上で、Phase 1 の天井観測まで到達することを MVP の完了条件とする。

### MVPの流れ

```text
[Phase 0]
  OmniVoice アーキテクチャ調査 (hook 候補 layer 列挙)
  評価器 spot check (10〜20 サンプル、主観相関確認)
       ↓
[Phase 1]
  emotion reference audio
       ↓
  emotion/prosody 抽出
       ↓
  instruction 生成
       ↓
  OmniVoice で生成 (steering なし)
       ↓
[Phase 2]
  speaker/emotion 評価 (信頼性補正済み閾値)
       ↓
  best candidate 選択
       ↓
  prompt-only の天井観測と Phase 3-4 着手要否の判断
```

MVP の出力としては final.wav に加え、**「Phase 3-4 が必要か」の判定文書**を残すことを必須とする。これによって不要な steering 実装を回避できる。

MVP 完了後、必要と判断された場合に hidden steering / projection removal を Phase 3-4 として段階的に追加する。

---

## 18. 推奨実装順

1. OmniVoiceWrapper
2. PromptComposer
3. EmotionAnalyzer
4. ProsodyAnalyzer
5. SpeakerAnalyzer
6. CandidateEvaluator
7. BestCandidateSelector
8. CLI
9. HiddenExtractor
10. ActivationSteering
11. ProjectionRemoval
12. Black-box optimization
13. Optional VC / teacher comparison

---

## 19. テスト方針

### Unit Test

- remove_projection の出力が v_spk と直交に近いこと
- score計算が期待値通りであること
- threshold filtering が正しく動くこと
- prompt生成が必須項目を含むこと
- alpha=0 で steering が無効になること

### Integration Test

- 入力3点セットから final.wav が生成されること
- result.json が保存されること
- candidates が複数生成されること
- speaker threshold を下回る候補が棄却されること

### Regression Test

- 同一seed・同一configで同一または近似出力が得られること
- projection removal の on/off 比較結果が保存されること

---

## 20. 将来拡張

### 20.1 Adapter学習

学習データを用意できるようになった場合、以下に拡張する。

- Emotion Encoder
- Emotion Cross-Attention Adapter
- LoRA
- GRLによる speaker adversarial loss
- contrastive emotion loss
- orthogonal loss
- mutual information minimization

### 20.2 複数参照対応

複数の emotion reference audio から共通感情成分を抽出する。

```text
v_emotion = mean([
  hidden(emotional_ref_i) - hidden(neutral_ref_i)
])
```

これにより話者固有成分を平均で薄める。

### 20.3 UI

- target speaker prompt 選択
- emotion reference audio アップロード
- 感情強度スライダー
- speaker preservation スライダー
- 候補音声のAB比較
- emotion/speaker score 表示
- Pareto front 表示

---

## 21. 受け入れ基準

初期版では以下を満たせば成功とする。

- [ ] text, target_speaker_audio, emotion_reference_audio から final.wav を生成できる
- [ ] emotion reference の感情ラベルを抽出できる
- [ ] prosody特徴からinstructionを生成できる
- [ ] target speaker similarity を評価できる
- [ ] emotion similarity を評価できる
- [ ] 複数候補から最良候補を選択できる
- [ ] result.json に評価値が保存される
- [ ] speaker similarity threshold によって話者性崩壊候補を棄却できる
- [ ] projection removal の実装テストが通る
- [ ] READMEに実行方法が記載されている

---

## 22. コーディングエージェントへの実装指示

### 22.1 最初に実装すること

まず、OmniVoice本体改造なしのMVPを作成する。

実装対象：

```text
src/ovet/analyzers/emotion_analyzer.py
src/ovet/analyzers/prosody_analyzer.py
src/ovet/analyzers/speaker_analyzer.py
src/ovet/prompts/prompt_composer.py
src/ovet/omnivoice/wrapper.py
src/ovet/evaluation/evaluator.py
src/ovet/evaluation/selector.py
scripts/run_emotion_transfer.py
```

### 22.2 次に実装すること

MVPが動いたら hidden hook と activation steering を実装する。

実装対象：

```text
src/ovet/omnivoice/hidden_hooks.py
src/ovet/omnivoice/steering.py
scripts/extract_hidden.py
```

### 22.3 実装時の方針

- 最初は抽象クラスとダミー実装でもよい
- 各外部モデルは差し替え可能にする
- 設定はyamlで管理する
- result.json を必ず出力する
- 失敗した候補も捨てずにメタデータを保存する
- 音声ファイルパスは pathlib.Path を使う
- ログを残す
- 例外時にはどのモジュールで失敗したか分かるようにする

---

## 23. 最小README例

````markdown
# OmniVoice Emotion Transfer

Training-free emotion transfer pipeline for OmniVoice.

## Minimal Usage

```bash
python scripts/run_emotion_transfer.py \
  --text "今日は来てくれて、本当にありがとう。" \
  --target-speaker samples/target.wav \
  --emotion-ref samples/sad_ref.wav \
  --output-dir outputs/test01
```

## With Steering

```bash
python scripts/run_emotion_transfer.py \
  --text "今日は来てくれて、本当にありがとう。" \
  --target-speaker samples/target.wav \
  --emotion-ref samples/sad_ref.wav \
  --neutral-ref samples/neutral.wav \
  --enable-steering \
  --projection-removal \
  --alpha-grid 0.0,0.2,0.4,0.6 \
  --layers 8,12,16 \
  --output-dir outputs/test02
```
````

---

## 24. まとめ

本設計の中核は以下である。

```text
[Phase 0]   アーキテクチャ調査 + 評価器 spot check
              ↓ gate
[Phase 1]   感情参照音声から emotion/prosody を抽出
              ↓
            OmniVoice instruction に変換 (prompt-only)
              ↓ gate: 天井観測 + 不足要素の言語化
[Phase 2]   複数候補を生成し、信頼性検証済みの評価器で選別
              ↓ gate: ここで要件達成なら Phase 3 以降は不要
[Phase 3]   必要に応じて hidden activation steering で感情を強める
              ↓
[Phase 4a]  Geometric Validation Spike (線形分離可能性検証)
              ↓ gate
[Phase 4b]  感情方向から話者方向成分を射影除去
              ↓
[Phase 5]   black-box optimization で Pareto 最適候補を探索
              ↓
[Phase 6]   (オプション) VC / Teacher 比較
```

この方式は厳密な学習済み disentanglement ではないが、追加学習データなしで実装可能な範囲では、話者性崩壊を抑えつつ感情転写を強める現実的なアプローチである。

実装方針の要点は以下:

- **gate 駆動**: 各フェーズには明示的な前提条件を置き、満たさない限り次に進まない。これにより「効かない実装」を抱え込むリスクを抑える
- **Phase 1 は天井観測の場**: prompt-only で要件が達成できれば Phase 3-4 は不要。実装コストを節約する判断材料を必ず残す
- **評価器の事前検証**: 自動スコアと主観評価の相関を Phase 0 で確認しなければ、Phase 5 の最適化は artifact を最適化するだけになる
- **線形分離仮定の事前検証**: Phase 4 は本実装の前に Geometric Validation Spike を必ず通す
- **Hidden 抽出の精緻化**: whole-utterance 単純平均は最後の手段とし、同一テキスト対差分・有声フレーム平均を優先する
