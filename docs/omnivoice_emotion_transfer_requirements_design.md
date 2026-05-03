# OmniVoice Voice Cloning 感情保持強化パイプライン 要件定義書・設計書

## 1. 概要

本ドキュメントは、OmniVoice の voice cloning モードをベースに、追加学習データを用いずに **cross-lingual 環境での感情保持を強化** するための推論時制御パイプラインの要件定義および設計をまとめたものである。

本プロジェクトの主目的は、**同一話者の感情参照音声 (ref_audio) から voice cloning した際に、ターゲットテキストの言語が ref と異なる場合に発生する感情減衰を補正する** ことである。

本設計では、OmniVoice 本体の大規模再学習は行わない。代わりに、OmniVoice の inner LLM (Qwen3) に対する hidden state の activation steering、language direction の射影除去、複数候補生成・評価・選別ロジックを組み合わせる。

---

## 2. 背景

OmniVoice は voice cloning において **話者性 (timbre) は cross-lingual に保たれる** が、**感情成分 (特に valence と energy variance) はターゲット言語側へ正規化される** 傾向が観測される。これにより以下の問題が生じる。

- 感情のこもった参照音声を ref として渡しても、別言語のターゲットテキストでは感情が薄まる
- 同一話者の同一感情サンプルでも、ターゲット言語が ref 言語と異なると prosody が中立化される
- `instruct` パラメータは固定語彙 (gender / age / pitch / accent / whisper) のみ受け付け、自由記述による感情指定はできない
- pitch proxy (high/low) では「悲しみ」「怒り」「恐れ」などの細かい感情を表現できない

本設計の方針は、推論時に inner LLM (Qwen3) の hidden state に対して感情方向の steering を加え、必要に応じて language direction を射影除去することで、**voice cloning の話者性保持を維持したまま cross-lingual で薄まる感情成分を補強する** ことである。

---

## 3. 目的

### 3.1 主目的

追加学習データなしで、OmniVoice voice cloning における **cross-lingual 感情減衰** を補正する。具体的には:

- ref_audio の言語と異なるターゲットテキスト言語で生成した際、ref の感情 (valence / arousal / energy variance) を出力に保持する
- 主対象ユースケース:
  - **Case 1b**: ref_audio が target speaker の感情演技サンプル (同一ファイルが speaker と emotion の両方を兼ねる)
  - **Case 1c**: ref_audio が target speaker の中立サンプル、感情は別途注入する (同一話者の neutral → emotional)

### 3.2 副目的

- 感情強度をスライダー的に制御できるようにする (steering の alpha 連続制御)
- 生成候補を自動評価し、最良候補を選べるようにする
- 評価器の言語非依存性を確保する (V/A/D dimensional + prosodic 主軸)
- 将来的な LoRA / Adapter / 再学習に拡張可能な構成にする

---

## 4. 非目的

本プロジェクトでは、初期段階では以下を行わない。

- 別話者の感情を別話者の声に転写する **cross-speaker emotion transfer (Case 1a)**
  - これは emotion_ref の話者性 leakage 問題を伴い、学習なしでの解決が困難
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
| ref_audio | OmniVoice voice cloning に渡す参照音声 (話者性と感情の両方を担う) |
| ref_text | ref_audio に対応する書き起こし (Whisper 自動 or 手動) |
| target text | 生成対象テキスト (ref_audio と異なる言語のことが多い) |
| Case 1b | ref_audio = target speaker の感情演技サンプル (同一ファイル) |
| Case 1c | ref_audio = target speaker の中立サンプル、感情は別途注入 |
| cross-lingual emotion attenuation | ref と target text の言語が異なる際に感情成分が中立化される現象 (本設計の主対象) |
| activation steering | inner LLM (Qwen3) の hidden state に方向ベクトルを加算して出力傾向を変える手法 |
| emotion steering vector (v_emo) | neutral と emotional の hidden state 差分から作る感情方向ベクトル |
| language direction vector (v_lang) | 同一話者の言語間 hidden state 差分から作る言語方向ベクトル |
| projection removal | v_emo から v_lang 成分を射影除去し言語非依存な感情方向を抽出する処理 |
| V/A/D | Valence / Arousal / Dominance dimensional emotion (audeering モデル) |
| inference-time intervention | 学習ではなく推論時の hook と幾何処理によって出力を制御する手法 |

---

## 6. 全体アーキテクチャ

```text
[target text]   [target language]
       │              │
       ▼              ▼
[ref_audio (target speaker, ある感情)]
       │
       ├─ Whisper ASR ─────── ref_text (auto)
       ├─ Prosody Analyzer ── F0/energy/rate baseline
       └─ Emotion Probe ───── emotion label / V-A-D / e2v emb
                              │
                              ▼
              [Steering Vector Builder] (任意 / Phase 3+)
                              │
              ┌── v_emo: emotional - neutral hidden 差分
              ├── v_lang: same-speaker, language-pair hidden 差分
              └── v_emo_clean = v_emo − proj(v_emo, v_lang)
                              │
                              ▼
           [OmniVoice Generator (Qwen3 inner LLM)]
                              │
              ┌── voice cloning (ref_audio + ref_text)
              ├── instruct = pitch/style proxy (任意)
              ├── forward hook on layers[i] (Phase 3+)
              └── alpha * v_emo_clean を hidden に加算
                              │
                              ▼
                  [Multi-Candidate Generation]
                  (alpha grid × layer set × instruct proxy)
                              │
                              ▼
                  [Candidate Evaluator]
                  ├── V/A/D distance to ref (主)
                  ├── prosodic ratio to ref (F0_std, energy_std)
                  ├── emotion2vec embedding cos to ref
                  ├── speaker similarity to ref (sanity)
                  ├── content accuracy (ASR/CER)
                  └── audio quality
                              │
                              ▼
                  [Best Candidate Selector]
                              │
                              ▼
                          [Final WAV]
```

入力は **ref_audio 1 本**。話者性と感情はこの 1 本から取得する (Case 1b)。
Case 1c (ref が中立 + 感情を別注入) は Phase 3+ で steering vector を別音声から作る形で対応する。

---

## 7. 実装方針の対応関係

本設計は、voice cloning + cross-lingual emotion preservation を主対象とした再構成版である。元設計の案1〜5・案F〜Jのうち、本問題に有効なものを採用する。

### 7.1 採用する手法

| 手法 | 内容 | 本設計での位置付け |
|---|---|---|
| Voice cloning + auto ref_text | OmniVoice の標準モード。話者性は cross-lingual に保たれる | **基盤** |
| instruct proxy | 限定語彙 (pitch/style) を感情に粗くマッピング | Phase 1 補助 |
| Activation Steering | Qwen3 hidden に v_emo を加算し感情を強化 | **Phase 3 主役** |
| Language Projection Removal | v_emo から v_lang 成分を射影除去し言語非依存化 | **Phase 4 主役 (Phase 4a で線形分離可能性を検証)** |
| Multi-candidate generation + eval | alpha/layer の grid で生成し V/A/D + prosodic で選別 | Phase 2/5 |
| 後段 VC (OpenVoice等) | 学習データ依存のためオプション扱い | Phase 6 オプション |

### 7.2 採用しない手法とその理由

| 手法 | 不採用理由 |
|---|---|
| Cross-speaker emotion transfer (案 1a) | emotion_ref の話者性 leakage 問題が学習なしで解決困難 |
| Free-form natural language instruct | OmniVoice の `instruct` 制約により不可 (固定語彙のみ) |
| Speaker direction の射影除去 (元 Phase 4) | 同一話者前提では意味を持たない (代わりに language direction を射影除去) |
| 別話者の neutral 平均から speaker direction 推定 (元案H) | 同一話者前提で不要 |

---

## 8. 機能要件

## 8.1 入力

システムは以下を入力として受け取る。

| 入力 | 必須 | 説明 |
|---|---:|---|
| text | 必須 | 読み上げ対象テキスト |
| language | 推奨 | 生成言語 (Japanese / English / Chinese 等)。OmniVoice 性能向上のため指定推奨 |
| ref_audio | 必須 | 参照音声 1 本。話者性と感情の両方を担う (Case 1b) |
| ref_text | 任意 | ref_audio の書き起こし。未指定時は Whisper 自動転写 |
| neutral_ref_audio | 任意 | Phase 3 steering 用の中立音声 (Case 1c で必要)。同一話者推奨 |
| emotion_label_hint | 任意 | happy/sad/angry 等のユーザー指定 (instruct proxy mapping に使用) |
| steering_alpha | 任意 | Phase 3 hidden steering の強度 (0.0〜1.0+) |
| output_dir | 任意 | 出力保存先 |
| num_candidates | 任意 | 候補生成数 |
| alpha_grid | 任意 | steering 強度の探索値 |

`ref_audio` は target speaker の音声であることが前提。話者が異なる音声を渡した場合の挙動は本設計の対象外 (Case 1a 非対応)。

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

- ref_audio から感情ラベル、感情強度、emotion embedding、V/A/D を抽出する
- emotion_label_hint がある場合は、それを優先または補助情報として扱う
- emotion embedding は候補評価の `e2v_cos` に使用する
- V/A/D (audeering モデル) は主要評価軸として使用する

### 採用モデル

- **iic/emotion2vec_plus_base** (label + embedding)
- **audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim** (V/A/D dimensional)
- 不採用: emotion2vec_plus_large (baseline で _base より精度低かった)

### インターフェース

```python
class EmotionAnalyzer:
    def analyze(self, wav_path: str) -> EmotionAnalysis:
        ...

class VADAnalyzer:
    def analyze(self, wav_path: str) -> VADFeatures:
        ...
```

```python
@dataclass
class EmotionAnalysis:
    label: str
    confidence: float
    embedding: np.ndarray         # emotion2vec utterance-level
    logits: dict[str, float]      # 9-class probability

@dataclass
class VADFeatures:
    valence:   float   # 0..1 程度 (audeering 出力スケール)
    arousal:   float
    dominance: float
```

---

## 8.4 DSP特徴量抽出

### 要件

ref_audio と生成出力から以下の音響特徴を抽出する。生成出力側の値は ref に対する **比 (ratio)** として候補評価に使う。

| 特徴量 | 用途 |
|---|---|
| F0 mean | 声の高さ傾向 |
| F0 std / range | 抑揚の大きさ (cross-lingual で比較的保たれる) |
| energy mean | 声の強さ |
| energy std | 強弱変化 (**cross-lingual で減衰しやすい**) |
| speech rate | 話速 |
| pause ratio | 間の多さ |
| pause duration mean | 間の長さ |
| voiced ratio | 有声音比率 |

### 採用ライブラリ

- librosa (yin による F0、RMS による energy)
- pyworld (より精緻な F0 抽出、必要に応じ)
- silero-vad / webrtcvad (pause / voiced ratio)

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
    duration: float
```

---

## 8.5 Instruct Proxy 生成

### 要件と制約

OmniVoice の `instruct` パラメータは **固定語彙のみ** を受け付ける (自由記述は `ValueError`)。

**有効な英語語彙**:
- gender: `male / female`
- age: `child / teenager / young adult / middle-aged / elderly`
- pitch: `very low pitch / low pitch / moderate pitch / high pitch / very high pitch`
- style: `whisper`
- accents: `american / british / australian / canadian / chinese / indian / japanese / korean / portuguese / russian accent`

そのため、自然文での感情指示は不可能。EmotionAnalysis / ProsodyFeatures から **語彙内の proxy にマッピング** することのみ可能。

### Proxy mapping (初期実装)

| 感情ラベル | instruct proxy | 備考 |
|---|---|---|
| anger | `high pitch` | 高めの pitch + 自然な energy 増加を期待 |
| sad | `low pitch` | 低めの pitch + 沈んだ trajectory |
| fear | `high pitch, whisper` | 高 pitch + 弱声で不安感の近似 |
| happy | `high pitch` | anger と区別しにくい (語彙不足) |
| surprise | `high pitch` | 同上 |
| disgust | `low pitch` | 同上 |
| calm | `moderate pitch` | 中立 |

性別・年齢の自動検出 (audeering age-gender model) を併用して `female, low pitch` のように複合指示も可能。

### 限界

- 「悲しみ強度 0.7」のような連続制御は不可
- 同一 proxy で異なる感情を区別できない (anger と happy が両方 high pitch)
- これらの限界は Phase 3 (activation steering) で補う前提

### インターフェース

```python
class InstructProxyComposer:
    def compose(
        self,
        emotion_label: str,
        speaker_attrs: SpeakerAttrs | None = None,  # gender/age など (任意)
    ) -> str:
        """OmniVoice 語彙内の instruct 文字列を返す。"""
        ...
```

---

## 8.6 OmniVoice生成

### 要件

- text / language / ref_audio (+ ref_text) を入力として voice cloning 生成する
- 任意で instruct proxy を併用できる
- activation steering なしでも動作する (Phase 1 baseline)
- activation steering を有効化できる (Phase 3+)
- 複数 alpha 候補を生成できる (Phase 5)

### インターフェース

```python
class OmniVoiceWrapper:
    def generate(
        self,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str | None = None,    # None なら Whisper 自動転写
        instruct: str | None = None,     # InstructProxyComposer 出力
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
    emotion_vector: dict[int, np.ndarray] | None = None  # v_emo_clean (per layer)
    language_vector: dict[int, np.ndarray] | None = None  # v_lang (Phase 4 で使用)
    projection_removal: bool = True  # v_emo - proj(v_emo, v_lang)
```

---

## 8.7 Activation Steering (Qwen3 hidden 介入)

### 対象モデル

OmniVoice の inner LLM は **Qwen3 (28 layers, hidden_size=1024, 16 heads)**。
hook 対象は `model.llm.layers[i]` (`Qwen3DecoderLayer`)。

### 要件

inner LLM の中間 hidden state に対して、言語非依存な感情方向ベクトルを加算する。

```text
h_l' = h_l + alpha * v_emo_clean_l
```

ここで、

```text
v_emo_clean_l = v_emo_l - projection(v_emo_l, v_lang_l)
```

とする。**従来の speaker projection ではなく language projection** を採用するのは、本設計が同一話者前提 (Case 1b/1c) で速 leakage を考慮する必要がないためである。代わりに cross-lingual で混入する language-specific bias を除くことが目的。

### 感情方向ベクトル v_emo

`neutral_ref_audio` がある場合 (Case 1c):

```text
v_emo_l = hidden_l(emotional_ref) - hidden_l(neutral_ref)
```

`neutral_ref_audio` がない場合 (Case 1b):
- 同一話者の中立サンプルが入手できない時は、別話者でも可 (異なる話者性差分が混じるが、Phase 4a で線形分離可能性を検証する)
- もしくは Phase 4a の Geometric Validation Spike で「emotional 単独 hidden の中の感情成分」を抽出する手法を検討

```text
v_emo_l = hidden_l(emotional_ref) - hidden_l(generic_neutral_audio)
```

### 言語方向ベクトル v_lang

target text が L2 (例: 英語) で ref_audio が L1 (例: 日本語) の場合、L1 と L2 の発話を **同一話者で** ペアにして hidden を取り、

```text
v_lang_l = hidden_l(speaker_X_in_L1) - hidden_l(speaker_X_in_L2)
```

を抽出する。

候補とソース:
- **候補 1**: 同一話者の (JP 中立, EN 中立) ペアから抽出 (理想)
- **候補 2**: 多話者の言語ペアの平均から抽出 (パラレル多言語コーパスがあれば)
- **候補 3**: OmniVoice 自身を使い、同じテキスト内容の対訳を異なる言語で生成し、その出力 hidden の差分を v_lang として抽出 (zero-data な選択肢)

### 射影除去

```python
def remove_projection(v_emo: np.ndarray, v_lang: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """v_emo から v_lang 方向成分を射影除去。"""
    coef = np.sum(v_emo * v_lang) / (np.sum(v_lang * v_lang) + eps)
    return v_emo - coef * v_lang
```

### 複数層対応

```python
clean_vectors = {}
for layer_id in layer_ids:
    v_emo  = emotion_vectors[layer_id]
    v_lang = language_vectors[layer_id]
    if projection_removal:
        clean_vectors[layer_id] = remove_projection(v_emo, v_lang)
    else:
        clean_vectors[layer_id] = v_emo
```

### 実装上の注意

- hidden shape は `[batch, time, dim]` (Qwen3 標準)
- 差分を取る前に時間方向平均を取るか、token alignment を行う
- **whole-utterance の単純時間平均は最後の手段**。優先順位は以下:
  1. 同一テキスト読み上げ対 (neutral_ref と emotional_ref が同一発話内容) で frame-level 差分 → 平均
  2. VAD で有声フレームのみに限定した平均
  3. whole-utterance 単純平均 (sanity check 用途)
- layerごとに norm が異なるため、L2 normalize または RMS normalize を行う
- alpha は小さめから探索する
- emotion vector / language vector の生成パスは差分定義 (どの音声を引いたか) をメタデータとして保存し、再現性を確保する
- Qwen3 全 28 layer のうち感情に関与する layer は経験的に中間層 (8-20) と推測される。Phase 0 のアーキテクチャ調査結果に基づき初期 layer set を絞る

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

### 評価軸の優先順位 (baseline 検証で確定)

Phase 0 の spot check ([baseline v3](../baseline/outputs_v3/)) で 4 種類の評価器を比較し、以下の優先順位で採用する。

| 軸 | 評価器 | 言語非依存性 | 連続性 | 信頼性 | 優先度 |
|---|---|---|---|---|---|
| **V/A/D distance** | audeering wav2vec2-msp-dim | ◎ | ◎ | 高 | **主要メトリック** |
| **prosodic ratio** | librosa (F0/energy std) | ◎ | ◎ | 中 | 主要 |
| **emotion2vec embedding cos** | iic/emotion2vec_plus_base | △ | ◎ | 中 | 補助 |
| **emotion2vec p(target)** | 同上 | ✗ (ZH 偏重) | ✗ (二値的) | 低 | ZH 出力のみ |
| **speaker similarity** | ECAPA-TDNN | ○ | ◎ | 中 | sanity check |

emotion2vec_plus_large はベースライン検証で _base より悪かったため不採用。

### 各候補について以下のスコアを算出する。

| スコア | 計算式 | 目的 |
|---|---|---|
| `vad_dist` | `‖(V,A,D)_out - (V,A,D)_ref‖₂` | V/A/D 空間での ref からの距離 (低い=良い) |
| `valence_diff` | `\|V_out - V_ref\|` | valence (本問題で最も顕著に減衰する軸) |
| `arousal_diff` | `\|A_out - A_ref\|` | arousal (副軸) |
| `f0_std_ratio` | `f0_std_out / f0_std_ref` | 1.0 が ref と等価。ピッチ変化の保持度 |
| `energy_std_ratio` | `e_std_out / e_std_ref` | 1.0 が ref と等価。強弱変化の保持度 |
| `e2v_cos` | `cos(e2v_emb_out, e2v_emb_ref)` | emotion2vec embedding 類似度 |
| `e2v_p_target` | emotion2vec の target ラベル確率 | ZH 出力のみ参考 |
| `speaker_sim` | `cos(spk_emb_out, spk_emb_ref)` | 話者性保持の sanity (cloning ベースなので通常高い) |
| `content_error` | CER/WER (Whisper) | 読み上げ内容が崩れていないか |
| `audio_quality` | UTMOS / DNSMOS | 音質劣化検出 |

### スコア式

```text
score =
  - w_vad        * vad_dist
  - w_valence    * valence_diff
  - w_arousal    * arousal_diff
  - w_f0_dev     * |1 - f0_std_ratio|
  - w_energy_dev * |1 - energy_std_ratio|
  + w_e2v_cos    * e2v_cos
  - w_content    * content_error
  + w_quality    * audio_quality
```

初期値 (baseline v3 の知見に基づく):

```python
weights = {
    "vad":        1.0,   # 主要メトリック
    "valence":    0.8,   # 単軸で見ても重要
    "arousal":    0.4,
    "f0_dev":     0.3,
    "energy_dev": 0.5,   # cross-lingual で減衰しやすい
    "e2v_cos":    0.3,   # ZH 寄りバイアスあるため重み控えめ
    "content":    0.8,
    "quality":    0.5,
}
```

### hard constraint

候補選択時には以下を満たす候補のみを採用対象とする。

```text
content_error <= content_error_threshold
audio_quality >= quality_threshold
speaker_sim >= speaker_threshold  # cloning なので通常満たす、sanity 用
```

初期値:

```python
content_error_threshold = 0.20
quality_threshold       = 0.50
speaker_threshold       = 0.85   # cloning ベースのため厳しめでも通る前提
```

注記: 元設計の `emotion_ref_speaker_similarity` (別話者からの leakage 抑制) は本設計では不要 (同一話者前提)。

### 評価器信頼性の前提

baseline v3 で確認済みの注意点:
- emotion2vec の label probability は **JP 出力で大きく揺れる** (ja_a 1.000 / ja_b 0.062 のような変動)。**主スコアに使わない**
- V/A/D は言語横断で安定 (3 言語で similar magnitude)
- prosodic ratio は安定だが ref の絶対値が小さい場合 (sad の energy std=0.020 など) ノイズに敏感

Phase 5 の grid search に進む前に、評価器の安定性 (同一候補を 3 回生成して metric が ±0.05 以内に収まるか) を確認する。

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
        if c.content_error > thresholds.content_error:
            continue
        if c.audio_quality < thresholds.quality:
            continue
        if c.speaker_sim < thresholds.speaker:
            continue
        valid.append(c)

    if valid:
        return max(valid, key=lambda c: c.total_score)

    # fallback: 内容崩壊を最優先で避け、その上で V/A/D が ref に近いものを選ぶ
    return max(
        candidates,
        key=lambda c: (
            -c.content_error,
            c.audio_quality,
            -c.vad_dist,
            c.e2v_cos,
        )
    )
```

---

## 9. CLI設計

### 9.1 最小実行 (Phase 1: voice cloning baseline)

```bash
python run_emotion_clone.py \
  --text "Thank you so much for coming today." \
  --language English \
  --ref-audio jvnv_F1_anger.wav \
  --output-dir outputs/test01
```

ref_text は省略可能 (Whisper 自動転写)。`--language` を指定すると OmniVoice の言語推論精度が上がる。

### 9.2 instruct proxy 併用

```bash
python run_emotion_clone.py \
  --text "Thank you so much for coming today." \
  --language English \
  --ref-audio jvnv_F1_anger.wav \
  --emotion-hint anger \
  --output-dir outputs/test02
```

`--emotion-hint` から InstructProxyComposer が `"high pitch"` 等を生成し、OmniVoice の `instruct` に渡す。

### 9.3 Phase 3+: activation steering

```bash
python run_emotion_clone.py \
  --text "Thank you so much for coming today." \
  --language English \
  --ref-audio jvnv_F1_anger.wav \
  --neutral-ref neutral_F1.wav \
  --enable-steering \
  --alpha-grid 0.0,0.4,0.8,1.2 \
  --layers 8,12,16 \
  --projection-removal-language \
  --lang-pair-ref-l1 jvnv_F1_neutral_ja.wav \
  --lang-pair-ref-l2 jvnv_F1_neutral_en.wav \
  --output-dir outputs/test03
```

`--projection-removal-language` を有効にすると Phase 4 の v_lang 射影除去が走る。

### 9.4 評価のみ

```bash
python evaluate_candidates.py \
  --candidate-dir outputs/test03/candidates \
  --ref-audio jvnv_F1_anger.wav \
  --text "Thank you so much for coming today."
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
    instruct_proxy.yaml

  scripts/
    run_emotion_clone.py
    evaluate_candidates.py
    extract_hidden.py
    phase4a_validation_spike.py

  src/
    ovet/
      __init__.py

      analyzers/
        emotion_analyzer.py    # emotion2vec_plus_base
        vad_analyzer.py         # audeering V/A/D
        prosody_analyzer.py     # librosa F0/energy/rate
        speaker_analyzer.py     # ECAPA-TDNN
        asr_analyzer.py          # Whisper-large-v3-turbo
        quality_analyzer.py     # UTMOS/DNSMOS

      prompts/
        instruct_proxy.py       # 固定語彙 mapping
        templates.py

      omnivoice/
        wrapper.py              # voice cloning + auto ref_text
        hidden_hooks.py         # Qwen3 layers[i] forward hook
        steering.py             # alpha * v_emo_clean
        projection.py            # language projection removal

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
    language: str                      # "Japanese" / "English" / "Chinese" 等
    ref_audio: Path                    # target speaker の音声 (感情あり/なし)
    ref_text: str | None = None        # None なら Whisper 自動転写
    neutral_ref_audio: Path | None = None   # Phase 3 (Case 1c) で使用
    lang_pair_ref_l1: Path | None = None    # Phase 4 v_lang 抽出用 (同一話者 L1)
    lang_pair_ref_l2: Path | None = None    # Phase 4 v_lang 抽出用 (同一話者 L2)
    output_dir: Path = Path("outputs")
    emotion_label_hint: str | None = None   # instruct proxy mapping に使用

@dataclass
class Candidate:
    wav_path: Path
    instruct: str | None               # OmniVoice 固定語彙 proxy
    alpha: float                        # steering 強度 (0 で baseline)
    layer_ids: list[int]
    projection_removal_language: bool
    scores: dict[str, float]            # vad_dist, valence_diff, arousal_diff,
                                        # f0_std_ratio, energy_std_ratio,
                                        # e2v_cos, speaker_sim, content_error, audio_quality
    total_score: float

@dataclass
class Thresholds:
    content_error: float = 0.20
    quality:       float = 0.50
    speaker:       float = 0.85   # cloning ベース、sanity 用
```

---

## 12. 実装ステップ

各フェーズには gate (前段の完了条件) があり、gate を満たさない場合は次フェーズに進まない方針とする。

```text
Phase 0 (環境 + アーキテクチャ調査 + 評価器 baseline)  ✅ 完了
  ↓ gate: hook 可能 layer 確認 + 評価器の言語非依存性確保
Phase 1 (Voice Cloning baseline + instruct proxy)
  ↓ gate: cross-lingual 減衰量を定量化、Phase 3-4 着手要否を判断
Phase 2 (Multi-candidate + threshold filter)
  ↓ gate: 評価器の安定性確認 (反復生成で metric ±0.05)
Phase 3 (Activation Steering on Qwen3 hidden)
  ↓ gate: alpha=0 で baseline 一致、alpha 増で V/A/D が ref に近づく
Phase 4 (Language Projection Removal)
  ├─ 4a: Geometric Validation Spike (lang/emo 線形分離可能性)
  │   ↓ gate: linear probe で言語/感情が分離できる
  └─ 4b: 本実装
  ↓ gate: cross-lingual valence_diff 縮小、monolingual 大幅悪化なし
Phase 5 (Black-box Optimization)
  ↓ gate: 評価器が validated で artifact 最適化のリスクが低い
Phase 6 (Optional VC / Teacher 比較) -- オプション
```


## Phase 0: 環境構築・前提検証 ✅ 完了

- [x] OmniVoice をローカルで通常実行できるようにする (`/workspace/OmniVoice_v2/upstream/OmniVoice`, k2-fsa 公式)
- [x] 専用 venv 作成 (`/workspace/OmniVoice_v2/venv`, Python 3.11.10)
- [x] PyTorch 2.8.0+cu126 を A100 80GB で動作確認
- [x] OmniVoice voice cloning で日本語 / 英語の生成を確認 (`/workspace/OmniVoice_v2/audio_out/smoke_*.wav`)
- [x] **OmniVoice アーキテクチャ調査**
  - inner LLM = **Qwen3** (28 layers, hidden_size=1024, 16 heads)
  - hook 候補 = `model.llm.layers[i]` (Qwen3DecoderLayer)
  - audio I/O は `model.audio_embeddings` / `model.audio_heads` / `model.audio_tokenizer` (Higgs Audio V2)
  - **`instruct` パラメータは固定語彙のみ** (gender/age/pitch/whisper/accent)。自由記述は ValueError
- [x] **評価器の言語非依存性 spot check** (`/workspace/OmniVoice_v2/baseline/outputs_v3/`)
  - emotion2vec_plus_base: anger/sad/fear で reference P>0.99、ただし出力言語で大きく揺れる
  - emotion2vec_plus_large: _base より精度低かったため不採用
  - audeering V/A/D: 言語非依存で安定、主要メトリックに採用
  - prosodic (F0/energy std): 言語非依存で安定、補助メトリックに採用
- [x] **JVNV F1 (6 感情) で baseline 実験**
  - cross-lingual で valence_diff 0.15-0.32, energy_std_ratio 0.39-0.83 の減衰を観測
  - 主軸: V/A/D distance + prosodic ratio + e2v_cos

完了条件 (達成済):

- voice cloning が再現可能 ✓
- hook 可能 layer = Qwen3 28 layers ✓
- 評価器 4 軸の信頼性レンジが把握済み、Phase 2 の重み初期値を決定済 ✓
- baseline で cross-lingual 減衰の量と軸 (valence/energy_std) を特定済 ✓

---

## Phase 1: Voice Cloning baseline (Phase 3-4 必要性の判定 gate)

目的：OmniVoice 本体を改造せず、voice cloning + 任意の instruct proxy で **cross-lingual 環境で感情がどこまで保たれるか** を観測する。
本フェーズは同時に **proxy + cloning だけで満たせる範囲と、満たせない減衰量を定量化し、Phase 3-4 (steering) に進む必要性を判定する gate** として位置付ける。

- [ ] EmotionAnalyzer (emotion2vec_base) を実装
- [ ] VADAnalyzer (audeering V/A/D) を実装
- [ ] ProsodyAnalyzer (librosa) を実装
- [ ] InstructProxyComposer (固定語彙 mapping) を実装
- [ ] OmniVoiceWrapper (voice cloning + auto ASR ref_text) を実装
- [ ] 生成スクリプト (`scripts/run_emotion_clone.py`) を実装
- [ ] 出力音声を主観評価 (5-10 サンプル)
- [ ] **減衰量の定量化**: ref と output の V/A/D distance, valence_diff, energy_std_ratio を全 cross-lingual ペアで集計
- [ ] **不足要素の特定**: proxy のみで補正できる範囲と、できない範囲を明文化

完了条件：

- target language が ref と同じ場合に話者性・感情が大きく崩れずに出力される
- result.json に V/A/D / prosodic / e2v_cos / speaker_sim / CER が保存される
- cross-lingual ペアでの減衰量が baseline v3 比で定量レポートされている
- Phase 3-4 着手要否の判断材料が文書化されている

---

## Phase 2: Candidate生成・評価

目的：複数候補を生成し、cross-lingual emotion preservation と内容崩壊のバランスが良いものを自動選択する。

- [ ] SpeakerAnalyzer (ECAPA-TDNN) を実装
- [ ] CandidateEvaluator (4 軸統合) を実装
- [ ] BestCandidateSelector を実装
- [ ] alpha なし (Phase 3 前) の variation 生成: instruct proxy のバリエーションのみ
- [ ] result.json 保存を実装
- [ ] report.md 保存を実装
- [ ] 評価器の安定性確認: 同一 config で 3 回生成し、metric の標準偏差を計測

完了条件：

- 複数候補から最良候補を選択できる
- content_error / quality / speaker_sim threshold で崩壊候補を棄却できる
- 評価器の標準偏差が ±0.05 以内であり、Phase 5 grid search の信頼性が担保される

---

## Phase 3: Activation Steering

目的：OmniVoice内部hiddenに感情方向を足して、感情の出方を強める。

**前提**: Phase 0 で hook 可能な layer (Qwen3 28 layers) を確認済み。Phase 1 の cross-lingual 減衰量観測で「proxy + cloning では補正できない減衰」が定量化されていること。これらが揃わないうちは本フェーズに着手しない。

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

## Phase 4: Language Projection Removal

目的：感情方向ベクトル v_emo から **言語方向成分** v_lang を射影除去し、cross-lingual で混入する言語固有のバイアスを取り除く。

元設計の「speaker direction を引く」は同一話者前提では意味を持たないため、本フェーズでは射影除去の対象を **言語方向 v_lang** に変更する。

### Phase 4a: Geometric Validation Spike (本実装の前)

`v_emo - proj(v_emo, v_lang)` は「感情と言語性が hidden 空間で線形に分離可能」という強い仮定の上に乗っている。本実装に進む前に、この仮定を最低限のコストで検証する。

- [ ] 同一話者の (L1 中立, L2 中立) ペアを最低 3 話者分収集する
- [ ] PCA / linear probe で「言語軸」と「感情軸」の直交性を見る
- [ ] §8.7 の v_lang 候補 3 種 (同一話者ペア / 多話者平均 / OmniVoice self-generated) を比較し、最も言語情報を捕捉するものを 1 つ選定する
- [ ] projection removal 適用前後の hidden を language classifier に通し、言語情報が落ちることを確認する
- [ ] 同時に emotion classifier に通し、感情情報が保持されているかを確認する
- [ ] 結果が芳しくない場合は Phase 4 本実装を見送り、Phase 3 単独で着地させる判断材料とする

### Phase 4b: 本実装

- [ ] v_lang を Phase 4a で選定した定義で実装する
- [ ] remove_projection を実装する
- [ ] layerごとに projection removal を適用する
- [ ] projection removal on/off の比較評価を行う
- [ ] cross-lingual 出力での valence_diff / energy_std_ratio が改善するか確認する

完了条件：

- Phase 4a の geometric validation で線形分離の有効性が確認されている
- projection removal により cross-lingual 出力の valence_diff が縮小する
- monolingual 出力 (ref と同一言語) で大きな悪化が起きない

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

### 13.1 自動評価 (baseline 検証で優先順位確定)

| 指標 | ツール | 言語非依存 | 主要/補助 |
|---|---|---|---|
| **V/A/D distance to ref** | audeering wav2vec2-msp-dim | ◎ | **主要 (本問題の核)** |
| **valence_diff (単軸)** | 同上 | ◎ | **主要 (sad で最も顕著に減衰)** |
| **arousal_diff (単軸)** | 同上 | ◎ | 補助 |
| **prosodic ratio (F0_std, energy_std)** | librosa | ◎ | **主要 (energy std は cross-lingual で減衰)** |
| emotion2vec embedding cos to ref | iic/emotion2vec_plus_base | △ | 補助 |
| emotion2vec p(target) | 同上 | ✗ (ZH 偏重) | ZH 出力のみ |
| speaker similarity | ECAPA-TDNN / SpeechBrain | ○ | sanity (cloning ベース) |
| CER/WER | Whisper-large-v3-turbo | ◎ | 内容崩壊検出 |
| UTMOS / DNSMOS | 音質評価モデル | ○ | 音質劣化検出 |

ベースライン v3 で確認された注意:
- emotion2vec p(target) は **JP 出力で大きく揺れる** ため主スコアにしない
- emotion2vec_plus_large は _base より精度低かったため不採用
- F0_std は cross-lingual でも比較的保たれる (0.72-1.22 ratio)
- valence と energy_std が **cross-lingual で最も減衰する** → これが Phase 3-4 の改善対象

### 13.2 主観評価

最低限、以下の 5 段階評価を行う (cross-lingual ペアで)。

| 項目 | 評価 |
|---|---|
| ref と同じ感情に聞こえるか | 1〜5 |
| ref と同じ話者に聞こえるか | 1〜5 |
| 言語が target text と一致しているか | 1〜5 |
| 内容の聞き取りやすさ | 1〜5 |
| 音質 | 1〜5 |
| 総合自然性 | 1〜5 |

主観評価サンプル: cross-lingual ペア (例: ja_ref → en_out, zh_ref → ja_out) を最低 5 組、各 3 名の評価者で実施。

---

## 14. 設定ファイル例

```yaml
# configs/default.yaml

generation:
  # MVP では反復速度を確保するため候補数を絞る
  # Phase 5 (black-box optimization) で広げる
  num_candidates: 8
  alpha_grid: [0.0, 0.4, 0.8, 1.2]
  layer_sets:
    - []                # baseline (no steering)
    - [12]
    - [8, 12, 16]       # mid-range Qwen3 layers
  projection_removal_language: true   # v_emo - proj(v_emo, v_lang)

thresholds:
  content_error:    0.20
  quality:          0.50
  speaker:          0.85   # cloning ベース、sanity 用

scoring:
  vad:        1.0   # 主要メトリック (V/A/D distance to ref)
  valence:    0.8   # 単軸でも重視 (cross-lingual 減衰の主要因)
  arousal:    0.4
  f0_dev:     0.3   # |1 - f0_std_ratio|
  energy_dev: 0.5   # |1 - energy_std_ratio|, cross-lingual で減衰
  e2v_cos:    0.3   # ZH 偏重バイアス考慮し控えめ
  content:    0.8
  quality:    0.5

instruct_proxy:
  enabled: true
  emotion_to_pitch:
    anger:    "high pitch"
    sad:      "low pitch"
    fear:     "high pitch, whisper"
    happy:    "high pitch"
    surprise: "high pitch"
    disgust:  "low pitch"
    calm:     "moderate pitch"
  prepend_attrs: false   # gender/age を冒頭に追加するか

evaluation:
  primary_models:
    vad:       "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    e2v:       "iic/emotion2vec_plus_base"
    speaker:   "speechbrain/spkrec-ecapa-voxceleb"
    asr:       "openai/whisper-large-v3-turbo"
  prosodic:
    f0_method: "yin"
    f0_min:    50
    f0_max:    500
```

---

## 15. 主要アルゴリズム

## 15.1 Training-free Cross-Lingual Emotion Preservation

```python
def run_pipeline(request: GenerationRequest):
    # ref_audio から特徴を抽出 (analyzer は emotion2vec_base + audeering V/A/D + librosa)
    emotion = emotion_analyzer.analyze(request.ref_audio)
    vad     = vad_analyzer.analyze(request.ref_audio)
    prosody = prosody_analyzer.analyze(request.ref_audio)

    # InstructProxyComposer は OmniVoice 固定語彙内の文字列を返す (or None)
    instruct = instruct_proxy.compose(
        emotion_label=request.emotion_label_hint or emotion.label,
    )

    # ref_text 自動転写 (Whisper) または手動指定
    ref_text = request.ref_text or omnivoice.transcribe(request.ref_audio)

    steering_vectors = None
    if config.generation.enable_steering:
        # Phase 3: emotional - neutral hidden 差分から v_emo を作る
        emotion_vectors = hidden_extractor.compute_emotion_vectors(
            emotional_audio=request.ref_audio,
            neutral_audio=request.neutral_ref_audio,  # Case 1c なら必須
            layer_ids=config.generation.layer_ids,
        )

        if config.generation.projection_removal_language:
            # Phase 4: 同一話者の (L1, L2) ペアから v_lang を作り射影除去
            language_vectors = hidden_extractor.compute_language_vectors(
                l1_audio=request.lang_pair_ref_l1,
                l2_audio=request.lang_pair_ref_l2,
                layer_ids=config.generation.layer_ids,
            )
            steering_vectors = make_clean_vectors(
                emotion_vectors=emotion_vectors,
                language_vectors=language_vectors,
                projection_removal=True,
            )
        else:
            steering_vectors = emotion_vectors

    candidates = candidate_generator.generate(
        text=request.text,
        language=request.language,
        ref_audio=request.ref_audio,
        ref_text=ref_text,
        instruct=instruct,
        steering_vectors=steering_vectors,
        alpha_grid=config.generation.alpha_grid,
        layer_sets=config.generation.layer_sets,
    )

    evaluated = evaluator.evaluate_all(
        candidates=candidates,
        ref_audio=request.ref_audio,
        text=request.text,
    )

    best = selector.select(evaluated)

    save_results(best, evaluated, request.output_dir)

    return best
```

---

## 15.2 Language Projection Removal

```python
def remove_projection(v_emo, v_lang, eps=1e-8):
    # v_emo から v_lang 方向成分を除去する。
    coef = np.sum(v_emo * v_lang) / (np.sum(v_lang * v_lang) + eps)
    return v_emo - coef * v_lang
```

---

## 15.3 Candidate Scoring

```python
def compute_total_score(scores, weights):
    return (
        - weights["vad"]        * scores["vad_dist"]
        - weights["valence"]    * scores["valence_diff"]
        - weights["arousal"]    * scores["arousal_diff"]
        - weights["f0_dev"]     * abs(1.0 - scores["f0_std_ratio"])
        - weights["energy_dev"] * abs(1.0 - scores["energy_std_ratio"])
        + weights["e2v_cos"]    * scores["e2v_cos"]
        - weights["content"]    * scores["content_error"]
        + weights["quality"]    * scores["audio_quality"]
    )
```

---

## 16. 実装上のリスク

| リスク | 内容 | 対策 |
|---|---|---|
| OmniVoice instruct が固定語彙で表現力不足 | proxy では細かい感情制御ができない | Phase 3 (steering) で補強。instruct は補助のみ |
| steeringで音質崩壊 | hidden加算が過剰 | alpha探索、layer限定、norm制御 |
| Cross-lingual で energy_std が大きく減衰 | baseline で 0.39-0.83 に縮小を確認 | Phase 3 で v_emo を energy 軸方向に効かせる、layer set を広めに |
| Cross-lingual で valence が中立化 | baseline で sad の valence_diff 0.24-0.32 を確認 | Phase 4 の language projection removal で補正 |
| emotion2vec の JP 出力に対する不安定性 | 同一系候補でも 0.06-1.00 と振れる | 主スコアから外し V/A/D + prosodic 主軸に切替 (済) |
| 内容崩壊 | 感情を強めると発音が崩れる | ASR/CER制約 |
| 生成時間増大 | 候補生成数が多い | alpha/layer探索を段階化 |
| hidden 抽出の粒度不足 | whole-utterance 単純平均で content/prosody/speaker/lang が混入 | 同一テキスト対差分・有声フレーム平均を優先 |
| 線形分離仮定の破綻 | 感情と言語性が hidden 上で線形に分離できない | Phase 4a の geometric validation で事前検証、不可なら Phase 4 を見送る |
| v_lang 定義の不確定性 | 3 候補で結果が大きく変わる | Phase 4a で linear probe による比較・選定 |
| hook 可能性の不明 | Qwen3 の特定 layer が hook できない | Phase 0 のアーキテクチャ調査で事前確認 (Qwen3 28 layers, hook 可能を確認済) |
| Phase 1 で十分なケースの見落とし | proxy + cloning で要件達成しているのに steering を実装してしまう | Phase 1 の天井観測を gate として明示 |
| Case 1c (neutral ref) での感情注入失敗 | neutral 音声から始めて任意感情を注入するのは難しい | Phase 3-4 完成後にのみ着手、Case 1b で先に基盤を確立 |
| 話者・コーパスの偏り | JVNV F1 のみで検証 → 一般化保証なし | Phase 5 で multi-speaker テスト |

---

## 17. 推奨初期MVP

最初に作るべき最小構成は以下 (Phase 0 + Phase 1 + Phase 2 のスコープ)。

```text
InstructProxyComposer        # 感情ラベル → OmniVoice 語彙 mapping
ProsodyAnalyzer               # F0/energy/rate (librosa + pyworld)
EmotionAnalyzer               # emotion2vec_plus_base
VADProsodyAnalyzer            # audeering V/A/D
SpeakerAnalyzer               # ECAPA-TDNN
OmniVoiceWrapper              # voice cloning + ASR auto-transcribe
CandidateEvaluator            # V/A/D + prosodic + e2v_cos + speaker_sim + CER
BestCandidateSelector
```

MVPでは activation steering は入れない。Phase 0 のアーキテクチャ調査 (済: Qwen3 28 layers) と評価器 baseline (済: V/A/D + prosodic 主軸) を踏まえ、Phase 1 の cross-lingual 天井観測まで到達することを MVP の完了条件とする。

### MVPの流れ

```text
[Phase 0]  ✅ 完了
  OmniVoice = Qwen3 28 layers, hidden_size=1024
  hook 候補 layer = model.llm.layers[i]
  評価器 = audeering V/A/D + emotion2vec_base + librosa prosodic
  ベースライン: cross-lingual で valence と energy_std が減衰
       ↓
[Phase 1]
  ref_audio (target speaker, ある感情)
       ↓
  ref_text 自動転写 (Whisper) または手動指定
       ↓
  emotion/prosody 抽出 + instruct proxy 生成
       ↓
  OmniVoice voice cloning (steering なし)
       ↓
[Phase 2]
  V/A/D distance + prosodic ratio + e2v_cos で評価
       ↓
  best candidate 選択
       ↓
  cross-lingual 減衰の天井観測 (proxy 単体での補正幅を計測)
```

MVP の出力としては final.wav に加え、**「proxy + cloning で改善できる範囲を超える減衰量」の定量レポート**を残すことを必須とする。これが Phase 3-4 着手の判断材料となる。

MVP 完了後、必要と判断された場合に hidden steering / language projection removal を Phase 3-4 として段階的に追加する。

---

## 18. 推奨実装順

Phase 0 完了済み。次から:

1. OmniVoiceWrapper (voice cloning + auto ref_text)
2. InstructProxyComposer (固定語彙 mapping)
3. EmotionAnalyzer (emotion2vec_base wrapper)
4. ProsodyAnalyzer (librosa)
5. VADProsodyAnalyzer (audeering V/A/D, baseline で実装済)
6. SpeakerAnalyzer (ECAPA-TDNN)
7. CandidateEvaluator (4 軸統合)
8. BestCandidateSelector
9. CLI (run_emotion_clone.py)
10. HiddenExtractor (Qwen3 forward hook、Phase 3 起点)
11. ActivationSteering (Phase 3)
12. LanguageProjectionRemoval (Phase 4 — Phase 4a Spike 必須)
13. Black-box optimization (Phase 5)
14. Optional VC / teacher comparison (Phase 6)

---

## 19. テスト方針

### Unit Test

- `remove_projection(v_emo, v_lang)` の出力が `v_lang` と直交に近いこと (内積 ≈ 0)
- score 計算が期待値通りであること
- threshold filtering が正しく動くこと
- InstructProxyComposer が OmniVoice の vocabulary 内文字列を返すこと (ValueError を起こさない)
- alpha=0 で steering が無効になり、baseline 出力と一致すること

### Integration Test

- ref_audio + text + language から final.wav が生成されること
- ref_text 自動転写 (Whisper) が動くこと
- result.json に V/A/D / prosodic / e2v_cos / speaker_sim / CER が保存されること
- candidates が複数生成されること
- content_error / quality / speaker_sim threshold を下回る候補が棄却されること

### Regression Test

- 同一 seed・同一 config で同一または近似出力が得られること
- projection_removal_language の on/off 比較結果が保存されること
- baseline v3 の数値 (anger ja_a で V/A/D distance ≈ 0.18) が再現すること

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

target speaker の複数の感情演技サンプルから共通感情成分を抽出する (同一話者前提)。

```text
v_emotion = mean([
  hidden(emotional_ref_i) - hidden(neutral_ref_i)
])
```

これによりサンプル個別のノイズや録音条件のバイアスを平均で薄める。

### 20.3 UI

- ref_audio (target speaker, 感情あり/なし) のアップロード
- target text 入力
- target language 選択
- 感情強度スライダー (steering alpha)
- 候補音声の AB 比較
- V/A/D / prosodic 指標の可視化
- baseline (no-steering) 比の改善幅表示
- Pareto front 表示

---

## 21. 受け入れ基準

初期版では以下を満たせば成功とする。

- [ ] text, ref_audio, language から final.wav を生成できる (Case 1b)
- [ ] ref_audio の感情ラベルと V/A/D を抽出できる
- [ ] prosody 特徴 (F0_std, energy_std) を抽出できる
- [ ] InstructProxyComposer が OmniVoice 固定語彙内で proxy を生成できる
- [ ] cloning ベースで text 言語と異なる ref で final.wav を生成できる
- [ ] V/A/D distance / prosodic ratio / e2v_cos / speaker_sim / CER を評価できる
- [ ] 複数候補から最良候補を選択できる
- [ ] result.json に全指標が保存される
- [ ] content_error / quality threshold によって崩壊候補を棄却できる
- [ ] language projection removal の unit test が通る (内積直交性)
- [ ] README に実行方法が記載されている
- [ ] baseline v3 の cross-lingual 減衰量に対する改善幅をレポートできる

---

## 22. コーディングエージェントへの実装指示

### 22.1 最初に実装すること

まず、OmniVoice本体改造なしのMVPを作成する (Phase 0 完了済み、Phase 1-2 着手)。

実装対象：

```text
src/ovet/analyzers/emotion_analyzer.py        # emotion2vec_plus_base wrapper
src/ovet/analyzers/vad_analyzer.py             # audeering V/A/D wrapper (baseline で実装済を整理)
src/ovet/analyzers/prosody_analyzer.py         # librosa F0/energy/rate
src/ovet/analyzers/speaker_analyzer.py         # ECAPA-TDNN
src/ovet/prompts/instruct_proxy.py             # 固定語彙 mapping
src/ovet/omnivoice/wrapper.py                  # voice cloning + auto ASR
src/ovet/evaluation/evaluator.py               # 4 軸統合
src/ovet/evaluation/selector.py
scripts/run_emotion_clone.py
scripts/evaluate_candidates.py
```

### 22.2 次に実装すること

MVP が動いたら hidden hook と activation steering を実装する (Phase 3+)。

実装対象：

```text
src/ovet/omnivoice/hidden_hooks.py     # Qwen3 layers[i] forward hook
src/ovet/omnivoice/steering.py          # alpha * v_emo_clean を追加
src/ovet/omnivoice/projection.py        # language projection removal
scripts/extract_hidden.py               # v_emo / v_lang 抽出ユーティリティ
scripts/phase4a_validation_spike.py     # 線形分離可能性検証
```

### 22.3 実装時の方針

- 最初は抽象クラスとダミー実装でもよい
- 各外部モデルは差し替え可能にする
- 設定は yaml で管理する
- result.json には全指標 (V/A/D / prosodic / e2v / speaker / content) を保存する
- 失敗した候補も捨てずにメタデータを保存する
- 音声ファイルパスは `pathlib.Path` を使う
- ログを残す
- 例外時にはどのモジュールで失敗したか分かるようにする
- Phase 0 baseline (`baseline/outputs_v3/results.json`) を回帰の基準に保つ

---

## 23. 最小README例

````markdown
# OmniVoice Cross-Lingual Emotion Preservation

Training-free pipeline that preserves emotion in OmniVoice voice cloning when
the target text language differs from the reference audio language.

## Minimal Usage (Phase 1 baseline)

```bash
python scripts/run_emotion_clone.py \
  --text "Thank you so much for coming today." \
  --language English \
  --ref-audio samples/jvnv_F1_anger.wav \
  --output-dir outputs/test01
```

ref_text は省略可 (Whisper 自動転写)。

## With instruct proxy

```bash
python scripts/run_emotion_clone.py \
  --text "Thank you so much for coming today." \
  --language English \
  --ref-audio samples/jvnv_F1_anger.wav \
  --emotion-hint anger \
  --output-dir outputs/test02
```

## With activation steering (Phase 3+)

```bash
python scripts/run_emotion_clone.py \
  --text "Thank you so much for coming today." \
  --language English \
  --ref-audio samples/jvnv_F1_anger.wav \
  --neutral-ref samples/neutral_F1.wav \
  --enable-steering \
  --projection-removal-language \
  --lang-pair-ref-l1 samples/F1_neutral_ja.wav \
  --lang-pair-ref-l2 samples/F1_neutral_en.wav \
  --alpha-grid 0.0,0.4,0.8,1.2 \
  --layers 8,12,16 \
  --output-dir outputs/test03
```
````

---

## 24. まとめ

### 解きたい問題 (再定義)

OmniVoice の **voice cloning は話者性 (timbre) を cross-lingual に保つが、感情成分 (特に valence と energy variance) は target 言語側に正規化される**。本設計は学習データなしで、この cross-lingual emotion attenuation を補正することを目的とする。

対象ユースケース:
- **Case 1b** (主): ref_audio が target speaker の感情演技サンプル → 別言語のテキストで cloning しても感情を保つ
- **Case 1c** (副): ref_audio が target speaker の中立サンプル → 別言語/別感情のテキストで cloning する際に感情を注入する

### コアパイプライン

```text
[Phase 0]  ✅ 完了
  - OmniVoice = Qwen3 28 layers (hook 可能を確認)
  - 評価器 = audeering V/A/D + emotion2vec_base + librosa prosodic
  - baseline (JVNV F1×3 感情×6 テキスト×3 言語)
    → cross-lingual で valence_diff 0.15-0.32, energy_std_ratio 0.39-0.83 を観測

       ↓ gate: hook 確認 + 評価器の言語非依存性確保

[Phase 1]  Voice Cloning baseline (instruct proxy 任意)
  ref_audio + (auto ref_text via Whisper) + target language
       → OmniVoice cloning
       → V/A/D + prosodic + e2v_cos で評価

       ↓ gate: cross-lingual 減衰量を定量化、Phase 3-4 着手要否を判断

[Phase 2]  Multi-candidate + threshold filter

       ↓ gate: 評価器の安定性確認 (反復生成で metric ±0.05)

[Phase 3]  Activation Steering on Qwen3
  v_emo = hidden(emotional_ref) - hidden(neutral_ref)
  h_l' = h_l + alpha * v_emo

       ↓ gate: alpha=0 で baseline 一致、alpha 増で V/A/D が ref に近づく

[Phase 4a] Geometric Validation Spike
  - 線形分離可能性 (lang vs emo)
  - v_lang 候補 3 種から 1 つ選定

       ↓ gate: linear probe で言語/感情の分離を確認

[Phase 4b] Language Projection Removal
  v_emo_clean = v_emo - proj(v_emo, v_lang)
       → cross-lingual で valence_diff が縮小、monolingual で大幅悪化なし

       ↓

[Phase 5]  Black-box optimization
  alpha × layer set × proxy の grid で Pareto 最適候補を探索

       ↓

[Phase 6]  (オプション) VC / Teacher 比較
```

### 実装方針の要点

- **gate 駆動**: 各フェーズに明示的な前提条件を置き、満たさない限り次に進まない
- **同一話者前提**: target_speaker と emotion 参照は同一話者 (同一ファイルまたは同一話者の別音声)。**話者 leakage は問題にならず、その代わり language 方向が射影除去対象**
- **`instruct` 制約への対応**: 自由記述不可なので Phase 1 は固定語彙の proxy mapping にとどめ、本命の感情制御は Phase 3 (steering) に置く
- **評価軸の優先**: V/A/D dimensional + prosodic ratio が言語非依存で安定。emotion2vec p(target) は出力言語によって不安定なので主スコアにしない
- **Hidden 抽出の精緻化**: whole-utterance 単純平均は最後の手段とし、同一テキスト対差分・有声フレーム平均を優先する
- **線形分離仮定の事前検証**: Phase 4b の本実装前に必ず Phase 4a の Geometric Validation Spike を通す
- **回帰の基準**: Phase 0 baseline (`baseline/outputs_v3/results.json`) を保持し、後続フェーズの改善幅を常に baseline 比で報告する
