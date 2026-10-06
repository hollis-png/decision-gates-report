# Confidence gates of typed decision models under growing option sets

Code and per-request scores for the technical report
*Confidence Gates of Typed Decision Models under Growing Option Sets: A Technical Report* (Yu-Feng Yen, 2026).

The report studies what happens to a confidence gate (auto-handle a request when the top probability is at least τ)
when options are added to a routing table, for Laya, Von and Qwen3-4B on Banking77 and CLINC150.
It is a diagnosis of released models, not a new method. Code comments are partly in Chinese.

## Contents

```
code/                 experiment and analysis scripts
data/scores.tar.xz    per-request raw scores and temperatures (45 MB; unpack in code/)
requirements.txt      versions used for the reported runs
```

`tar -xJf data/scores.tar.xz -C code/` restores `a2_results/`, `a5_results/`, `iia_v3_results/`, `iia_v2_results/`,
`launch_results/` and `von_len_results/`. Every record stores option keys, raw scores and the model's own temperature,
so all calibrations are recomputed offline. No utterance text is included (the datasets are loaded from Hugging Face).

**Not included:** responses of the commercial model Jev (its terms of use were not checked for redistribution);
`jev_eval.py` and `jev_separate.py` reproduce them with your own key in the environment variable `TYPESAFE_API_KEY`.

## Scoring (GPU)

| Script | What it scores | Used for |
|---|---|---|
| `a2_dose.py` | routing sweep K = 2..12, one option added at each K (option list reshuffled per K) | Table 1, Figures 1–2 |
| `a5_large.py` | order-preserving sweep up to 77 / 150 options; `*_long` = Laya with a raised option budget; `qwen3_4b_num` = number labels | Section 6, Figure 4 |
| `launch.py` | route-launch simulation (five distractors, then plus the gold intent) | launch table |
| `iia_v3.py` | five-option base sets with edits (shuffle, add 1/3/5, …) | reordering reference points |
| `von_len.py` | neutral sentences appended to the state | Von temperature / gate flips |
| `jev_eval.py`, `jev_separate.py` | the same edits on Jev (needs an API key) | Jev numbers |

Models: Laya checkpoints from `convaiinnovations/laya` (placed in `./laya_hf`), Von from `wfzyx/von`,
Qwen3-4B-Instruct-2507 and the other LLMs from Hugging Face.

## Analysis (CPU, from the stored scores)

| Script | Output |
|---|---|
| `layers.py` | coverage shift and reordering per K (Table 1, Figure 1) |
| `review5.py` | matched-coverage decomposition, cross-dataset calibration, reordering reference points, launch intervals (appendix tables) |
| `calib_methods.py` | calibrators (single, bucketed, smooth T(K), top-score Platt) and the fitting/evaluation split |
| `calib_variance.py`, `calib_cross.py` | bucket-temperature noise, cross-dataset fits |
| `a5_analysis.py`, `a5_frontier.py`, `a5_riskctl.py`, `a5_fig.py` | large option sets: per-K metrics, ceiling at 5% error, Learn-then-Test thresholds, Figure 4 |
| `paper_figs.py` | Figures 1–2 (`python paper_figs.py <out_dir> fig1v3 fig2`) |
| `ci_tables.py`, `tradeoff_rt.py`, `hybrid.py` | LLM confidence comparison (appendix) |

The fitting/evaluation split hashes an utterance key so that both distractor variants and all K of an utterance fall on
the same side; bootstrap intervals resample utterances.

## License

MIT (code). The per-request scores are derived from Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0) and the typed-decisions test set (see its dataset card).
