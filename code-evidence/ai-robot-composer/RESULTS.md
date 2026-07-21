# Robot Composer Results

This report combines the evaluation recorded in the final notebook with a fresh, reproducible analysis of the representative generated MIDI files shipped here.

## Model Artefact

| Statistic | Value |
|---|---:|
| Trainable parameters | 12,550,659 |
| Checkpoint size | 47.9 MB |
| Note/chord vocabulary | 4,343 tokens |
| Duration vocabulary | 11 classes |
| Training sequence length | 64 events |
| Cached training sequences | 310,748 |

## Original Notebook Evaluation

| Metric | Recorded value | Interpretation |
|---|---:|---|
| Note perplexity | 1.7574 | Lower values indicate more confident next-token prediction |
| Short-term KL divergence | 11.1348 | Local latent-space regularization |
| Long-term KL divergence | 2.2843 | Phrase-level latent-space regularization |
| Example average pitch | 59.50 | MIDI pitch number |
| Example average duration | 0.667 | Quarter lengths |
| Example average velocity | 84.48 | MIDI velocity |

The associated CV reports an approximately 18% improvement in perceived dominance expression over baseline outputs. The available export does not contain the raw human-study responses, so that result has not been independently recalculated in this repository.

## Included Sample Analysis

| Sample V/A/D | Events | Chords | Avg pitch | Pitch spread | Avg duration | Avg velocity |
|---|---:|---:|---:|---:|---:|---:|
| [0.0 / 0.0 / 0.0](samples/generated_V0.0_A0.0_D0.0.mid) | 148 | 12 | 65.97 | 15.37 | 0.669 | 92.11 |
| [0.0 / 0.0 / 1.0](samples/generated_V0.0_A0.0_D1.0.mid) | 142 | 18 | 60.97 | 15.91 | 0.613 | 92.07 |
| [0.0 / 1.0 / 0.0](samples/generated_V0.0_A1.0_D0.0.mid) | 147 | 8 | 59.56 | 8.85 | 0.716 | 87.42 |
| [0.0 / 1.0 / 1.0](samples/generated_V0.0_A1.0_D1.0.mid) | 142 | 10 | 67.68 | 16.42 | 0.746 | 92.44 |
| [1.0 / 0.0 / 0.0](samples/generated_V1.0_A0.0_D0.0.mid) | 139 | 12 | 61.35 | 10.39 | 0.676 | 86.25 |
| [1.0 / 0.0 / 1.0](samples/generated_V1.0_A0.0_D1.0.mid) | 147 | 4 | 66.05 | 13.85 | 0.745 | 85.14 |
| [1.0 / 1.0 / 0.0](samples/generated_V1.0_A1.0_D0.0.mid) | 142 | 12 | 67.39 | 11.24 | 0.715 | 88.42 |
| [1.0 / 1.0 / 1.0](samples/generated_V1.0_A1.0_D1.0.mid) | 149 | 22 | 62.97 | 14.56 | 0.700 | 89.10 |

Across the included samples:

- Mean event count: **144.5**
- Mean pitch: **63.99**
- Mean duration: **0.698**
- Mean velocity: **89.12**

These descriptive statistics confirm that each checked-in artefact is parseable MIDI and make its pitch, rhythm, and dynamics inspectable. They are not evidence that the conditioning dimensions were perceived correctly; that requires a controlled listener study.

## Reproduction

```bash
robot-composer smoke-test
robot-composer report --output RESULTS.md
python -m unittest discover -s tests
```
