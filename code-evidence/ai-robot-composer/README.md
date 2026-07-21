# AIVAD Robot Composer

Final year bachelors project for generating piano MIDI conditioned on continuous Valence, Arousal, and Dominance (VAD) values. The model is a PyTorch hierarchical variational autoencoder with separate short-term and long-term latent spaces, recurrent sequence modelling, and prediction heads for notes, durations, and velocities.

This repository is a curated, runnable version of the original notebook submission. It keeps the trained model, vocabulary, representative generated samples, inference path, evaluation utilities, and core training loss while removing notebook state, duplicate MIDI collections, and an 827 MB cached dataset.

## Evidence

- 310,748 training sequences of 64 events were represented in the original cached dataset.
- The learned vocabulary contains 4,343 note and chord tokens plus 11 duration classes.
- The original notebook recorded note perplexity of 1.7574.
- It recorded short-term and long-term KL divergence of 11.1348 and 2.2843 respectively.
- Human evaluation was reported as an approximately 18% improvement in perceived dominance expression over baseline outputs. The raw questionnaire responses were not present in the available export, so this claim is reported rather than independently recalculated here.

See [RESULTS.md](RESULTS.md) for model provenance and reproducible analysis of the included MIDI samples.

## Architecture

```text
note + duration embeddings
          |
bidirectional short-term LSTM
          |
short-term latent distribution
          |
three-layer long-term LSTM + VAD projection
          |
long-term latent distribution
          |
LSTM decoder
          |
note, duration, and velocity outputs
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Run The Trained Model

Inspect the checkpoint and vocabulary:

```bash
robot-composer inspect
```

Run a checkpoint forward-pass test:

```bash
robot-composer smoke-test
```

Generate a MIDI file with VAD conditioning:

```bash
robot-composer generate --valence 0.8 --arousal 0.2 --dominance 0.6 --output outputs/example.mid
```

Use `--c-major-only` to constrain every generated note or chord to C major. Generation also supports `--length`, `--temperature`, `--top-k`, `--tempo`, and `--seed`.

Analyze a MIDI file:

```bash
robot-composer analyze samples/generated_V1.0_A1.0_D1.0.mid
```

Regenerate the sample results page:

```bash
robot-composer report --output RESULTS.md
```

Run the automated checks:

```bash
python -m unittest discover -s tests
```

Equivalent module commands begin with `python -m robot_composer`.

## Repository Contents

```text
.
|-- artifacts/
|   |-- V7.1_final_paper_VAD.pth   # original trained state dictionary
|   `-- vocabulary.json            # extracted inference metadata
|-- samples/                       # eight representative VAD corner samples
|-- src/robot_composer/
|   |-- analysis.py
|   |-- cli.py
|   |-- generation.py
|   |-- model.py
|   |-- reporting.py
|   |-- training.py
|   `-- vocabulary.py
|-- tests/
|-- RESULTS.md
|-- pyproject.toml
`-- README.md
```

## Reproducibility Boundary

Inference, MIDI generation, sample analysis, and model smoke testing are self-contained. Full retraining is not self-contained because the original work used the external EMOPIA MIDI corpus, MuseScore repairs, FluidSynth, and a soundfont for audio-derived dominance features. Those third-party assets are intentionally excluded.


## Data And Publication Note

The original project used the EMOPIA dataset. This curated repository includes only model-generated MIDI examples, not the source corpus. Before publishing university work, confirm the applicable coursework-sharing rules and choose an appropriate licence for your own code.

## References

1. Rodà, Canazza, and De Poli, "Clustering Affective Qualities of Classical Music: Beyond the Valence-Arousal Plane," IEEE Transactions on Affective Computing, 2014.
2. Hung et al., "EMOPIA: A Multi-Modal Pop Piano Dataset for Emotion Recognition and Emotion-Based Music Generation," 2021.

