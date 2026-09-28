# Fashion-MNIST CNN

This coursework project trains a compact PyTorch convolutional classifier on
Fashion-MNIST. The model uses three convolutional blocks, batch normalisation,
global average pooling, and a small classifier head. Training includes data
augmentation, a reproducible repeated-split hyperparameter search, validation
tracking, and saved weights.

## Files

- `fashion_model.py`: the CNN definition.
- `engine.py`: one-epoch training and validation loops.
- `fashion_training.py`: data loading, augmentation, model training, and
  hyperparameter search.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python fashion_training.py
```

Fashion-MNIST is downloaded to `data/` on the first run. Training writes logs
and model weights under `submission/`, which are local outputs and are not
included in the portfolio.
