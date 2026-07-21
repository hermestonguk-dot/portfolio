# Model Artefacts

`V7.1_final_paper_VAD.pth` is the trained PyTorch state dictionary from the final notebook submission.

`vocabulary.json` is mechanically extracted from the original `train_dataset.pth`. It contains only the note/chord mapping, duration classes, sequence length, velocity scale, and sequence count needed to interpret the checkpoint. The original 827 MB cached tensor dataset is intentionally excluded.

