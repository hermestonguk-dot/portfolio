"""Train and validate the compact Fashion-MNIST convolutional classifier."""
import os
import numpy as np
import torch, torchvision
import csv

try:
    from . import engine
    from .fashion_model import Net
except ImportError:  # Supports running the file directly from this folder.
    import engine
    from fashion_model import Net


def train_fashion_model(fashion_mnist, 
                        n_epochs, 
                        batch_size=4,
                        learning_rate=0.001,
                        USE_GPU=False,
                        weight_decay=1e-4):
    """Train the model and return the best validation state dictionary."""
    # Optionally use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create train-val split
    train_size = int(0.8 * len(fashion_mnist))
    val_size = len(fashion_mnist) - train_size
    train_data, val_data = torch.utils.data.random_split(fashion_mnist, [train_size, val_size])

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             )

    # Initialize model, loss function, and optimizer
    model = Net()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,)
    best_state = None
    best_val_acc = -np.inf
    
    history = []
    # Training loop
    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = engine.eval(model, val_loader, criterion, device)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        train_acc = correct / total

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy:{train_acc:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        #saves history for plotting
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        
    os.makedirs("submission", exist_ok=True)
    csv_path = os.path.join("submission", "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training log to {csv_path}")
    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    return model.state_dict()


def get_transforms(mode='train'):
    """
    Define any data augmentations or preprocessing here if needed.
    Only standard torchvision transforms are permitted (no lambda functions), please check that 
    these pass by running model_calls.py before submission. Transforms will be set to .eval()
    (deterministic) mode during evaluation, so avoid using stochastic transforms like RandomCrop
    or RandomHorizontalFlip unless they can be set to p=0 during eval.
    """
    mean,std = 0.286, 0.353
    
    if mode == 'train':
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
            torchvision.transforms.RandomCrop(28, padding=2),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize((mean,), (std,)),
        ])
        
    elif mode == 'eval': # no stochastic transforms, or use p=0
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
            torchvision.transforms.Normalize((mean,), (std,)),
        ])
        for tf in tfs.transforms:
            if hasattr(tf, 'train'):
                tf.eval()  # set to eval mode if applicable # type: ignore
    else:
        raise ValueError(f"Unknown mode {mode} for transforms, must be 'train' or 'eval'.")
    return tfs


def load_training_data():
    # Load FashionMNIST dataset
    # Do not change the dataset or its parameters
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
    )
    # We load in data as the raw PIL images - recommended to have a look in visualise_dataset.py! 
    # To use them for training or inference, we need to transform them to tensors. 
    # We set this transform here, as well as any other data preprocessing or augmentation you 
    # wish to apply.
    fashion_mnist.transform = get_transforms(mode='train')
    return fashion_mnist

from torch.utils.data import DataLoader, random_split

def MCCV(
    fashion_mnist,
    n_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    device,
    seed: int = 0,
):
    """
    Train Net on one random train/val split and return the best validation accuracy.
    Uses engine.train / engine.eval for the inner loops.
    """

    # Make the split reproducible
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_size = int(0.8 * len(fashion_mnist))
    val_size = len(fashion_mnist) - train_size
    train_data, val_data = torch.utils.data.random_split(fashion_mnist, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             )

    model = Net()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,)

    best_val_acc = -np.inf
    history = []
    
    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = engine.eval(model, val_loader, criterion, device)

        print(
            f"[seed={seed}] Epoch {epoch+1}/{n_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
    return best_val_acc, history

def hyperparameter_search(fashion_mnist, device, n_folds: int = 3):
    """
    hyperparameter search with repeated 80/20 splits, each configuration is repeated n_folds times with differing seeds.
    Returns the best configuration.
    """

    param_grid = [
        {
            "name": "cfg_A",
            "n_epochs": 10,      
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "name": "cfg_B",
            "n_epochs": 10,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 5e-5,  
        },
        {
            "name": "cfg_C",
            "n_epochs": 10,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 0,  
        },
        {
            "name": "cfg_D",
            "n_epochs": 10,
            "batch_size": 64,
            "learning_rate": 5e-4,
            "weight_decay": 5e-5,
        },
    ]

    best_config = None
    best_cv_acc = -np.inf
    results = []

    
    for cfg in param_grid:
        fold_accs = []
        print(f"\n=== Testing hyperparameter config: {cfg['name']} ===")

        for fold in range(n_folds):
            acc, history = MCCV(
                fashion_mnist=fashion_mnist,
                n_epochs=cfg["n_epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
                device=device,
                seed=fold,
            )
            fold_accs.append(acc)
            
            for record in history:
                results.append(
                    {
                        "config": cfg["name"],
                        "fold": fold,
                        "epoch": record["epoch"],
                        "train_loss": record["train_loss"],
                        "val_loss": record["val_loss"],
                        "val_acc": record["val_acc"],
                        "n_epochs": cfg["n_epochs"],
                        "batch_size": cfg["batch_size"],
                        "learning_rate": cfg["learning_rate"],
                        "weight_decay": cfg["weight_decay"],
                    }
                )
                
        mean_acc = float(np.mean(fold_accs))
        print(
            f"Config {cfg['name']} "
            f"fold_accs={fold_accs} "
            f"mean_val_acc={mean_acc:.4f}"
        )

        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_config = cfg

    print(
        f"\nBest config {best_config['name']} "
        f"(mean_val_acc={best_cv_acc:.4f})"
    )
    os.makedirs("submission", exist_ok=True)
    csv_path = os.path.join("submission", "hyperparam_search_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "fold",
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "n_epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved hyperparameter search {csv_path}")

    return best_config

def main():
    # example usage
    # you could create a separate file that calls train_fashion_model with different parameters
    # or modify this as needed to add cross-validation, hyperparameter tuning, etc.
    fashion_mnist = load_training_data()

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 3. Hyperparameter search / cross-validation
    best_cfg = hyperparameter_search(fashion_mnist, device, n_folds=3)

    # 4. Final training with the best hyperparameters, but longer n_epochs
    #    Here we call train_fashion_model (as required by the coursework)
    final_n_epochs = 30  # e.g. your Config G final setting

    print(
        f"\nFinal training with best hyperparameters "
        f"(lr={best_cfg['learning_rate']}, wd={best_cfg['weight_decay']}, "
        f"batch_size={best_cfg['batch_size']}, epochs={final_n_epochs}) ==="
    )

    model_weights = train_fashion_model(
        fashion_mnist,
        n_epochs=final_n_epochs,
        batch_size=best_cfg["batch_size"],
        learning_rate=best_cfg["learning_rate"],
        USE_GPU=USE_GPU,
        weight_decay=best_cfg["weight_decay"],
    )
    # Train model 
    # TODO: this may be done within a loop for hyperparameter search / cross-validation

    # Save model weights
    # However you tune and evaluate your model, make sure to save the final weights 
    # to submission/model_weights.pth before submission!
    model_save_path = os.path.join('submission', 'model_weights.pth')
    torch.save(model_weights, f=model_save_path)


if __name__ == "__main__":
    main()
