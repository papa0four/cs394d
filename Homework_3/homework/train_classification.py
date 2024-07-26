import argparse
import sys
from datetime import datetime
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add homework directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric

def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 10,
    lr: float = 0.001,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"classifier_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    # Model, Loss, Optimizer
    model = Classifier(**kwargs).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metric = AccuracyMetric()

    # Data Loading
    train_loader = load_data('classification_data/train', transform_pipeline='default', num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = load_data('classification_data/val', transform_pipeline='default', num_workers=4, batch_size=batch_size, shuffle=False)

    # Training Loop
    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        metric.reset()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Convert logits to predicted class indices
            preds = torch.argmax(outputs, dim=1)
            metric.add(preds, labels)

            train_accuracy = metric.compute()["accuracy"]
            logger.add_scalar('train_loss', loss.item(), global_step)
            logger.add_scalar('train_accuracy', train_accuracy, global_step)
            global_step += 1

        epoch_loss = running_loss / len(train_loader)
        epoch_metric = metric.compute()["accuracy"]
        print(f"Epoch [{epoch+1}/{num_epoch}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_metric:.4f}")

        logger.add_scalar('epoch_train_accuracy', epoch_metric, epoch)

        # Validation Loop
        model.eval()
        metric.reset()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Convert logits to predicted class indices
                preds = torch.argmax(outputs, dim=1)
                metric.add(preds, labels)

        val_loss /= len(val_loader)
        val_metric = metric.compute()["accuracy"]
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_metric:.4f}")

        logger.add_scalar('val_loss', val_loss, epoch)
        logger.add_scalar('val_accuracy', val_metric, epoch)

    # Save the model
    save_model(model)
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)

    # Additional model parameters can be passed via kwargs
    # parser.add_argument("--in_channels", type=int, default=3)
    # parser.add_argument("--num_classes", type=int, default=6)

    # Pass all arguments to train
    train(**vars(parser.parse_args()))
