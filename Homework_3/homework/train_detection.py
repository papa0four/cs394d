import argparse
import sys
from datetime import datetime
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add homework directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 20,
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
    log_dir = Path(exp_dir) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    # Model, Loss, Optimizer
    model = Detector(**kwargs).to(device)
    criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_depth = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metric = DetectionMetric(num_classes=3)

    # Data Loading
    train_loader = load_data('road_data/train', transform_pipeline='default', num_workers=2, batch_size=batch_size, shuffle=True)
    val_loader = load_data('road_data/val', transform_pipeline='default', num_workers=2, batch_size=batch_size, shuffle=False)

    # Training Loop
    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        metric.reset()
        running_loss = 0.0
        for batch in train_loader:
            images, seg_targets, depth_targets = batch["image"], batch["track"], batch["depth"]
            images, seg_targets, depth_targets = images.to(device), seg_targets.to(device), depth_targets.to(device)

            optimizer.zero_grad()
            seg_output, depth_output = model(images)
            loss_seg = criterion_seg(seg_output, seg_targets)
            loss_depth = criterion_depth(depth_output, depth_targets)
            loss = loss_seg + loss_depth
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = seg_output.argmax(dim=1)
            metric.add(preds, seg_targets, depth_output, depth_targets)

            train_metrics = metric.compute()
            train_accuracy = train_metrics["accuracy"]
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
            for batch in val_loader:
                images, seg_targets, depth_targets = batch["image"], batch["track"], batch["depth"]
                images, seg_targets, depth_targets = images.to(device), seg_targets.to(device), depth_targets.to(device)
                seg_output, depth_output = model(images)
                loss_seg = criterion_seg(seg_output, seg_targets)
                loss_depth = criterion_depth(depth_output, depth_targets)
                loss = loss_seg + loss_depth
                val_loss += loss.item()

                preds = seg_output.argmax(dim=1)
                metric.add(preds, seg_targets, depth_output, depth_targets)

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
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)

    # Additional model parameters can be passed via kwargs
    # parser.add_argument("--in_channels", type=int, default=3)
    # parser.add_argument("--num_classes", type=int, default=3)

    # Pass all arguments to train
    train(**vars(parser.parse_args()))
