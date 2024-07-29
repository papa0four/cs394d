import argparse
import sys
from datetime import datetime
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).resolve().parent))

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric

def train(exp_dir="logs", model_name="detector", num_epoch=10, lr=0.001, batch_size=32, seed=2024, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("CUDA not available, using CPU")

    torch.manual_seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed)

    log_dir = Path(exp_dir) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    model = Detector(**kwargs).to(device)
    criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_depth = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    metric = DetectionMetric()

    train_loader = load_data('road_data/train', transform_pipeline='aug', num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = load_data('road_data/val', transform_pipeline='default', num_workers=4, batch_size=batch_size, shuffle=False)

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        metric.reset()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != 'image'}

            optimizer.zero_grad()
            seg_outputs, depth_outputs = model(images)
            loss_seg = criterion_seg(seg_outputs, targets["track"])
            loss_depth = criterion_depth(depth_outputs.squeeze(1), targets["depth"])
            loss = loss_seg + loss_depth
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            metric.add(seg_outputs.argmax(dim=1), targets["track"], depth_outputs.squeeze(1), targets["depth"])

            logger.add_scalar('train_loss', loss.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / len(train_loader)
        epoch_metrics = metric.compute()
        print(f"Epoch [{epoch+1}/{num_epoch}], Training Loss: {epoch_loss:.4f}, IoU: {epoch_metrics['iou']:.4f}, Depth Error: {epoch_metrics['abs_depth_error']:.4f}")

        logger.add_scalar('epoch_train_iou', epoch_metrics['iou'], epoch)
        logger.add_scalar('epoch_train_depth_error', epoch_metrics['abs_depth_error'], epoch)

        model.eval()
        metric.reset()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != 'image'}
                seg_outputs, depth_outputs = model(images)
                loss_seg = criterion_seg(seg_outputs, targets["track"])
                loss_depth = criterion_depth(depth_outputs.squeeze(1), targets["depth"])
                val_loss += (loss_seg + loss_depth).item()

                metric.add(seg_outputs.argmax(dim=1), targets["track"], depth_outputs.squeeze(1), targets["depth"])

        val_loss /= len(val_loader)
        val_metrics = metric.compute()
        print(f"Validation Loss: {val_loss:.4f}, Validation IoU: {val_metrics['iou']:.4f}, Validation Depth Error: {val_metrics['abs_depth_error']:.4f}")

        logger.add_scalar('val_loss', val_loss, epoch)
        logger.add_scalar('val_iou', val_metrics['iou'], epoch)
        logger.add_scalar('val_depth_error', val_metrics['abs_depth_error'], epoch)

        scheduler.step(val_loss)

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

    train(**vars(parser.parse_args()))
