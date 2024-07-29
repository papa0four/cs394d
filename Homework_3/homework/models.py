from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        # self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # self.fc2 = nn.Linear(128, num_classes)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        # logits = torch.randn(x.size(0), 6)
        # z = self.relu(self.conv1(z))
        # z = self.pool(z)
        # z = self.relu(self.conv2(z))
        # z = self.pool(z)
        # z = self.relu(self.conv3(z))
        # z = self.pool(z)

        # # Flatten
        # z = z.view(z.size(0), -1)

        # # Fully connected layers
        # z = self.relu(self.fc1(z))
        # logits = self.fc2(z)

        # return logits
        z = self.relu(self.bn1(self.conv1(z)))
        z = self.pool(z)
        z = self.relu(self.bn2(self.conv2(z)))
        z = self.pool(z)

        z = z.view(z.size(0), -1)
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        logits = self.fc2(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.enc1 = self.conv_block(in_channels, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 32)
        self.enc4 = self.conv_block(32, 64)

        self.bottleneck = self.conv_block(64, 128)

        self.dec_seg4 = self.upconv_block(128, 64)
        self.dec_seg3 = self.upconv_block(64, 32)
        self.dec_seg2 = self.upconv_block(32, 16)
        self.dec_seg1 = self.upconv_block(16, 8)
        self.seg_head = nn.Conv2d(8, num_classes, kernel_size=1)

        self.dec_depth4 = self.upconv_block(128, 64)
        self.dec_depth3 = self.upconv_block(64, 32)
        self.dec_depth2 = self.upconv_block(32, 16)
        self.dec_depth1 = self.upconv_block(16, 8)
        self.depth_head = nn.Conv2d(8, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels, kernel_size=2, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        e1 = self.enc1(z)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4_seg = self.dec_seg4(F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True))
        d4_seg = self.center_crop(d4_seg, e4.size()[2:])
        d3_seg = self.dec_seg3(F.interpolate(d4_seg + e4, scale_factor=2, mode='bilinear', align_corners=True))
        d3_seg = self.center_crop(d3_seg, e3.size()[2:])
        d2_seg = self.dec_seg2(F.interpolate(d3_seg + e3, scale_factor=2, mode='bilinear', align_corners=True))
        d2_seg = self.center_crop(d2_seg, e2.size()[2:])
        d1_seg = self.dec_seg1(F.interpolate(d2_seg + e2, scale_factor=2, mode='bilinear', align_corners=True))
        d1_seg = self.center_crop(d1_seg, e1.size()[2:])
        seg_logits = self.seg_head(F.interpolate(d1_seg + e1, size=(96, 128), mode='bilinear', align_corners=True))

        d4_depth = self.dec_depth4(F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True))
        d4_depth = self.center_crop(d4_depth, e4.size()[2:])
        d3_depth = self.dec_depth3(F.interpolate(d4_depth + e4, scale_factor=2, mode='bilinear', align_corners=True))
        d3_depth = self.center_crop(d3_depth, e3.size()[2:])
        d2_depth = self.dec_depth2(F.interpolate(d3_depth + e3, scale_factor=2, mode='bilinear', align_corners=True))
        d2_depth = self.center_crop(d2_depth, e2.size()[2:])
        d1_depth = self.dec_depth1(F.interpolate(d2_depth + e2, scale_factor=2, mode='bilinear', align_corners=True))
        d1_depth = self.center_crop(d1_depth, e1.size()[2:])
        depth = self.depth_head(F.interpolate(d1_depth + e1, size=(96, 128), mode='bilinear', align_corners=True))

        return seg_logits, depth.squeeze(1)
    
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self.forward(x)
        pred = logits.argmax(dim=1)
        depth = raw_depth
        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
