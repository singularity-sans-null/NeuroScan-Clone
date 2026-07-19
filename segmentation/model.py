"""
model.py
--------
U-Net implementation for Brain Tumor Segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation import config
# import config

# DOUBLE CONVOLUTION BLOCK
class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.layers(x)


# Encoder Block
class Down(nn.Module):
    """
    Encoder Block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):

        super().__init__()

        self.double_conv = DoubleConv(
            in_channels,
            out_channels
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        skip = self.double_conv(x)

        pooled = self.pool(skip)

        return skip, pooled

# Decoder Block
class Up(nn.Module):
    """
    Decoder Block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.double_conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):
        """
        Parameters
        ----------
        x : Decoder feature map.

        skip : Corresponding encoder feature map.
        """

        # Upsample decoder features
        x = self.up(x)

        # Handle shape mismatch automatically


        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ]
        )

        # Concatenate along channel dimension
        x = torch.cat(
            [skip, x],
            dim=1
        )

        # Feature refinement
        x = self.double_conv(x)

        return x
    
# U-NET
class UNet(nn.Module):
    """
    U-Net Architecture

    Encoder
        ↓
    Bottleneck
        ↓
    Decoder
        ↓
    1×1 Convolution
    """
    def __init__(
        self,
        in_channels: int = config.IN_CHANNELS,
        out_channels: int = config.OUT_CHANNELS,
        features: list[int] = config.FEATURES
    ):
        super().__init__()

        self.features = features
        # Encoder
        self.encoder = nn.ModuleList()

        current_channels = in_channels

        for feature in features:

            self.encoder.append(

                Down(
                    current_channels,
                    feature
                )

            )

            current_channels = feature
        # Bottleneck
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2
        )
        # Decoder
        self.decoder = nn.ModuleList()

        decoder_channels = features[-1] * 2

        for feature in reversed(features):

            self.decoder.append(

                Up(
                    decoder_channels,
                    feature
                )

            )

            decoder_channels = feature
            
        # Final Prediction Layer

        self.final_conv = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )
        
    # Forward Pass
    
    def forward(self, x):

        skips = []
        

        for down in self.encoder:

            skip, x = down(x)

            skips.append(skip)


        x = self.bottleneck(x)
        
        skips = skips[::-1]

        for up, skip in zip(
            self.decoder,
            skips
        ):

            x = up(
                x,
                skip
            )


        x = self.final_conv(x)

        return x

# Create model

def create_model(
    in_channels: int = config.IN_CHANNELS,
    out_channels: int = config.OUT_CHANNELS
):
    """
    Creates a U-Net model.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    Returns
    -------
    UNet
    """

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels
    )

    return model

# WEIGHT INITIALIZATION


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using Kaiming (He) Initialization.

    Suitable for ReLU activation functions.
    """

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

            if module.bias is not None:

                nn.init.constant_(
                    module.bias,
                    0
                )

        elif isinstance(module, nn.BatchNorm2d):

            nn.init.constant_(
                module.weight,
                1
            )

            nn.init.constant_(
                module.bias,
                0
            )

        elif isinstance(module, nn.ConvTranspose2d):

            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

            if module.bias is not None:

                nn.init.constant_(
                    module.bias,
                    0
                )


# Parameter counter
def count_parameters(model: nn.Module):
    """
    Count trainable and total parameters.

    Returns
    -------
    tuple
        (trainable_parameters, total_parameters)
    """

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    return trainable, total

# Model summary
def print_model_summary(model: nn.Module):
    """
    Print model information.
    """

    trainable, total = count_parameters(model)

    print("=" * 60)
    print("U-Net Model Summary")
    print("=" * 60)

    print(f"Model                : {model.__class__.__name__}")
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")

    print("=" * 60)


# TEST the Model

if __name__ == "__main__":

    DEVICE = config.DEVICE

    print(f"\nUsing Device: {DEVICE}\n")

    model = create_model()

    initialize_weights(model)

    model.to(DEVICE)

    print_model_summary(model)

    # Dummy input tensor

    x = torch.randn(
        1,
        3,
        224,
        224
    ).to(DEVICE)

    with torch.no_grad():

        y = model(x)

    print("\nInput Shape")
    print(x.shape)

    print("\nOutput Shape")
    print(y.shape)

    print("\nExpected Output Shape")
    print("(1, 1, 224, 224)")

    print("\nModel test completed successfully.")