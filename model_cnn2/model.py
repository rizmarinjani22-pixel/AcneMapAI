
import torch
import torch.nn as nn


# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

IMAGE_CHANNELS = 3
IMAGE_SIZE = 224

NUM_CLASSES = 4

CLASS_NAMES = [
    "mild",
    "moderate",
    "severe",
    "very_severe"
]


# ==========================================
# 2. CONVOLUTION BLOCK
# ==========================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            # Convolution
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            # Batch Normalization
            nn.BatchNorm2d(
                out_channels
            ),

            # ReLU Activation
            nn.ReLU(
                inplace=True
            ),

            # Max Pooling
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

        )

    def forward(self, x):

        return self.block(x)


# ==========================================
# 3. ACNE SEVERITY CNN
# ==========================================

class AcneLesionCNN(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES
    ):

        super().__init__()

        # ----------------------------------
        # STAGE 1-5:
        # INPUT AND FEATURE EXTRACTION
        # ----------------------------------

        self.features = nn.Sequential(

            # Block 1
            ConvBlock(
                3,
                32
            ),

            # Block 2
            ConvBlock(
                32,
                64
            ),

            # Block 3
            ConvBlock(
                64,
                128
            ),

            # Block 4
            ConvBlock(
                128,
                256
            )

        )

        # ----------------------------------
        # STAGE 6:
        # GLOBAL AVERAGE POOLING
        # ----------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # ----------------------------------
        # STAGE 7:
        # FULLY CONNECTED LAYER
        # ----------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.4
            )

        )

        # ----------------------------------
        # STAGE 8:
        # OUTPUT LAYER
        # ----------------------------------

        self.output_layer = nn.Linear(
            128,
            num_classes
        )

    # ======================================
    # FORWARD PROPAGATION
    # ======================================

    def forward(self, x):

        # Input and feature extraction
        x = self.features(x)

        # Global Average Pooling
        x = self.global_pool(x)

        # Fully Connected
        x = self.classifier(x)

        # Output logits
        x = self.output_layer(x)

        return x

    # ======================================
    # PREDICTION PROBABILITY
    # ======================================

    def predict_proba(self, x):

        logits = self.forward(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        return probabilities


# ==========================================
# 4. MODEL INFORMATION
# ==========================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ==========================================
# 5. MODEL TEST
# ==========================================

def main():

    print("=" * 50)

    print(
        "ACNEMAP CNN 2 - LESION MODEL"
    )

    print("=" * 50)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # Initialize model
    model = AcneLesionCNN(
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    print(
        "\nModel Architecture:"
    )

    print(model)

    print(
        "\nTotal Parameters:",
        f"{count_parameters(model):,}"
    )

    # Dummy input
    dummy_input = torch.randn(
        1,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(device)

    # Forward test
    model.eval()

    with torch.no_grad():

        output = model(dummy_input)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    print(
        "\n===== FORWARD TEST ====="
    )

    print(
        "Input Shape:",
        dummy_input.shape
    )

    print(
        "Output Shape:",
        output.shape
    )

    print(
        "\nPrediction Probabilities:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name}: "
            f"{probabilities[0][index].item():.4f}"
        )

    print(
        "\nCNN MODEL TEST COMPLETED"
    )


if __name__ == "__main__":

    main()
