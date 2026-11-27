
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Blok konvolusi:
    Conv2D -> BatchNorm -> ReLU -> MaxPooling
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class AcneSeverityCNN(nn.Module):
    """
    CNN from scratch untuk klasifikasi severity jerawat
    berdasarkan gambar seluruh wajah.

    Kelas:
    0 = mild
    1 = moderate
    2 = severe
    3 = very_severe
    """

    def __init__(self, num_classes=4):
        super(AcneSeverityCNN, self).__init__()

        # Feature Extraction
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    # Pengujian arsitektur CNN
    model = AcneSeverityCNN(num_classes=4)

    # Simulasi satu batch berisi 8 gambar ukuran 224x224
    dummy_input = torch.randn(8, 3, 224, 224)

    output = model(dummy_input)

    print("===== TEST CNN ACNEMAP AI =====")
    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Jumlah parameter:",
          sum(p.numel() for p in model.parameters()))

    print("\nUrutan kelas:")
    print("0 = mild")
    print("1 = moderate")
    print("2 = severe")
    print("3 = very_severe")

    print("\nCNN berhasil dijalankan!")
