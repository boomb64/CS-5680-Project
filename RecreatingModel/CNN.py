import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A residual block with 4 convolutional layers, as interpreted from the paper.
    This implementation uses two stacked residual units, each with 2 Conv layers.
    (Conv -> BN -> ReLU -> Conv -> BN) + Skip -> ReLU
    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        # First residual unit
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(in_channels)

        # Second residual unit
        self.conv2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(in_channels)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # First unit
        shortcut1 = x
        out = self.relu(self.bn1_1(self.conv1_1(x)))
        out = self.bn1_2(self.conv1_2(out))
        out += shortcut1
        out = self.relu(out)

        # Second unit
        shortcut2 = out
        out = self.relu(self.bn2_1(self.conv2_1(out)))
        out = self.bn2_2(self.conv2_2(out))
        out += shortcut2
        out = self.relu(out)

        return out


class FacialEmotionCNN(nn.Module):
    """
    PyTorch implementation of the CNN from the paper (Table 2).
    Input shape is assumed to be (batch_size, 1, 128, 96)
    """

    def __init__(self, in_channels=1, num_classes=6):
        super(FacialEmotionCNN, self).__init__()

        # Input: (1, 128, 96)
        self.conv1 = nn.Sequential(
            # 5x5 filter, stride 2. Padding=2 to get 64x48 output
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # Out: (32, 64, 48)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Out: (32, 32, 24)

        self.conv2 = nn.Sequential(
            # 3x3 filter, stride 1. Padding=1 to keep size
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # Out: (64, 32, 24)

        self.res1 = ResidualBlock(in_channels=64)  # Out: (64, 32, 24)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # Out: (128, 32, 24)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Out: (128, 16, 12)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # Out: (128, 16, 12)

        self.res2 = ResidualBlock(in_channels=128)  # Out: (128, 16, 12)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # Out: (256, 16, 12)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Out: (256, 8, 6)

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # Out: (512, 8, 6)

        self.flatten = nn.Flatten()  # Out: 512 * 8 * 6 = 24576

        # Calculate flattened size
        self.fc_input_features = 512 * 8 * 6

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Paper mentions dropout
        )  # Out: 1024

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Paper mentions dropout
        )  # Out: 512

        self.output_layer = nn.Linear(512, num_classes)  # Out: 6

        # Note on L2 Regularization (weight_decay):
        # The paper mentions regularization. In PyTorch, this is
        # typically added in the optimizer, not the model definition.
        # e.g., optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.conv6(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output_layer(x)

        # Note: A Softmax layer is not included here because
        # nn.CrossEntropyLoss() applies it automatically.
        return x


if __name__ == '__main__':
    # Define the input shape: (batch_size, channels, height, width)
    # Based on the paper, input is (1, 128, 96)
    input_shape = (1, 1, 128, 96)

    # Create the model
    model = FacialEmotionCNN()

    print("Model Architecture Created Successfully.\n")

    # Try to use torchinfo for a detailed summary (like Keras)
    # If not installed, just print the model structure
    try:
        from torchinfo import summary

        print("Model Summary (using torchinfo):")
        print("=" * 30)
        # We pass batch_size=1, which is common for summaries
        summary(model, input_size=input_shape)

    except ImportError:
        print("torchinfo not installed. Printing basic model structure.")
        print("For a detailed summary, run: pip install torchinfo")
        print("=" * 30)
        print(model)

    # Test with a dummy input
    print("\nTesting model with dummy input...")
    dummy_input = torch.randn(*input_shape)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Test successful.")