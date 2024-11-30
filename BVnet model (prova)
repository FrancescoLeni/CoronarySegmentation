import torch
import torch.nn as nn

class BVNet(nn.Module):
    def __init__(self, n_classes=1):
        super(BVNet, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output_layer = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)  # Downscale
        x2 = self.encoder2(x1)  # Downscale

        # Bottleneck
        x_b = self.bottleneck(x2)

        # Decoder
        x_d2 = self.decoder2(x_b) + x2  # Skip connection
        x_d1 = self.decoder1(x_d2) + x1  # Skip connection
        
        # Output
        out = self.output_layer(x_d1)
        return out


model = BVNet(n_classes=1)
model = model.to(device)


criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
train_model(model, train_loader, val_loader, epochs=50, optimizer=optimizer, criterion=criterion, device=device)
