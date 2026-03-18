import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # padding=1 ensures the spatial dimensions don't shrink during convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. Contracting Path (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 2. Expansive Path (Decoder)
        for feature in reversed(features):
            # Up-sampling layer
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            # Two convolutions after the up-sampling and concatenation
            self.ups.append(DoubleConv(feature*2, feature))

        # 3. Bottleneck (Bottom of the U)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # 4. Final Output Layer (Maps to desired number of classes)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Run through the Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # Save for later
            x = self.pool(x)

        # Run through the Bottleneck
        x = self.bottleneck(x)
        
        # Reverse the skip connections list to easily match with the Decoder
        skip_connections = skip_connections[::-1]

        # Run through the Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # Up-sample
            skip_connection = skip_connections[idx//2] # Grab the matching skip connection

            # Safety check: if shapes don't match due to max-pooling rounding issues, resize
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate along the channel dimension (dim=1)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # Run the double convolution
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# --- Test Block ---
if __name__ == "__main__":
    # Create a dummy tensor matching the shape from data_loader (Batch, Channels, Height, Width)
    x = torch.randn((4, 3, 512, 512))
    
    # Initialize the model
    model = UNet(in_channels=3, out_channels=1)
    
    # Pass the dummy data through the model
    predictions = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {predictions.shape}")
    
    assert predictions.shape == (4, 1, 512, 512), "Output shape does not match expected mask shape!"
    print("Success! The U-Net architecture is correctly assembled and processing tensors.")