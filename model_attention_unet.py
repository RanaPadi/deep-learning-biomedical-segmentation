import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    Computes an attention coefficient to highlight relevant features and 
    suppress background noise.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        # W_g processes the gating signal (from the deeper layer)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # W_x processes the skip connection (from the encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Psi computes the final attention weights (alpha)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Handle spatial dimension mismatches due to max-pooling
        if x1.shape != g1.shape:
            x1 = TF.resize(x1, size=g1.shape[2:])
            
        # Combine, activate, and calculate attention weights
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        if psi.shape != x.shape:
            psi = TF.resize(psi, size=x.shape[2:])
            
        # Multiply original skip connection by the attention weights
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention_blocks = nn.ModuleList()

        # 1. Contracting Path (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 2. Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # 3. Expansive Path (Decoder)
        for feature in reversed(features):
            # Up-sampling layer
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            
            # Attention Gate setup
            self.attention_blocks.append(
                AttentionGate(F_g=feature, F_l=feature, F_int=feature//2)
            )
            
            # Double convolution after concatenation
            self.ups.append(DoubleConv(feature*2, feature))

        # Final Layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Run Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Run Bottleneck
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]

        # Run Decoder
        for idx in range(0, len(self.ups), 2):
            # Up-sample to get the gating signal
            gating_signal = self.ups[idx](x) 
            skip_connection = skip_connections[idx//2] 

            # Filter the skip connection through the Attention Gate
            att_skip = self.attention_blocks[idx//2](gating_signal, skip_connection)

            if gating_signal.shape != att_skip.shape:
                gating_signal = TF.resize(gating_signal, size=att_skip.shape[2:])

            # Concatenate the filtered skip connection with the up-sampled features
            concat_skip = torch.cat((att_skip, gating_signal), dim=1)
            
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# --- Test Block ---
if __name__ == "__main__":
    x = torch.randn((4, 3, 512, 512))
    model = AttentionUNet(in_channels=3, out_channels=1)
    predictions = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {predictions.shape}")
    assert predictions.shape == (4, 1, 512, 512), "Shapes do not match!"
    print("Success! Attention U-Net is built correctly.")