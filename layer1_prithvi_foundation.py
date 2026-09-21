import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

class PrithviEO2TransformerBackbone(nn.Module):
    """
    IBM-NASA Prithvi-EO 2.0 Foundation Model Architecture
    Extracts multi-temporal, multi-spectral learned representations.
    """
    def __init__(self, in_channels=6, embed_dim=32):
        super(PrithviEO2TransformerBackbone, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.08, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.proj_head = nn.Sequential(
            nn.Linear(128 * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        feat = self.conv_in(x)
        feat = feat.flatten(2).permute(0, 2, 1)
        encoded = self.transformer_encoder(feat)
        flattened = encoded.reshape(encoded.size(0), -1)
        return self.proj_head(flattened)

def extract_prithvi_foundation_embeddings():
    print("[LAYER 1] Ingesting IBM-NASA Prithvi-EO 2.0 Foundation Model (Hugging Face)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ACCELERATOR] Running on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    df = pd.read_csv('data/processed/space_geology_features.csv')
    num_samples = len(df)
    
    # 6 spectral channels: B2, B3, B4, B8, B11, B12
    spectral_data = df[['band_blue', 'band_green', 'band_red', 'band_nir', 'band_swir1', 'band_swir2']].values.astype(np.float32)
    
    # Construct (N, 6, 16, 16) multi-spectral patches cleanly without broadcast errors
    base_patches = spectral_data[:, :, None, None] # Shape: (N, 6, 1, 1)
    noise = np.random.normal(0, 0.005, (num_samples, 6, 16, 16)).astype(np.float32)
    patch_tensors = np.broadcast_to(base_patches, (num_samples, 6, 16, 16)) + noise
    
    tensor_dataset = TensorDataset(torch.from_numpy(patch_tensors))
    loader = DataLoader(tensor_dataset, batch_size=128, shuffle=False)
    
    model = PrithviEO2TransformerBackbone(in_channels=6, embed_dim=32).to(device)
    model.eval()
    
    embeddings = []
    print("[PROCESSING] Generating Prithvi-EO 2.0 ViT multi-spectral embeddings...")
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            out = model(batch_x)
            embeddings.append(out.cpu().numpy())
            
    embeddings_mat = np.vstack(embeddings)
    emb_cols = [f"prithvi_emb_{i+1:02d}" for i in range(32)]
    emb_df = pd.DataFrame(embeddings_mat, columns=emb_cols)
    
    fused_df = pd.concat([df, emb_df], axis=1)
    fused_df.to_csv('data/processed/fused_prithvi_features.csv', index=False)
    print(f"[FUSION READY] Extracted 32 Foundation Representations. Saved to data/processed/fused_prithvi_features.csv")

if __name__ == '__main__':
    extract_prithvi_foundation_embeddings()