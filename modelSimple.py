import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class OWTTransformer(nn.Module):
    def __init__(self, input_dim=cfg.input_dim, output_dim=cfg.output_dim, d_model=128, num_heads=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Embedding layers
        self.embedding = nn.Linear(input_dim, d_model)

        # Transformer layers
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)

        self.intermediate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        # Output layers
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Input embedding
        x = self.embedding(x)

        # Reshape input
        x = x.permute(1, 0, 2)

        # Transformer layers
        x = self.transformer(x, x)

        # Reshape output
        x = x.permute(1, 0, 2)

        # Intermediate layers
        x = self.intermediate(x)

        # Output layers
        x = self.fc(x)

        return x
