import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    MLP with two hidden layers: 128 and 64 units.
    """

    def __init__(self, input_dim, n_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleTabNet(nn.Module):
    """
    Lightweight TabNet-style network with sequential feature masks.
    """

    def __init__(self, input_dim, n_classes, n_steps=3, hidden_dim=64, gamma=1.3):
        super().__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.gamma = gamma

        self.mask_layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_steps)]
        )
        self.decision_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(n_steps)
            ]
        )
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        prior = torch.ones_like(x)
        aggregated = 0.0

        for step in range(self.n_steps):
            mask_logits = self.mask_layers[step](x)
            mask = F.softmax(mask_logits * prior, dim=1)
            masked_x = x * mask
            decision = self.decision_layers[step](masked_x)
            aggregated = aggregated + decision
            prior = prior * (self.gamma - mask)

        return self.head(aggregated)


class FTTransformer(nn.Module):
    """
    Feature-token Transformer for tabular classification.
    """

    def __init__(
        self,
        input_dim,
        n_classes,
        d_token=32,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token

        self.feature_weight = nn.Parameter(torch.randn(input_dim, d_token) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_token))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_dim + 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, n_classes),
        )

    def _tokenize(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        return tokens

    def forward(self, x):
        batch_size = x.size(0)
        feature_tokens = self._tokenize(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, feature_tokens], dim=1)
        tokens = tokens + self.pos_embedding
        encoded = self.encoder(tokens)
        cls_rep = encoded[:, 0, :]
        return self.head(cls_rep)


def initialize_model(model_name, input_dim, n_classes):
    if model_name == "MLP":
        return SimpleMLP(input_dim=input_dim, n_classes=n_classes)
    if model_name == "TabNet":
        return SimpleTabNet(input_dim=input_dim, n_classes=n_classes)
    if model_name == "FTTransformer":
        return FTTransformer(input_dim=input_dim, n_classes=n_classes)
    raise ValueError(f"Unsupported deep model: {model_name}")
