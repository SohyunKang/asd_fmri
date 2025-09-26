import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, hidden_dim=128, dropout=0.2):
        super(MLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_model(config, input_dim, output_dim):
    """
    Args:
        config (dict): config["model"]
        input_dim (int): feature 개수
        output_dim (int): 클래스 개수
    """
    name = config["name"].lower()

    if name == "mlp":
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=config.get("num_layers", 3),
            hidden_dim=config.get("hidden_dim", 128),
            dropout=config.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
