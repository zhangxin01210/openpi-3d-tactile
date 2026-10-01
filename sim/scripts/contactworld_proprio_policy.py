"""Small joint-state/time behavioral-cloning diagnostic for ContactWorld."""
import torch
from torch import nn


INPUT_DIM = 19  # 9 q, 9 dq, and elapsed control-step index / 100.
OUTPUT_DIM = 6


class ProprioPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.network(x)


def load_policy(path, device):
    bundle = torch.load(path, map_location=device, weights_only=True)
    model = ProprioPolicy().to(device)
    model.load_state_dict(bundle["model"])
    model.eval()
    return model, bundle


@torch.no_grad()
def act(model, bundle, q, dq, frame_index):
    device = next(model.parameters()).device
    x = torch.cat((q.reshape(-1), dq.reshape(-1),
                   torch.tensor([frame_index / 100], device=device, dtype=torch.float32)))
    normalized = (x - bundle["input_mean"]) / bundle["input_std"]
    action = model(normalized) * bundle["action_std"] + bundle["action_mean"]
    return torch.maximum(torch.minimum(action, bundle["action_max"]), bundle["action_min"])
