import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedWeightLearnable(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2048, 544),         # 34 * 4 * 4
            nn.BatchNorm1d(544),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        # 학습 가능한 패치 중요도 파라미터 (초기값: 1.0)
        self.alpha = nn.Parameter(torch.ones(34))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, 544)
        x = x.view(x.size(0), 34, 4, 4)  # (B, 34, 4, 4)
        x_mean = x.mean(dim=[2, 3])     # (B, 34)

        # 중요도 가중치 적용 및 0~1 정규화 (Sigmoid or Softmax 선택 가능)
        gated = torch.sigmoid(self.alpha)         # (34,)
        weighted = gated * x_mean                  # (B, 34)

        return weighted.sum(dim=1)                # (B,)

if __name__ == "__main__":
    input_tensor = torch.randn(32, 2048)
    model = GatedWeightLearnable()
    out = model(input_tensor)
    print("Output shape:", out.shape)
    print(out)
