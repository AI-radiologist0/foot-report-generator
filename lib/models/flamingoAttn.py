from torch import nn

class FlamingoCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, image_tokens):
        # x: (B, T, D), text token embedding
        # image_tokens: (B, N_img, D)
        attn_out, _ = self.cross_attn(query=x, key=image_tokens, value=image_tokens)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x
