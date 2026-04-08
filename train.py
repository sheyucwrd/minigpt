import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. 数据 ==========
text = "hello world"

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)

# 输入和标签
x = data[:-1].unsqueeze(0)   # [1, T]
y = data[1:].unsqueeze(0)    # [1, T]

# ========== 2. Self-Attention ==========
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        Q = self.query(x)   # [B, T, C]
        K = self.key(x)     # [B, T, C]
        V = self.value(x)   # [B, T, C]

        # attention score: [B, T, T]
        scores = Q @ K.transpose(-2, -1) / (C ** 0.5)

        # causal mask: 不能看未来
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)   # [B, T, T]
        out = weights @ V                     # [B, T, C]

        return out

# ========== 3. MiniGPT ==========
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=32, max_len=32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.attn = SelfAttention(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        # idx: [B, T]
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)  # [B, T, C]
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos).unsqueeze(0)  # [1, T, C]

        x = tok_emb + pos_emb
        x = self.attn(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, vocab_size),
                targets.view(B * T)
            )

        return logits, loss

# ========== 4. 训练 ==========
model = MiniGPT(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(300):
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")

# ========== 5. 测试预测 ==========
model.eval()
with torch.no_grad():
    logits, _ = model(x)
    pred = torch.argmax(logits, dim=-1)

print("input : ", decode(x[0].tolist()))
print("target: ", decode(y[0].tolist()))
print("pred  : ", decode(pred[0].tolist()))

def generate(model, start, max_new_tokens=10):
    model.eval()
    
    idx = torch.tensor([encode(start)], dtype=torch.long)  # [1, T]

    for _ in range(max_new_tokens):
        logits, _ = model(idx)

        # 只取最后一个时间步
        logits = logits[:, -1, :]  # [1, vocab]
        probs = torch.softmax(logits, dim=-1)

        # 采样（比argmax更像GPT）
        next_id = torch.multinomial(probs, num_samples=1)  # [1,1]

        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())
print("generated:", generate(model, "h"))