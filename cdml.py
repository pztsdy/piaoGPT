from sympy import true
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time
import math
import sys
import numpy as np
try:
    import torch_directml
except ImportError:
    print("未安装 torch_directml，无法使用 DirectML 设备。请安装 torch_directml 或使用 CUDA 设备。")
    torch_directml = None
from tqdm import trange

def get_device():
    try:
        dml_device = torch_directml.device()
        print("检测到 DirectML 设备")
        return dml_device
    except Exception as e:
        print(f"DirectML 检测失败: {e}")
    
    if torch.cuda.is_available():
        print("检测到 CUDA 设备")
        return torch.device("cuda")

    print("未检测到 CUDA / DirectML，使用 CPU")
    return torch.device("cpu")

DEVICE = get_device()

# ======== 基本配置 ========
MODEL_PATH = "piaoGPT.pth"
CORPUS_PATH = ["lang_pool.txt", "lang_pool_wiki.txt"]
TOKEN_CACHE_PATH = "data_tokens.npy"
EMBED_DIM = 384
NUM_HEADS = 4
NUM_LAYERS = 6
BLOCK_SIZE = 128
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 30
PRINT_EVERY = 1

# ======== 模型结构 ========
class piaoGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, block_size=BLOCK_SIZE):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        src_mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(idx.device)
        for block in self.blocks:
            x = block(x, src_mask=src_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ======== 语料加载 ========
def load_corpus(paths):
    text = ""
    for path in paths:
        print(f"正在加载语料库：{path}")
        with open(path, "r", encoding="utf-8") as f:
            text += f.read()
    print(f"语料库加载完成。")
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    print(f"词汇表大小: {len(vocab)}，语料字符数: {len(text)}")
    return text, vocab, stoi, itos

# ======== 采样生成 ========
def generate(model, idx, max_new_tokens, stoi, itos, temperature=1.0, top_k=0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, top_k)
            mask = logits < topk_vals[:, -1].unsqueeze(1)
            logits = logits.masked_fill(mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    out = ''.join([itos[i.item()] for i in idx[0]])
    return out

# ======== 训练批次获取 ========
def get_batch(data_tensor, batch_size, block_size):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# ======== 加载模型 ========
def load_state_if_exists(model, path):
    if os.path.exists(path):
        print(f"正在加载模型权重：{path}")
        try:
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state)
            print("模型加载成功！")
            return True
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            return False
    print("未找到模型权重，将从头开始训练或聊天。")
    return False

# ======== 时间 / 进度工具 ========
def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m}m{s}s"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"

def print_progress_bar(current, total, avg_epoch_time, eta_seconds, bar_length=30):
    frac = float(current) / total
    filled = int(round(bar_length * frac))
    bar = '=' * filled + '-' * (bar_length - filled)
    percent = frac * 100
    avg_str = format_time(avg_epoch_time)
    eta_str = format_time(eta_seconds)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({percent:5.1f}%) | avg_epoch={avg_str} | ETA={eta_str}")
    sys.stdout.flush()

# ======== 主程序 ========
if __name__ == "__main__":
    text, vocab, stoi, itos = load_corpus(CORPUS_PATH)
    vocab_size = len(vocab)

    # 尝试加载缓存
    if os.path.exists(TOKEN_CACHE_PATH):
        print(f"加载缓存 tokens: {TOKEN_CACHE_PATH}")
        data_tensor = torch.tensor(np.load(TOKEN_CACHE_PATH), dtype=torch.long)
    else:
        data_tensor = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        np.save(TOKEN_CACHE_PATH, data_tensor.cpu().numpy())
        print(f"保存 tokens 缓存至 {TOKEN_CACHE_PATH}")

    model = piaoGPT(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"当前设备: {DEVICE}")
    print(f"词汇表大小: {vocab_size}")
    print(f"语料库大小: {len(text)} 字符")

    while True:
        print("\n请选择操作：")
        print("1. 训练模型（继续训练）")
        print("2. 训练模型（从头训练，覆盖现有权重）")
        print("3. 直接聊天")
        print("4. 退出")
        choice = input("请输入选项 (1/2/3/4): ")

        if choice in ['1', '2']:
            if choice == '1':
                load_state_if_exists(model, MODEL_PATH)
            else:
                model = piaoGPT(vocab_size).to(DEVICE)
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

            try:
                epochs_to_train = int(input(f"请输入训练轮数 (默认 {EPOCHS}): "))
            except ValueError:
                epochs_to_train = EPOCHS

            epoch_times = []
            model.train()
            start_all = time.time()

            for epoch in trange(epochs_to_train, desc="训练进度", unit="epoch"):
                xb, yb = get_batch(data_tensor, BATCH_SIZE, BLOCK_SIZE)
                logits = model(xb)
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                yb = yb.view(B*T)
                loss = F.cross_entropy(logits, yb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_time = time.time() - start_all
            print(f"\n训练完成，总耗时: {format_time(total_time)}。保存模型至 {MODEL_PATH}")
            torch.save(model.state_dict(), MODEL_PATH)

        elif choice == '3':
            if not load_state_if_exists(model, MODEL_PATH):
                print("没有找到模型权重，请先训练模型。")
                continue
            print("\n进入聊天模式。输入 '/exit' 退出，'/temp <值>' 调整温度，'/topk <值>' 调整top_k。")
            temperature = 0.8
            top_k = 50
            print(f"当前生成参数: temperature={temperature}, top_k={top_k}")
            while True:
                user_input = input("你: ")
                if user_input.lower() == '/exit':
                    break
                elif user_input.lower().startswith('/temp '):
                    try:
                        temperature = float(user_input.split(' ')[1])
                    except:
                        print("无效的温度值")
                    continue
                elif user_input.lower().startswith('/topk '):
                    try:
                        top_k = int(user_input.split(' ')[1])
                    except:
                        print("无效的 top_k 值")
                    continue
                context = torch.tensor([[stoi.get(ch, 0) for ch in user_input]], dtype=torch.long).to(DEVICE)
                output = generate(model, context, max_new_tokens=100, stoi=stoi, itos=itos, temperature=temperature, top_k=top_k)
                reply = output[len(user_input):]
                print(f"piaoGPT: {reply}")

        elif choice == '4':
            print("退出程序。")
            break
        else:
            print("无效选项，请重新输入。")
