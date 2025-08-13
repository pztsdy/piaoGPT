import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time
import sys
import numpy as np
from tqdm import trange
try:
    import torch_directml
except ImportError:
    torch_directml = None
    print("未安装 torch_directml，使用 CUDA 或 CPU。")

# ======== 日志参数 ========
LOG_FILE = "lastest_train.log"  # 日志文件名
LOG_EVERY_K = 10                # 每隔多少个 epoch 记录示例问答
EXAMPLE_PROMPTS = [
    "什么是数学？",
    "马克思是谁？",
    "温度是什么？"
]  # 日志中使用的示例问答

# ======== 超参数 ========
MODEL_PATH = "piaoGPT.pth"
TOKEN_CACHE_PATH = "data_tokens.npy"
EMBED_DIM = 384
NUM_HEADS = 4
NUM_LAYERS = 6
BLOCK_SIZE = 128
BATCH_SIZE = 32
LR = 3e-4
EPOCHS_QA = 50        # 高质量问答训练轮数
EPOCHS_WIKI = 30      # Wikipedia 知识扩展训练轮数
MAX_NEW_TOKENS = 50   # 日志示例生成长度

# ======== 设备选择 ========
def get_device():
    if torch_directml:
        try:
            dml_device = torch_directml.device()
            print("检测到 DirectML 设备")
            return dml_device
        except:
            pass
    if torch.cuda.is_available():
        print("检测到 CUDA 设备")
        return torch.device("cuda")
    print("未检测到 CUDA / DirectML，使用 CPU")
    return torch.device("cpu")

DEVICE = get_device()

# ======== 模型结构 ========
class piaoGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, batch_first=True)
            for _ in range(NUM_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

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
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    print(f"词汇表大小: {len(vocab)}，语料字符数: {len(text)}")
    return text, vocab, stoi, itos

# ======== 采样生成 ========
def generate(model, idx, stoi, itos, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, top_k=0):
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
    return ''.join([itos[i.item()] for i in idx[0]])

# ======== 获取训练批次 ========
def get_batch(data_tensor, batch_size=BATCH_SIZE):
    ix = torch.randint(len(data_tensor) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data_tensor[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_tensor[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ======== 日志写入 ========
def write_log(epoch, loss_val, epoch_time, model, stoi, itos):
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if not log_exists:
            f.write("epoch,loss,time_seconds,example_outputs\n")
        examples = []
        for prompt in EXAMPLE_PROMPTS:
            context = torch.tensor([[stoi.get(ch,0) for ch in prompt]], dtype=torch.long).to(DEVICE)
            out = generate(model, context, stoi, itos)
            reply = out[len(prompt):]
            examples.append(f"{prompt}->{reply}")
        f.write(f"{epoch},{loss_val:.6f},{epoch_time:.2f},{examples}\n")

# ======== 加载模型 ========
def load_state_if_exists(model, path):
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state)
            print(f"模型 {path} 加载成功")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
    return False

# ======== 训练函数 ========
def train_model(corpus_paths, epochs, model=None, optimizer=None):
    text, vocab, stoi, itos = load_corpus(corpus_paths)
    vocab_size = len(vocab)

    if os.path.exists(TOKEN_CACHE_PATH):
        data_tensor = torch.tensor(np.load(TOKEN_CACHE_PATH), dtype=torch.long)
    else:
        data_tensor = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        np.save(TOKEN_CACHE_PATH, data_tensor.cpu().numpy())

    if model is None:
        model = piaoGPT(vocab_size).to(DEVICE)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    start_all = time.time()
    for epoch in trange(epochs, desc="训练进度", unit="epoch"):
        start_epoch = time.time()
        xb, yb = get_batch(data_tensor)
        logits = model(xb)
        B,T,C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        epoch_time = time.time() - start_epoch
        if (epoch+1) % LOG_EVERY_K == 0:
            write_log(epoch+1, loss.item(), epoch_time, model, stoi, itos)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"训练完成，总耗时: {time.time()-start_all:.2f}s，模型已保存至 {MODEL_PATH}")
    return model, optimizer, stoi, itos

# ======== 聊天函数 ========
def chat_loop(model, stoi, itos):
    temperature = 0.8
    top_k = 50
    print("进入聊天模式，输入 /exit-chat 返回主菜单")
    while True:
        user_input = input("你: ")
        if user_input.lower() == '/exit-chat':
            break
        elif user_input.lower().startswith('/temp '):
            try:
                temperature = float(user_input.split(' ')[1])
                print(f"temperature 设置为 {temperature}")
            except:
                print("无效的温度值")
            continue
        elif user_input.lower().startswith('/topk '):
            try:
                top_k = int(user_input.split(' ')[1])
                print(f"top_k 设置为 {top_k}")
            except:
                print("无效的 top_k 值")
            continue
        context = torch.tensor([[stoi.get(ch,0) for ch in user_input]], dtype=torch.long).to(DEVICE)
        output = generate(model, context, stoi, itos, temperature=temperature, top_k=top_k)
        reply = output[len(user_input):]
        print(f"piaoGPT: {reply}")

# ======== 主循环 ========
if __name__ == "__main__":
    # 初始化模型和优化器
    model = None
    optimizer = None
    stoi = itos = None

    while True:
        print("\n=== 主菜单 ===")
        print("1. 顺序训练模型（问答+Wikipedia）")
        print("2. 聊天")
        print("3. 退出")
        choice = input("请输入选项 (1/2/3): ")

        if choice == '1':
            # 第一阶段：高质量问答
            print("阶段1：训练高质量问答 (lang_pool.txt)")
            model, optimizer, stoi, itos = train_model(["lang_pool.txt"], epochs=EPOCHS_QA, model=model, optimizer=optimizer)
            # 第二阶段：Wikipedia
            print("阶段2：训练 Wikipedia 知识扩展 (lang_pool_wiki.txt)")
            model, optimizer, stoi, itos = train_model(["lang_pool_wiki.txt"], epochs=EPOCHS_WIKI, model=model, optimizer=optimizer)
            print("顺序训练完成 ✅")

        elif choice == '2':
            if model is None:
                model = piaoGPT(1).to(DEVICE)
                if not load_state_if_exists(model, MODEL_PATH):
                    print("未找到训练好的模型，请先顺序训练")
                    continue
            chat_loop(model, stoi, itos)

        elif choice == '3':
            print("退出程序")
            break
        else:
            print("无效选项，请重新输入")
