# piaoGPT
一个基于PyTorch实现的轻量级Transformer模型，用于简单的文本生成和聊天。

## 🚀 项目简介
`piaoGPT` 是一个小型但功能完整的GPT模型示例，旨在帮助理解Transformer架构和基本的文本生成过程。它使用自定义的字符级词汇表，并从本地语料库 (`lang_pool.txt`) 中学习。

## ✨ 功能特性
*   **轻量级Transformer架构**: 采用PyTorch的`TransformerEncoder`构建。
*   **字符级模型**: 直接处理字符，无需复杂的词元化。
*   **灵活的训练模式**:
    *   从头开始训练新模型。
    *   加载现有模型 (`piaoGPT.pth`) 继续训练。
*   **交互式聊天模式**: 训练完成后或直接加载模型后，可以进入聊天界面与`piaoGPT`进行对话。
*   **自定义语料库**: 通过修改 `lang_pool.txt` 文件，可以轻松更换或扩展模型的学习内容。

## 🛠️ 安装与依赖
1.  **克隆仓库**:
    ```bash
    git clone https://github.com/pztsdy/piaoGPT.git
    cd piaoGPT
    ```
2.  **安装PyTorch**:
    请根据您的操作系统和CUDA版本，访问PyTorch官方网站获取安装命令：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    通常，您可以使用pip安装：
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # 示例，请根据实际情况选择
    ```
    或者如果您没有GPU：
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
3.  **语料库**:
    确保 `lang_pool.txt` 文件存在于项目根目录中，它将作为模型的训练语料。您可以编辑此文件来添加更多对话或文本。

## 🚀 使用方法

运行 `cdml.py` 脚本：

```bash
python cdml.py
```

脚本启动后，您将看到一个交互式菜单：

### 首次运行（无 `piaoGPT.pth` 文件）
```
未检测到模型 'piaoGPT.pth'。
请选择操作：
1. 从头开始训练 piaoGPT.pth
2. 退出
>
```
*   输入 `1` 并按回车，模型将开始从头训练。训练完成后会自动保存模型并进入聊天模式。
*   输入 `2` 并按回车，程序将退出。

### 已存在 `piaoGPT.pth` 文件
```
检测到已存在的模型 'piaoGPT.pth'。
请选择操作：
1. 从 piaoGPT.pth 继续训练
2. 从头开始训练 piaoGPT.pth
3. 直接用 piaoGPT.pth 开始聊天
>
```
*   输入 `1` 并按回车，程序将加载 `piaoGPT.pth` 模型并继续训练。训练完成后进入聊天模式。
*   输入 `2` 并按回车，程序将忽略 `piaoGPT.pth`，从头开始训练新模型。训练完成后进入聊天模式。
*   输入 `3` 并按回车，程序将加载 `piaoGPT.pth` 模型并直接进入聊天模式，跳过训练。

### 聊天模式
在聊天模式下，您可以输入任何文本，`piaoGPT` 将尝试生成回复。
*   输入 `exit` 并按回车，即可退出聊天模式并结束程序。

## 许可证
本项目采用 GNU General Public License v3.0 (GPLv3) 协议。
详情请参阅 [LICENSE](LICENSE) 文件。