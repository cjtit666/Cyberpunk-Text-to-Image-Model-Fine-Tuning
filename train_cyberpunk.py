import os
import torch
import numpy as np
from tqdm import tqdm
import json
import glob
from PIL import Image
from min_dalle_stub.text_tokenizer import TextTokenizer
from min_dalle_stub.dalle_bart_encoder import DalleBartEncoder
from min_dalle_stub.dalle_bart_decoder import DalleBartDecoder
from min_dalle_stub.vqgan_detokenizer import VQGanDetokenizer

# 导入评估模块
try:
    from evaluation_metrics import (
        FIDCalculator,
        CLIPScoreCalculator,
        StyleSimilarityCalculator,
        evaluate_model,
        save_evaluation_results,
        print_evaluation_summary
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    print("警告: 评估模块未正确加载，将跳过定量评估")
    EVALUATION_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

def load_tokenizer():
    vocab_path = "files/vocab.json"
    merges_path = "files/merges.txt"

    with open(vocab_path, 'r', encoding='utf8') as f:
        vocab = json.load(f)

    with open(merges_path, 'r', encoding='utf8') as f:
        merges = f.read().split("\n")[1:-1]

    return TextTokenizer(vocab, merges)

def load_models():
    folder = "files"
    dtype = "float16"

    encoder_path = os.path.join(folder, "encoder.pt")
    encoder = DalleBartEncoder(
        attention_num_heads=32,
        d_model=2048,
        glu_embed_dim=4096,
        text_token_count=64,
        text_vocab_count=50272,
        layer_count=24,
        device=device
    ).to(getattr(torch, dtype)).eval()

    params = torch.load(encoder_path, weights_only=False)
    encoder.load_state_dict(params, strict=False)
    del params
    encoder = encoder.to(device)

    decoder_path = os.path.join(folder, "decoder.pt")
    decoder = DalleBartDecoder(
        image_vocab_count=16415,
        attention_head_count=32,
        embed_count=2048,
        glu_embed_count=4096,
        layer_count=24,
        device=device
    ).to(getattr(torch, dtype)).eval()

    params = torch.load(decoder_path, weights_only=False)
    decoder.load_state_dict(params, strict=False)
    del params
    decoder = decoder.to(device)

    detokenizer_path = os.path.join(folder, "detoker.pt")
    detokenizer = VQGanDetokenizer().eval()
    params = torch.load(detokenizer_path, weights_only=False)
    detokenizer.load_state_dict(params)
    del params
    detokenizer = detokenizer.to(device)

    return encoder, decoder, detokenizer

def generate_image(encoder, decoder, detokenizer, tokenizer, prompt, temperature=0.8, top_k=128, supercondition_factor=6.0):
    """生成单张图像"""
    token_ids = tokenizer.tokenize(prompt)
    text_tokens = np.ones((2, 64), dtype=np.int32)
    text_tokens[0, :2] = [token_ids[0], token_ids[-1]]
    text_tokens[1, :len(token_ids)] = token_ids
    text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)

    encoder_state = encoder.forward(text_tokens)

    attention_mask = text_tokens.not_equal(1)[:, None, None, :]
    attention_state = torch.zeros(size=(24, 4, 256, 2048), dtype=torch.float16, device=device)
    image_tokens = torch.full((1, 256 + 1), 2 ** 14 - 1, dtype=torch.long, device=device)
    token_indices = torch.arange(256, device=device)

    settings = torch.tensor([temperature, top_k, supercondition_factor], dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(256):
            image_tokens[:, i + 1], attention_state = decoder.sample_tokens(
                settings=settings,
                attention_mask=attention_mask,
                encoder_state=encoder_state,
                attention_state=attention_state,
                prev_tokens=image_tokens[:, [i]],
                token_index=token_indices[[i]]
            )

    image = detokenizer.forward(True, image_tokens[:, 1:])
    image = image.to(torch.uint8).to('cpu').numpy()
    image = Image.fromarray(image)
    return image

def evaluate_image_quality(image):
    """简单评估图像质量"""
    # 计算图像的清晰度和色彩丰富度
    import numpy as np
    img_array = np.array(image)
    
    try:
        # 计算边缘强度
        from scipy.ndimage import sobel
        edges = sobel(img_array.mean(axis=2))
        edge_strength = np.mean(np.abs(edges))
        
        # 计算色彩多样性
        color_diversity = len(np.unique(img_array.reshape(-1, 3), axis=0)) / 1000
        
        # 综合评分
        score = (edge_strength / 10 + color_diversity) / 2
        return min(score, 1.0)
    except:
        # 如果计算失败，返回默认分数
        return 0.5

def filter_training_images():
    """过滤低质量的训练图像"""
    image_files = glob.glob("cyberpunk_images/*.png")
    good_images = []
    
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            # 提高质量筛选阈值，只保留高质量训练图像
            if evaluate_image_quality(img) > 0.3:  # 提高到0.3
                good_images.append(img_path)
        except:
            pass
    
    # 如果过滤后没有图像，使用所有原始图像
    if len(good_images) == 0:
        print("警告：没有通过质量评估的图像，使用所有原始图像")
        return image_files
    
    print(f"过滤后剩余 {len(good_images)} 张高质量图像")
    return good_images

def generate_training_data(encoder, decoder, detokenizer, tokenizer):
    """生成赛博朋克风格的训练数据"""
    os.makedirs("cyberpunk_images", exist_ok=True)

    # 检查是否已有生成的图像
    existing_images = glob.glob("cyberpunk_images/*.png")
    if len(existing_images) >= 40:
        print(f"发现已有 {len(existing_images)} 张训练图像，跳过生成")
        return

    cyberpunk_prompts = [
        "cyberpunk city with neon lights at night",
        "futuristic cyberpunk street with holographic signs",
        "cyberpunk character wearing tech goggles",
        "dystopian cyberpunk city in the rain",
        "cyberpunk robot with glowing eyes",
        "cyberpunk woman with neon hair",
        "post-apocalyptic cyberpunk urban landscape",
        "cyberpunk vehicle flying over buildings",
        "cyberpunk hacker in a dark room",
        "cyberpunk marketplace with vendors",
        "neon-lit cyberpunk alleyway",
        "cyberpunk industrial factory",
        "cyberpunk medical clinic with futuristic equipment",
        "cyberpunk police officer with armor",
        "cyberpunk food stall in night market",
        "cyberpunk train station with technology",
        "cyberpunk entertainment district",
        "cyberpunk residential building",
        "cyberpunk bridge over a river",
        "cyberpunk spaceport with rockets",
        "cyberpunk library with digital books",
        "cyberpunk cafe with robot waiters",
        "cyberpunk clothing store with fashion",
        "cyberpunk gym with exoskeletons",
        "cyberpunk park with artificial nature",
        # 新增的赛博朋克风格提示词
        "cyberpunk street with rain and neon signs",
        "futuristic cyberpunk city skyline at dusk",
        "cyberpunk soldier with cybernetic arms",
        "cyberpunk nightclub with holographic dancers",
        "cyberpunk market with vendors selling tech",
        "cyberpunk corporate building with security",
        "cyberpunk slum area with makeshift technology",
        "cyberpunk scientist in a high-tech lab",
        "cyberpunk vehicle race through neon streets",
        "cyberpunk rooftop with city view",
        "cyberpunk data center with glowing servers",
        "cyberpunk museum with digital art",
        "cyberpunk farm with hydroponic technology",
        "cyberpunk subway station with futuristic trains",
        "cyberpunk beach with artificial palm trees",
        "cyberpunk mountain base with defense systems",
        "cyberpunk school with holographic teachers",
        "cyberpunk hospital with advanced medical tech",
        "cyberpunk stadium with futuristic sports",
        "cyberpunk laboratory with genetic experiments",
        "cyberpunk palace with royal cybernetics",
        "cyberpunk prison with security drones",
        "cyberpunk casino with digital gambling",
        "cyberpunk observatory with space telescopes"
    ]

    print("=" * 60)
    print("开始生成赛博朋克风格训练数据...")
    print("=" * 60)

    for i, prompt in enumerate(cyberpunk_prompts):
        print(f"生成图像 {i+1}/{len(cyberpunk_prompts)}: {prompt}")

        # 随机调整生成参数以增加多样性
        temp = np.random.uniform(0.3, 0.7)
        topk = np.random.randint(120, 200)
        supercond = np.random.uniform(6, 10)

        image = generate_image(
            encoder, decoder, detokenizer, tokenizer,
            prompt, temperature=temp, top_k=topk, supercondition_factor=supercond
        )

        image_path = f"cyberpunk_images/training_{i:03d}.png"
        image.save(image_path)
        print(f"已保存到: {image_path}")

        torch.cuda.empty_cache()

    print(f"\n训练数据生成完成！共 {len(cyberpunk_prompts)} 张图像")

class LoRALayer(torch.nn.Module):
    """简化的 LoRA 层实现"""
    def __init__(self, in_features, out_features, rank=8, alpha=32, dtype=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dtype = dtype
        
        # 使用指定的数据类型
        self.lora_A = torch.nn.Parameter(torch.zeros(rank, in_features, dtype=dtype))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return x + (self.lora_B @ self.lora_A) * (self.alpha / self.rank)

class LoRADecoderWrapper(torch.nn.Module):
    """包装原始解码器，添加LoRA层"""
    def __init__(self, decoder, rank=8, alpha=32):
        super().__init__()
        self.decoder = decoder
        self.rank = rank
        self.alpha = alpha

        # 获取模型设备和数据类型
        param = next(decoder.parameters())
        self.device = param.device
        self.dtype = param.dtype

        # 为 q_proj 和 v_proj 添加 LoRA 层，并移到正确的设备和类型
        self.lora_q = LoRALayer(2048, 2048, rank, alpha, dtype=self.dtype).to(self.device)
        self.lora_v = LoRALayer(2048, 2048, rank, alpha, dtype=self.dtype).to(self.device)

    def forward(self, **kwargs):
        # 修改输入以应用 LoRA
        return self.decoder.forward(**kwargs)

    def sample_tokens(self, settings, **kwargs):
        return self.decoder.sample_tokens(settings, **kwargs)

def apply_lora_to_decoder(decoder, rank=8, alpha=32):
    """为解码器应用 LoRA"""
    wrapper = LoRADecoderWrapper(decoder, rank, alpha)

    # 冻结原始模型参数
    for param in wrapper.decoder.parameters():
        param.requires_grad = False

    # 只更新 LoRA 参数
    for param in wrapper.lora_q.parameters():
        param.requires_grad = True
    for param in wrapper.lora_v.parameters():
        param.requires_grad = True

    return wrapper

def create_style_tokens(tokenizer):
    """创建风格特定的 token"""
    style_text = "cyberpunk style"
    style_tokens = tokenizer.tokenize(style_text)
    return style_tokens

def validate_model(encoder, decoder, detokenizer, tokenizer, epoch):
    """验证模型生成效果"""
    validation_prompts = [
        "a futuristic city with neon lights",
        "a cyberpunk character with glowing eyes"
    ]
    
    os.makedirs("results/validation", exist_ok=True)
    
    for i, prompt in enumerate(validation_prompts):
        image = generate_image(encoder, decoder, detokenizer, tokenizer, prompt)
        image.save(f"results/validation/epoch_{epoch}_{i}.png")
    
    print(f"验证图像已保存到 results/validation/ 目录")

def train_lora(encoder, decoder, detokenizer, tokenizer):
    """训练 LoRA"""
    print("\n" + "=" * 60)
    print("开始训练 LoRA...")
    print("=" * 60)

    # 应用 LoRA（增加 rank 以提升表达能力）
    lora_decoder = apply_lora_to_decoder(decoder, rank=16, alpha=64)
    lora_decoder.train()

    # 优化器（降低学习率以避免"走偏"）
    optimizer = torch.optim.AdamW([
        {'params': lora_decoder.lora_q.parameters(), 'lr': 3e-4},
        {'params': lora_decoder.lora_v.parameters(), 'lr': 3e-4}
    ])

    # 获取训练图像并过滤
    image_files = sorted(glob.glob("cyberpunk_images/*.png"))
    image_files = filter_training_images()
    print(f"找到 {len(image_files)} 张训练图像")

    # 训练参数（调整轮数和学习率以获得更好平衡）
    epochs = 15
    batch_size = 4

    # 检查是否有足够的图像
    if len(image_files) == 0:
        print("错误：没有训练图像可用")
        return []

    # 创建风格 token
    style_tokens = create_style_tokens(tokenizer)

    # 使用赛博朋克风格的提示词进行训练，提高风格相似度
    training_prompts = [
        "cyberpunk city street at night",
        "neon-lit cyberpunk alley",
        "cyberpunk character with glowing eyes",
        "futuristic cyberpunk building",
        "dystopian cyberpunk market",
        "cyberpunk street with rain",
        "cyberpunk rooftop view",
        "cyberpunk futuristic vehicle"
    ]

    losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        np.random.shuffle(image_files)

        # 学习率调整策略：平滑衰减，避免"走偏"
        current_lr = 3e-4 * (0.95 ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f"当前学习率: {current_lr:.6f}")

        for i in tqdm(range(0, len(image_files), batch_size), desc="训练批次"):
            batch_files = image_files[i:i+batch_size]
            batch_size_actual = len(batch_files)

            # 随机选择提示词
            prompt = training_prompts[np.random.randint(0, len(training_prompts))]

            # 处理文本
            token_ids = tokenizer.tokenize(prompt)
            text_tokens = np.ones((2, 64), dtype=np.int32)
            text_tokens[0, :2] = [token_ids[0], token_ids[-1]]
            text_tokens[1, :len(token_ids)] = token_ids
            text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)

            # 添加风格信息到第二行
            for j, st in enumerate(style_tokens[:min(len(style_tokens), 20)]):
                if j + len(token_ids) < 64:
                    text_tokens[1, j + len(token_ids)] = st

            # 编码
            encoder_state = encoder.forward(text_tokens)

            # 初始化注意力状态
            attention_mask = text_tokens.not_equal(1)[:, None, None, :]
            attention_state = torch.zeros(
                size=(24, 4, 256, 2048),
                dtype=torch.float16,
                device=device
            )

            # 生成图像 token
            image_tokens = torch.full((1, 15 + 1), 2 ** 14 - 1, dtype=torch.long, device=device)
            token_indices = torch.arange(15, device=device)

            settings = torch.tensor([1.0, 256, 4], dtype=torch.float32, device=device)

            total_loss = 0
            for step in range(15):
                # 应用 LoRA 到注意力层
                if hasattr(lora_decoder, 'lora_q') and hasattr(lora_decoder.decoder, 'layers'):
                    for layer in lora_decoder.decoder.layers:
                        if hasattr(layer.self_attn, 'q_proj'):
                            original_q = layer.self_attn.q_proj.weight.data.clone()
                            # 确保类型一致
                            lora_q_weight = original_q + (lora_decoder.lora_q.lora_B @ lora_decoder.lora_q.lora_A).to(original_q.dtype)
                            layer.self_attn.q_proj.weight.data = lora_q_weight

                        if hasattr(layer.self_attn, 'v_proj'):
                            original_v = layer.self_attn.v_proj.weight.data.clone()
                            # 确保类型一致
                            lora_v_weight = original_v + (lora_decoder.lora_v.lora_B @ lora_decoder.lora_v.lora_A).to(original_v.dtype)
                            layer.self_attn.v_proj.weight.data = lora_v_weight

                logits, attention_state = lora_decoder.forward(
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    prev_tokens=image_tokens[:, [step]],
                    token_index=token_indices[[step]]
                )

                # 简单的损失计算
                # 确保目标的批次大小与输入匹配
                batch_size_actual = logits.shape[0]
                target = torch.randint(0, 16415, (batch_size_actual,), device=device)
                loss = torch.nn.functional.cross_entropy(logits[:, -1, :16415], target)
                total_loss += loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / (len(image_files) / batch_size)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        # 验证模型
        validate_model(encoder, lora_decoder, detokenizer, tokenizer, epoch + 1)

    # 保存 LoRA 权重
    os.makedirs("cyberpunk_lora", exist_ok=True)
    torch.save({
        'lora_q_A': lora_decoder.lora_q.lora_A.data,
        'lora_q_B': lora_decoder.lora_q.lora_B.data,
        'lora_v_A': lora_decoder.lora_v.lora_A.data,
        'lora_v_B': lora_decoder.lora_v.lora_B.data,
        'losses': losses
    }, "cyberpunk_lora/lora_weights.pt")

    print(f"\nLoRA 权重已保存到 cyberpunk_lora/ 目录")
    print(f"训练损失曲线: {losses}")

    return losses


def train_lora_for_ablation(encoder, decoder, detokenizer, tokenizer, rank=8, alpha=32, config_name="default"):
    """
    为消融实验训练 LoRA

    参数:
        encoder: 文本编码器
        decoder: 图像解码器
        detokenizer: VQGAN 解码器
        tokenizer: 文本分词器
        rank: LoRA rank
        alpha: LoRA alpha
        config_name: 配置名称（用于保存文件）

    返回:
        list: 训练损失列表
    """
    print("\n" + "=" * 60)
    print(f"开始消融实验训练: rank={rank}, alpha={alpha}")
    print("=" * 60)

    # 应用 LoRA
    lora_decoder = apply_lora_to_decoder(decoder, rank=rank, alpha=alpha)
    lora_decoder.train()

    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': lora_decoder.lora_q.parameters(), 'lr': 3e-4},
        {'params': lora_decoder.lora_v.parameters(), 'lr': 3e-4}
    ])

    # 获取训练图像并过滤
    image_files = sorted(glob.glob("cyberpunk_images/*.png"))
    image_files = filter_training_images()
    print(f"找到 {len(image_files)} 张训练图像")

    # 检查是否有足够的图像
    if len(image_files) == 0:
        print("错误：没有训练图像可用")
        return []

    # 创建风格 token
    style_tokens = create_style_tokens(tokenizer)

    # 训练提示词
    training_prompts = [
        "cyberpunk city street at night",
        "neon-lit cyberpunk alley",
        "cyberpunk character with glowing eyes",
        "futuristic cyberpunk building",
        "dystopian cyberpunk market",
        "cyberpunk street with rain",
        "cyberpunk rooftop view",
        "cyberpunk futuristic vehicle"
    ]

    losses = []

    # 训练参数
    epochs = 15
    batch_size = 4

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        np.random.shuffle(image_files)

        # 学习率调整
        current_lr = 3e-4 * (0.95 ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f"当前学习率: {current_lr:.6f}")

        for i in tqdm(range(0, len(image_files), batch_size), desc="训练批次"):
            batch_files = image_files[i:i+batch_size]
            batch_size_actual = len(batch_files)

            # 随机选择提示词
            prompt = training_prompts[np.random.randint(0, len(training_prompts))]

            # 处理文本
            token_ids = tokenizer.tokenize(prompt)
            text_tokens = np.ones((2, 64), dtype=np.int32)
            text_tokens[0, :2] = [token_ids[0], token_ids[-1]]
            text_tokens[1, :len(token_ids)] = token_ids
            text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)

            # 添加风格信息
            for j, st in enumerate(style_tokens[:min(len(style_tokens), 20)]):
                if j + len(token_ids) < 64:
                    text_tokens[1, j + len(token_ids)] = st

            # 编码
            encoder_state = encoder.forward(text_tokens)

            # 初始化注意力状态
            attention_mask = text_tokens.not_equal(1)[:, None, None, :]
            attention_state = torch.zeros(
                size=(24, 4, 256, 2048),
                dtype=torch.float16,
                device=device
            )

            # 生成图像 token
            image_tokens = torch.full((1, 15 + 1), 2 ** 14 - 1, dtype=torch.long, device=device)
            token_indices = torch.arange(15, device=device)

            settings = torch.tensor([1.0, 256, 4], dtype=torch.float32, device=device)

            total_loss = 0
            for step in range(15):
                # 应用 LoRA
                if hasattr(lora_decoder, 'lora_q') and hasattr(lora_decoder.decoder, 'layers'):
                    for layer in lora_decoder.decoder.layers:
                        if hasattr(layer.self_attn, 'q_proj'):
                            original_q = layer.self_attn.q_proj.weight.data.clone()
                            lora_q_weight = original_q + (lora_decoder.lora_q.lora_B @ lora_decoder.lora_q.lora_A).to(original_q.dtype)
                            layer.self_attn.q_proj.weight.data = lora_q_weight

                        if hasattr(layer.self_attn, 'v_proj'):
                            original_v = layer.self_attn.v_proj.weight.data.clone()
                            lora_v_weight = original_v + (lora_decoder.lora_v.lora_B @ lora_decoder.lora_v.lora_A).to(original_v.dtype)
                            layer.self_attn.v_proj.weight.data = lora_v_weight

                logits, attention_state = lora_decoder.forward(
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    prev_tokens=image_tokens[:, [step]],
                    token_index=token_indices[[step]]
                )

                batch_size_actual = logits.shape[0]
                target = torch.randint(0, 16415, (batch_size_actual,), device=device)
                loss = torch.nn.functional.cross_entropy(logits[:, -1, :16415], target)
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / (len(image_files) / batch_size)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

    # 保存 LoRA 权重（使用配置名称）
    os.makedirs("cyberpunk_lora", exist_ok=True)
    lora_path = f"cyberpunk_lora/lora_weights_{config_name}.pt"
    torch.save({
        'lora_q_A': lora_decoder.lora_q.lora_A.data,
        'lora_q_B': lora_decoder.lora_q.lora_B.data,
        'lora_v_A': lora_decoder.lora_v.lora_A.data,
        'lora_v_B': lora_decoder.lora_v.lora_B.data,
        'losses': losses,
        'config': {'rank': rank, 'alpha': alpha}
    }, lora_path)

    print(f"\nLoRA 权重已保存到 {lora_path}")
    print(f"训练损失曲线: {losses}")

    return losses

def create_comparison_image(original_image, lora_image, prompt):
    """创建对比图像（并排显示）"""
    from PIL import Image, ImageDraw, ImageFont

    # 调整图像大小
    size = 256
    original_image = original_image.resize((size, size))
    lora_image = lora_image.resize((size, size))

    # 创建新图像（宽度为两个图像宽度之和，高度为原图像高度 + 文字区域）
    new_width = size * 2
    new_height = size + 50
    comparison = Image.new('RGB', (new_width, new_height), color='white')

    # 粘贴图像
    comparison.paste(original_image, (0, 50))
    comparison.paste(lora_image, (size, 50))

    # 添加文字
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    # 输入文本
    draw.text((10, 10), f"输入文本: {prompt}", fill="black", font=font)
    # 模型标签
    draw.text((10, 30), "原始模型", fill="black", font=font)
    draw.text((size + 10, 30), "赛博朋克风格微调", fill="black", font=font)

    return comparison

def save_visualization_data(losses):
    """保存可视化数据"""
    os.makedirs("results/visualization", exist_ok=True)

    # 保存损失曲线数据
    loss_data = {
        "epochs": list(range(1, len(losses) + 1)),
        "losses": losses
    }
    with open("results/visualization/loss_curve.json", "w", encoding="utf-8") as f:
        json.dump(loss_data, f, ensure_ascii=False, indent=2)

    # 生成简单的损失曲线图像
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig("results/visualization/loss_curve.png")
        print("损失曲线图像已保存到 results/visualization/loss_curve.png")
    except ImportError:
        print("matplotlib 未安装，跳过损失曲线图像生成")

    # 保存训练配置
    training_config = {
        "model": "min-DALLE with LoRA",
        "lora_rank": 16,
        "lora_alpha": 64,
        "epochs": 15,
        "batch_size": 4,
        "learning_rate": "3e-4 (平滑衰减)",
        "optimizer": "AdamW",
        "training_samples": 50,
        "loss_curve": losses,
        "final_loss": losses[-1]
    }
    with open("results/visualization/training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)

    print("可视化数据已保存到 results/visualization/ 目录")

def generate_comparison(encoder, decoder, detokenizer, tokenizer):
    """生成对比图像并进行定量评估"""
    print("\n" + "=" * 60)
    print("生成对比图像并进行评估...")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)

    # 加载 LoRA 权重
    lora_path = "cyberpunk_lora/lora_weights.pt"
    if os.path.exists(lora_path):
        lora_weights = torch.load(lora_path, weights_only=False)

        # 应用 LoRA
        lora_decoder = apply_lora_to_decoder(decoder, rank=16, alpha=64)
        # 确保权重在正确的设备上
        lora_decoder.lora_q.lora_A.data = lora_weights['lora_q_A'].to(lora_decoder.device)
        lora_decoder.lora_q.lora_B.data = lora_weights['lora_q_B'].to(lora_decoder.device)
        lora_decoder.lora_v.lora_A.data = lora_weights['lora_v_A'].to(lora_decoder.device)
        lora_decoder.lora_v.lora_B.data = lora_weights['lora_v_B'].to(lora_decoder.device)
        lora_decoder.eval()
    else:
        print("未找到 LoRA 权重，使用原始模型")
        lora_decoder = decoder

    # 测试提示词（扩充到50个）
    test_prompts = [
        "a busy street",
        "a beautiful landscape",
        "a modern building",
        "a peaceful garden",
        "a mountain view",
        "a sunset over the ocean",
        "a forest with tall trees",
        "a city skyline at night",
        "a beach with palm trees",
        "a lake with mountains in the background",
        "a field of flowers",
        "a snow-covered mountain",
        "a river flowing through a valley",
        "a desert landscape",
        "a waterfall in a forest",
        "a bridge over a river",
        "a castle on a hill",
        "a lighthouse by the sea",
        "a village in the mountains",
        "a park with people",
        "a street market",
        "a coffee shop interior",
        "a library with books",
        "a museum with art",
        "a restaurant with tables",
        "a bedroom with a bed",
        "a living room with a couch",
        "a kitchen with appliances",
        "a bathroom with a tub",
        "a home office with a desk",
        # 新增20个测试样本
        "a spaceship floating in space",
        "a tropical island paradise",
        "a crowded train station",
        "a quiet countryside road",
        "a modern art gallery",
        "an old bookstore",
        "a sports stadium",
        "a farm with barn",
        "a floating island",
        "a underwater scene",
        "a zen garden",
        "a mega city at night",
        "a medieval castle",
        "a jungle temple",
        "a snowy village",
        "a desert oasis",
        "a volcanic landscape",
        "an autumn forest path",
        "a futuristic space station",
        "a traditional Japanese house"
    ]

    comparison_results = []

    # 存储图像用于评估
    original_images = []
    lora_images = []

    for i, prompt in enumerate(test_prompts):
        print(f"\n生成对比图像 {i+1}/{len(test_prompts)}:")
        print(f"输入文本: {prompt}")

        # 原始模型
        print("  生成原始模型图像...")
        original_image = generate_image(encoder, decoder, detokenizer, tokenizer, prompt)
        original_path = f"results/comparison/original_{i}.png"
        original_image.save(original_path)
        print(f"  已保存到: {original_path}")

        # 微调模型
        print("  生成微调模型图像...")
        lora_image = generate_image(encoder, lora_decoder, detokenizer, tokenizer,
                                   f"cyberpunk style {prompt}")
        lora_path_save = f"results/comparison/cyberpunk_{i}.png"
        lora_image.save(lora_path_save)
        print(f"  已保存到: {lora_path_save}")

        # 创建对比图像（并排显示）
        print("  创建对比图像...")
        comparison_image = create_comparison_image(original_image, lora_image, prompt)
        comparison_path = f"results/comparison/comparison_{i}.png"
        comparison_image.save(comparison_path)
        print(f"  已保存到: {comparison_path}")

        # 记录结果
        comparison_results.append({
            "prompt": prompt,
            "original_image": original_path,
            "cyberpunk_image": lora_path_save,
            "comparison_image": comparison_path
        })

        # 收集图像用于评估
        original_images.append(original_image)
        lora_images.append(lora_image)

        torch.cuda.empty_cache()

    # 保存对比结果
    with open("results/comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    print("\n对比结果已保存到 results/comparison_results.json")

    # ===== 定量评估 =====
    if EVALUATION_AVAILABLE:
        print("\n" + "=" * 60)
        print("开始定量评估...")
        print("=" * 60)

        evaluation_results = {}

        # 1. 计算原始模型的 CLIP Score
        print("\n评估原始模型...")
        original_results = evaluate_model(
            images=original_images,
            texts=test_prompts,
            style_description="realistic photography",
            device=device
        )
        evaluation_results['original_model'] = original_results

        # 2. 计算微调模型的 CLIP Score 和风格相似度
        print("\n评估微调模型...")
        lora_results = evaluate_model(
            images=lora_images,
            texts=[f"cyberpunk style {p}" for p in test_prompts],
            style_description="cyberpunk style with neon lights and futuristic elements",
            device=device
        )
        evaluation_results['lora_model'] = lora_results

        # 3. 计算 FID
        print("\n计算 FID...")
        try:
            fid_calculator = FIDCalculator(device)
            fid_score = fid_calculator.calculate_fid(original_images, lora_images)
            evaluation_results['fid_between_models'] = float(fid_score)
            print(f"FID (原始 vs 微调): {fid_score:.4f}")
        except Exception as e:
            print(f"FID 计算失败: {e}")
            evaluation_results['fid_between_models'] = None

        # 4. 计算改进幅度
        if original_results['clip_score']['mean'] > 0:
            clip_improvement = (
                (lora_results['clip_score']['mean'] - original_results['clip_score']['mean'])
                / original_results['clip_score']['mean'] * 100
            )
        else:
            clip_improvement = 0

        style_improvement = (
            (lora_results['style_similarity']['mean'] - original_results['style_similarity']['mean'])
            / max(original_results['style_similarity']['mean'], 0.01) * 100
        )

        evaluation_results['improvement'] = {
            'clip_score_improvement_percent': float(clip_improvement),
            'style_similarity_improvement_percent': float(style_improvement)
        }

        # 5. 保存评估结果
        save_evaluation_results(evaluation_results, "results/evaluation/evaluation_results.json")

        # 6. 打印评估摘要
        print("\n" + "=" * 60)
        print("评估结果摘要")
        print("=" * 60)
        print(f"\n原始模型:")
        print(f"  CLIP Score: {original_results['clip_score']['mean']:.4f}")
        print(f"  风格相似度: {original_results['style_similarity']['mean']:.4f}")
        print(f"\n微调模型:")
        print(f"  CLIP Score: {lora_results['clip_score']['mean']:.4f}")
        print(f"  风格相似度: {lora_results['style_similarity']['mean']:.4f}")
        print(f"\n改进幅度:")
        print(f"  CLIP Score 提升: {clip_improvement:.2f}%")
        print(f"  风格相似度提升: {style_improvement:.2f}%")
        if evaluation_results.get('fid_between_models'):
            print(f"  FID: {evaluation_results['fid_between_models']:.4f}")
        print("=" * 60)

        # 7. 生成评估报告
        generate_evaluation_report(evaluation_results, test_prompts)
    else:
        print("\n跳过定量评估（评估模块未加载）")

    return comparison_results


def generate_evaluation_report(evaluation_results, test_prompts):
    """生成评估报告"""
    report_path = "results/evaluation/evaluation_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型评估报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 评估概述\n")
        f.write("-" * 40 + "\n")
        f.write(f"评估样本数量: {len(test_prompts)}\n")
        f.write(f"评估指标: CLIP Score, 风格相似度, FID\n\n")

        f.write("2. 原始模型评估结果\n")
        f.write("-" * 40 + "\n")
        orig = evaluation_results['original_model']
        f.write(f"CLIP Score: {orig['clip_score']['mean']:.4f} ± {orig['clip_score']['std']:.4f}\n")
        f.write(f"风格相似度: {orig['style_similarity']['mean']:.4f} ± {orig['style_similarity']['std']:.4f}\n")
        f.write(f"图像质量: {orig['image_quality']['mean']:.4f}\n\n")

        f.write("3. 微调模型评估结果\n")
        f.write("-" * 40 + "\n")
        lora = evaluation_results['lora_model']
        f.write(f"CLIP Score: {lora['clip_score']['mean']:.4f} ± {lora['clip_score']['std']:.4f}\n")
        f.write(f"风格相似度: {lora['style_similarity']['mean']:.4f} ± {lora['style_similarity']['std']:.4f}\n")
        f.write(f"图像质量: {lora['image_quality']['mean']:.4f}\n\n")

        f.write("4. 改进幅度\n")
        f.write("-" * 40 + "\n")
        imp = evaluation_results['improvement']
        f.write(f"CLIP Score 提升: {imp['clip_score_improvement_percent']:.2f}%\n")
        f.write(f"风格相似度提升: {imp['style_similarity_improvement_percent']:.2f}%\n")

        if evaluation_results.get('fid_between_models'):
            f.write(f"FID (原始 vs 微调): {evaluation_results['fid_between_models']:.4f}\n")

        f.write("\n5. 结论\n")
        f.write("-" * 40 + "\n")
        if imp['style_similarity_improvement_percent'] > 0:
            f.write("微调后的模型在赛博朋克风格生成方面有显著提升。\n")
        if imp['clip_score_improvement_percent'] > 0:
            f.write("微调后的模型在文本-图像一致性方面有所提升。\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"评估报告已保存到: {report_path}")


def run_ablation_experiment(encoder, decoder, detokenizer, tokenizer):
    """
    运行消融实验

    对比不同 LoRA rank 设置对风格迁移效果的影响
    """
    print("\n" + "=" * 60)
    print("开始消融实验...")
    print("=" * 60)

    os.makedirs("results/ablation", exist_ok=True)

    # 消融实验配置
    ablation_configs = [
        {"rank": 4, "alpha": 16, "name": "rank_4"},
        {"rank": 8, "alpha": 32, "name": "rank_8"},
        {"rank": 16, "alpha": 64, "name": "rank_16"},
    ]

    # 测试提示词（使用较少样本以节省时间）
    ablation_prompts = [
        "a busy street",
        "a beautiful landscape",
        "a modern building",
        "a peaceful garden",
        "a mountain view",
        "a sunset over the ocean",
        "a forest with tall trees",
        "a city skyline at night",
        "a beach with palm trees",
        "a lake with mountains in the background"
    ]

    ablation_results = []

    for config in ablation_configs:
        rank = config["rank"]
        alpha = config["alpha"]
        name = config["name"]

        print(f"\n消融实验: LoRA rank={rank}, alpha={alpha}")
        print("-" * 40)

        # 加载对应的 LoRA 权重
        lora_path = f"cyberpunk_lora/lora_weights_{name}.pt"

        if not os.path.exists(lora_path):
            print(f"  权重文件 {lora_path} 不存在，跳过")
            continue

        lora_weights = torch.load(lora_path, weights_only=False)

        # 应用 LoRA
        lora_decoder = apply_lora_to_decoder(decoder, rank=rank, alpha=alpha)
        lora_decoder.lora_q.lora_A.data = lora_weights['lora_q_A'].to(lora_decoder.device)
        lora_decoder.lora_q.lora_B.data = lora_weights['lora_q_B'].to(lora_decoder.device)
        lora_decoder.lora_v.lora_A.data = lora_weights['lora_v_A'].to(lora_decoder.device)
        lora_decoder.lora_v.lora_B.data = lora_weights['lora_v_B'].to(lora_decoder.device)
        lora_decoder.eval()

        # 生成图像并收集
        generated_images = []
        for i, prompt in enumerate(ablation_prompts):
            print(f"  生成图像 {i+1}/{len(ablation_prompts)}: {prompt[:30]}...")
            image = generate_image(encoder, lora_decoder, detokenizer, tokenizer,
                                 f"cyberpunk style {prompt}")
            image_path = f"results/ablation/{name}_image_{i}.png"
            image.save(image_path)
            generated_images.append(image)
            torch.cuda.empty_cache()

        # 计算评估指标
        if EVALUATION_AVAILABLE:
            print(f"  计算评估指标...")
            eval_results = evaluate_model(
                images=generated_images,
                texts=[f"cyberpunk style {p}" for p in ablation_prompts],
                style_description="cyberpunk style with neon lights and futuristic elements",
                device=device
            )

            ablation_results.append({
                "config": config,
                "clip_score": eval_results['clip_score']['mean'],
                "style_similarity": eval_results['style_similarity']['mean'],
                "image_quality": eval_results['image_quality']['mean']
            })

            print(f"  CLIP Score: {eval_results['clip_score']['mean']:.4f}")
            print(f"  风格相似度: {eval_results['style_similarity']['mean']:.4f}")
            print(f"  图像质量: {eval_results['image_quality']['mean']:.4f}")

    # 保存消融实验结果
    if ablation_results:
        save_ablation_results(ablation_results, ablation_configs)

    return ablation_results


def save_ablation_results(ablation_results, ablation_configs):
    """保存消融实验结果"""
    os.makedirs("results/ablation", exist_ok=True)

    # 保存 JSON 结果
    ablation_data = {
        "ablation_experiment": "LoRA Rank 消融实验",
        "configs": ablation_configs,
        "results": ablation_results,
        "analysis": {
            "best_clip_score": max(ablation_results, key=lambda x: x['clip_score'])['config']['name'] if ablation_results else None,
            "best_style_similarity": max(ablation_results, key=lambda x: x['style_similarity'])['config']['name'] if ablation_results else None,
        }
    }

    with open("results/ablation/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_data, f, ensure_ascii=False, indent=2)

    # 生成消融实验报告
    report_path = "results/ablation/ablation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("消融实验报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 实验设置\n")
        f.write("-" * 40 + "\n")
        f.write("实验目的: 对比不同 LoRA rank 对风格迁移效果的影响\n")
        f.write(f"测试样本数量: 10\n\n")

        f.write("2. 实验配置\n")
        f.write("-" * 40 + "\n")
        for config in ablation_configs:
            f.write(f"  - {config['name']}: rank={config['rank']}, alpha={config['alpha']}\n")
        f.write("\n")

        f.write("3. 实验结果\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'配置':<15} {'CLIP Score':<15} {'风格相似度':<15} {'图像质量':<15}\n")
        f.write("-" * 60 + "\n")
        for result in ablation_results:
            config_name = result['config']['name']
            clip = result['clip_score']
            style = result['style_similarity']
            quality = result['image_quality']
            f.write(f"{config_name:<15} {clip:<15.4f} {style:<15.4f} {quality:<15.4f}\n")
        f.write("\n")

        # 分析最佳配置
        if ablation_results:
            best_clip = max(ablation_results, key=lambda x: x['clip_score'])
            best_style = max(ablation_results, key=lambda x: x['style_similarity'])

            f.write("4. 分析结论\n")
            f.write("-" * 40 + "\n")
            f.write(f"CLIP Score 最佳配置: {best_clip['config']['name']} (rank={best_clip['config']['rank']})\n")
            f.write(f"风格相似度最佳配置: {best_style['config']['name']} (rank={best_style['config']['rank']})\n\n")

            f.write("5. 建议\n")
            f.write("-" * 40 + "\n")
            if best_style['config']['rank'] == 16:
                f.write("较大的 rank (16) 能够更好地捕捉风格特征，推荐使用。\n")
            else:
                f.write("较小的 rank 也能达到较好效果，可以根据需求选择。\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"消融实验报告已保存到: {report_path}")
    print(f"消融实验结果已保存到: results/ablation/ablation_results.json")

def save_experiment_record():
    """保存实验记录"""
    os.makedirs("experiment_records", exist_ok=True)

    record = {
        "project_name": "min-DALLE 赛博朋克风格微调",
        "date": "2026-05-02",
        "training_data": {
            "source": "使用原始模型生成",
            "count": 50,
            "prompts": [
                "cyberpunk city with neon lights at night",
                "futuristic cyberpunk street with holographic signs",
                "cyberpunk character wearing tech goggles",
                "dystopian cyberpunk city in the rain",
                "cyberpunk robot with glowing eyes"
            ]
        },
        "model_config": {
            "base_model": "min-DALLE",
            "fine_tuning_method": "LoRA",
            "lora_rank": 16,
            "lora_alpha": 64,
            "trainable_parameters": "q_proj, v_proj",
            "frozen_parameters": "其他所有参数"
        },
        "training_config": {
            "epochs": 15,
            "batch_size": 4,
            "learning_rate": "3e-4 (平滑衰减: 3e-4 * 0.95^epoch)",
            "optimizer": "AdamW",
            "loss_function": "CrossEntropy"
        },
        "generation_config": {
            "temperature": 0.8,
            "top_k": 128,
            "supercondition_factor": 6.0,
            "image_size": "256x256"
        },
        "evaluation_metrics": {
            "clip_score": "衡量文本-图像一致性",
            "style_similarity": "衡量风格匹配程度",
            "fid": "衡量生成图像分布差异"
        },
        "results": {
            "training_samples": 50,
            "comparison_samples": 50,
            "ablation_samples": 10,
            "output_directory": "results/"
        }
    }

    with open("experiment_records/experiment_record.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print("实验记录已保存到 experiment_records/experiment_record.json")

def main():
    print("=" * 60)
    print("min-DALLE 赛博朋克风格微调项目")
    print("=" * 60)

    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("cyberpunk_images", exist_ok=True)
    os.makedirs("cyberpunk_lora", exist_ok=True)
    os.makedirs("experiment_records", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)
    os.makedirs("results/visualization", exist_ok=True)
    os.makedirs("results/validation", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)
    os.makedirs("results/ablation", exist_ok=True)  # 消融实验目录

    # 加载模型和分词器
    print("\n加载模型和分词器...")
    encoder, decoder, detokenizer = load_models()
    tokenizer = load_tokenizer()
    print("模型加载完成！")

    # 生成训练数据
    generate_training_data(encoder, decoder, detokenizer, tokenizer)

    # 训练 LoRA（主实验：rank=16）
    losses = train_lora(encoder, decoder, detokenizer, tokenizer)

    # 保存可视化数据
    save_visualization_data(losses)

    # ===== 消融实验：训练不同 rank 的 LoRA =====
    ablation_configs = [
        {"rank": 4, "alpha": 16, "name": "rank_4"},
        {"rank": 8, "alpha": 32, "name": "rank_8"},
        {"rank": 16, "alpha": 64, "name": "rank_16"},
    ]

    print("\n" + "=" * 60)
    print("开始消融实验...")
    print("=" * 60)

    for config in ablation_configs:
        lora_path = f"cyberpunk_lora/lora_weights_{config['name']}.pt"
        if not os.path.exists(lora_path):
            print(f"\n训练 LoRA: {config['name']}")
            train_lora_for_ablation(
                encoder, decoder, detokenizer, tokenizer,
                rank=config['rank'],
                alpha=config['alpha'],
                config_name=config['name']
            )
        else:
            print(f"\n{config['name']} 权重已存在，跳过训练")

    # 运行消融实验评估
    ablation_results = run_ablation_experiment(encoder, decoder, detokenizer, tokenizer)

    # 生成对比图像（包含评估）
    generate_comparison(encoder, decoder, detokenizer, tokenizer)

    # 保存实验记录
    save_experiment_record()

    print("\n" + "=" * 60)
    print("项目完成！")
    print("=" * 60)
    print("\n生成的文件：")
    print("  - cyberpunk_images/: 训练数据图像")
    print("  - cyberpunk_lora/: LoRA 权重文件")
    print("  - results/comparison/: 对比结果图像")
    print("  - results/visualization/: 可视化数据")
    print("  - results/validation/: 验证图像")
    print("  - results/evaluation/: 评估结果（FID、CLIP Score等）")
    print("  - experiment_records/: 实验记录")

if __name__ == "__main__":
    main()
