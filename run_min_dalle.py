import os
import requests
import json
import numpy as np
import torch
from tqdm import tqdm
from min_dalle_stub.text_tokenizer import TextTokenizer
from min_dalle_stub.dalle_bart_encoder import DalleBartEncoder
from min_dalle_stub.dalle_bart_decoder import DalleBartDecoder
from min_dalle_stub.vqgan_detokenizer import VQGanDetokenizer
from PIL import Image

def main():
    url = 'https://hf-mirror.com/kuprel/min-dalle/resolve/main/'
    folder = "files"

    if not os.path.exists(folder):
        os.makedirs(folder)

    vocab_path = os.path.join(folder, "vocab.json")
    merges_path = os.path.join(folder, "merges.txt")

    if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
        print("下载文件")

        vocab = requests.get(url + 'vocab.json')
        with open(vocab_path, 'wb') as f:
            f.write(vocab.content)

        merges = requests.get(url + 'merges.txt')
        with open(merges_path, 'wb') as f:
            f.write(merges.content)
    else:
        print("vocab和merges文件都已存在")

    with open(vocab_path, 'r', encoding='utf8') as f:
        vocab = json.load(f)

    with open(merges_path, 'r', encoding='utf8') as f:
        merges = f.read().split("\n")[1:-1]

    print("初始化分词器")
    tokenizer = TextTokenizer(vocab, merges)

    text = "panda with top hat reading a book"
    print(f"生成提示词: {text}")

    token_ids = tokenizer.tokenize(text)
    print(f"Token的数字索引: {token_ids}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    text_tokens = np.ones((2, 64), dtype=np.int32)
    text_tokens[0, :2] = [token_ids[0], token_ids[-1]]
    text_tokens[1, :len(token_ids)] = token_ids
    text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)

    encoder_path = os.path.join(folder, "encoder.pt")

    if not os.path.exists(encoder_path):
        print("下载预训练好的编码器权重文件")
        ws = requests.get(url + 'encoder.pt')
        with open(encoder_path, 'wb') as f:
            f.write(ws.content)
    else:
        print("编码器文件已存在")

    dtype = "float16"
    print(f"使用数据类型: {dtype}")

    print("开始初始化DalleBartEncoder")
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

    print("正在进行文本编码...")
    encoder_state = encoder.forward(text_tokens)
    print(f"文本向量的形状: {encoder_state.shape}")
    torch.cuda.empty_cache()

    decoder_path = os.path.join(folder, "decoder.pt")

    if not os.path.exists(decoder_path):
        print("下载解码器权重文件")
        weights = requests.get(url + 'decoder.pt')
        with open(decoder_path, 'wb') as f:
            f.write(weights.content)
    else:
        print("解码器文件已存在")

    print("初始化解码器")
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

    attention_mask = text_tokens.not_equal(1)[:, None, None, :]
    attention_state = torch.zeros(size=(24, 4, 256, 2048), dtype=getattr(torch, dtype), device=device)
    image_tokens = torch.full((1, 256 + 1), 2 ** 14 - 1, dtype=torch.long, device=device)
    token_indices = torch.arange(256, device=device)

    temperature = 0.5
    top_k = 128
    supercondition_factor = 6
    settings = torch.tensor([temperature, top_k, supercondition_factor], dtype=torch.float32, device=device)

    print("开始生成图像 token...")
    with torch.no_grad():
        for i in tqdm(range(256), desc="生成 token", unit="token"):
            torch.cuda.empty_cache()
            image_tokens[:, i + 1], attention_state = decoder.sample_tokens(
                settings=settings,
                attention_mask=attention_mask,
                encoder_state=encoder_state,
                attention_state=attention_state,
                prev_tokens=image_tokens[:, [i]],
                token_index=token_indices[[i]]
            )

    detokenizer_path = os.path.join(folder, "detoker.pt")

    if not os.path.exists(detokenizer_path):
        print("下载detokenizer参数")
        ws = requests.get(url + 'detoker.pt')
        with open(detokenizer_path, 'wb') as f:
            f.write(ws.content)
    else:
        print("detokenizer参数已存在")

    print("初始化VQGanDetokenizer")
    detokenizer = VQGanDetokenizer().eval()

    params = torch.load(detokenizer_path, weights_only=False)
    detokenizer.load_state_dict(params)
    del params
    detokenizer = detokenizer.to(device)
    torch.cuda.empty_cache()

    print("生成图像...")
    image = detokenizer.forward(True, image_tokens[:, 1:])

    image = image.to(torch.uint8).to('cpu').numpy()
    image = Image.fromarray(image)

    output_path = os.path.join(folder, "minDALLE.png")
    image.save(output_path)
    print(f"图像已保存到: {output_path}")

    image.show()

if __name__ == "__main__":
    main()
