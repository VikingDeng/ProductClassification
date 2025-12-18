import argparse
import os
import re
import yaml
import codecs
import nltk
import pandas as pd
from pathlib import Path
from transformers import CLIPTokenizer

# --------------------------
# 1. 全局配置（与训练集保持一致，确保预处理逻辑统一）
# --------------------------
# NLTK资源标记文件（和主进程一致）
NLTK_DOWNLOAD_MARKER = Path.home() / ".nltk_download_done"
# CLIP配置（和训练集一致）
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_TEXT_TOKENS = 75  # 训练集使用的max_tokens
CLIP_MAX_LEN = 77
# NLTK核心POS标签（名词/动词/形容词）
CORE_POS_TAGS = {'NN', 'NNP', 'NNS', 'VB', 'VBP', 'JJ'}
# NLTK资源映射（和训练集一致）
NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"
}

# --------------------------
# 2. 工具函数：读取YAML配置（处理UTF-8 BOM，兼容Windows）
# --------------------------
def read_yaml_config(config_path):
    """读取YAML配置文件，处理UTF-8 BOM，返回完整配置字典"""
    config_abs_path = os.path.abspath(config_path)
    if not os.path.exists(config_abs_path):
        raise FileNotFoundError(f"❌ YAML配置文件不存在：{config_abs_path}")
    # 读取文件，处理BOM
    with codecs.open(config_abs_path, 'r', encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f)
    return cfg

# --------------------------
# 3. 工具函数：下载NLTK资源（仅主进程执行，避免重复下载）
# --------------------------
def download_nltk_resources():
    """下载并检查NLTK资源，生成标记文件"""
    if NLTK_DOWNLOAD_MARKER.exists():
        print("✅ NLTK资源已下载，标记文件存在")
        return

    # 下载资源
    for res_name in NLTK_RESOURCES.keys():
        try:
            nltk.data.find(NLTK_RESOURCES[res_name])
            print(f"✅ NLTK资源[{res_name}]已存在，跳过下载")
        except LookupError:
            print(f"📥 正在下载NLTK资源[{res_name}]...")
            nltk.download(res_name, quiet=True)
            print(f"✅ NLTK资源[{res_name}]下载完成")

    # 生成标记文件
    NLTK_DOWNLOAD_MARKER.touch()
    print("✅ 所有NLTK资源下载完成，标记文件已创建")

# --------------------------
# 4. 工具函数：CLIP Tokenizer单例（避免重复加载）
# --------------------------
def get_clip_tokenizer(model_name=CLIP_MODEL_NAME):
    """CLIP Tokenizer单例"""
    if not hasattr(get_clip_tokenizer, 'tokenizer'):
        get_clip_tokenizer.tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return get_clip_tokenizer.tokenizer

# --------------------------
# 5. 核心文本预处理函数（和训练集完全一致）
# --------------------------
def clean_text(text):
    """清洗文本：处理空值、特殊字符、多余空格"""
    if pd.isna(text) or text == 'nan' or text.strip() == '':
        return ""
    text = re.sub(r'[^\w\s]', ' ', text)  # 移除标点
    text = re.sub(r'[\n\t]+', ' ', text)  # 移除换行/制表符
    text = re.sub(r'\s+', ' ', text).strip()  # 合并多余空格
    return text

def extract_core_tokens_from_text(text):
    """提取文本中的核心token（名词/动词/形容词）"""
    if not text:
        return []
    # 分词（使用punkt_tab）
    tokens = nltk.word_tokenize(text.lower())
    # 过滤停用词、数字、短token
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [
        token for token in tokens
        if token not in stop_words and not token.isdigit() and len(token) >= 2
    ]
    if not filtered_tokens:
        return []
    # 词性标注
    tagged_tokens = nltk.pos_tag(filtered_tokens)
    # 提取核心POS标签的token
    core_tokens = [token for token, tag in tagged_tokens if tag in CORE_POS_TAGS]
    return core_tokens if core_tokens else filtered_tokens

def clip_tokenize_and_truncate(text, max_tokens):
    """CLIP分词并截断，返回处理后的文本和token数量"""
    if not text:
        return "", 0
    tokenizer = get_clip_tokenizer()
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False,
        return_tensors="pt"
    )
    truncated_text = tokenizer.decode(encoding.input_ids[0], skip_special_tokens=True)
    token_count = len(encoding.input_ids[0])
    return truncated_text, token_count

def dynamic_simplify_text(title, description, max_tokens):
    """动态简化文本（和训练集逻辑一致），确保token数量不超过限制"""
    clean_title = clean_text(title)
    clean_desc = clean_text(description)

    # 生成文本候选（优先级从高到低）
    text_candidates = []
    if clean_desc:
        text_candidates.append(f"{clean_title} {clean_desc}")
    desc_core_tokens = extract_core_tokens_from_text(clean_desc)
    if desc_core_tokens:
        text_candidates.append(f"{clean_title} {' '.join(desc_core_tokens)}")
    text_candidates.append(clean_title)
    title_core_tokens = extract_core_tokens_from_text(clean_title)
    if title_core_tokens or desc_core_tokens:
        text_candidates.append(f"{' '.join(title_core_tokens)} {' '.join(desc_core_tokens)}")

    # 遍历候选，返回第一个符合要求的文本
    for text in text_candidates:
        if not text:
            continue
        truncated_text, token_count = clip_tokenize_and_truncate(text, max_tokens)
        if token_count <= max_tokens and truncated_text:
            return truncated_text

    return "product"  # 兜底文本

# --------------------------
# 6. 主预处理函数：处理测试集并保存
# --------------------------
def preprocess_test_set(cfg):
    """
    从配置中读取测试集路径，执行预处理，保存为test_precomputed.csv
    :param cfg: YAML配置字典
    """
    # 读取测试集配置
    if 'dataset' not in cfg or 'test' not in cfg['dataset']:
        raise ValueError("❌ 配置文件中缺少dataset.test部分")
    test_cfg = cfg['dataset']['test']
    data_root = test_cfg.get('data_root', './data')
    csv_file = test_cfg.get('csv_file', 'test.csv')

    # 构建测试集CSV路径
    data_root_abs = os.path.abspath(data_root)
    csv_abs_path = os.path.join(data_root_abs, csv_file) if not os.path.isabs(csv_file) else csv_file
    if not os.path.exists(csv_abs_path):
        raise FileNotFoundError(f"❌ 测试集CSV文件不存在：{csv_abs_path}")

    # 加载测试集数据
    print(f"📝 正在加载测试集：{csv_abs_path}")
    df = pd.read_csv(csv_abs_path)
    print(f"📝 测试集共{len(df)}条数据")

    # 检查必要的列（title/description/id）
    required_cols = ['id', 'title', 'description']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ 测试集CSV缺少必要列：{col}")

    # 执行文本预处理（和训练集一致）
    print(f"⚙️ 正在执行文本预处理（max_tokens={MAX_TEXT_TOKENS}）...")
    df['text_simplified'] = df.apply(
        lambda row: dynamic_simplify_text(
            row['title'],
            row['description'],
            MAX_TEXT_TOKENS
        ),
        axis=1
    )

    # 保存预处理后的CSV
    save_dir = os.path.dirname(csv_abs_path)
    save_path = os.path.join(save_dir, "test_precomputed.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ 测试集预处理完成，保存至：{save_path}")

    return save_path

# --------------------------
# 7. 主函数：解析参数+执行预处理
# --------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="通用测试集预处理脚本：使用NLTK处理文本并保存为test_precomputed.csv")
    parser.add_argument('--config', type=str, required=True, help="YAML配置文件路径（如configs/clip_v2_simplified_text.yaml）")
    args = parser.parse_args()

    # 步骤1：下载NLTK资源
    download_nltk_resources()

    # 步骤2：读取YAML配置
    cfg = read_yaml_config(args.config)

    # 步骤3：执行测试集预处理
    preprocess_test_set(cfg)

if __name__ == '__main__':
    main()