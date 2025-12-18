import yaml

def load_config(path):
    # 关键修改：显式指定encoding='utf-8'，覆盖系统默认编码
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)