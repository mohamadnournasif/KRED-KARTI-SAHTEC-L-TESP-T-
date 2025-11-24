import os, random, json, logging, logging.config, yaml, numpy as np

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def load_config(path: str = 'config/config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logging(cfg_path: str = 'config/logging.yaml'):
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
