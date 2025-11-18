"""配置文件"""
import os
import ast
import argparse
from pprint import pformat
import yaml
from launcher import get_project_root

_config_path = os.path.join(get_project_root(), "config.yaml")


class Config:
    """Convert dictionary to object with attribute access."""
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
    def __str__(self):
        return pformat(self.__dict__)

def parse_yaml(yaml_path):
    """Load YAML configuration."""
    with open(yaml_path, "r", encoding="utf-8") as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)
    return cfg

def parse_cli_args(cfg):
    """Dynamically add command-line arguments based on YAML config."""
    parser = argparse.ArgumentParser(description="Configuration override parser")
    for key, val in cfg.items():
        arg_type = ast.literal_eval if isinstance(val, bool) else type(val)
        parser.add_argument(f"--{key}", type=arg_type, default=val,
                            help=f"Override default value (default: {val})")
    args = parser.parse_args()
    return vars(args)

def get_config():
    """Main entry: load YAML + CLI args."""
    # 默认加载同目录下 config.yaml
    yaml_path = os.path.join(os.path.dirname(__file__), _config_path)
    cfg = parse_yaml(yaml_path)
    cli_cfg = parse_cli_args(cfg)
    print("Final merged configuration:")
    print(pformat(cli_cfg))
    return Config(cli_cfg)

config = get_config()
