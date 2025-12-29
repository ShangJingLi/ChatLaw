import os
from typing import List
import openi
from launcher import get_resources_path

llm_list = ["config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",]

tokenizer_list = ["merges.txt",
                  "tokenizer.json",
                  "tokenizer_config.json",
                  "vocab.json",]

audio_model_list = [".mdl",
                    ".msc",
                    ".mv",
                    "am.mvn",
                    "config.yaml",
                    "configuration.json",
                    "model_quant.onnx",
                    "tokens.json"]

vectorstore_list = ["index.faiss",
                    "index.pkl"]


def download_resources(resource_type):
    """
        根据缺少的文件通过启智社区下载补齐。

        Args:
            resource_type (str): 要下载的文件类型

        Returns:
            bool:
                True  - 目录存在且所有文件齐全
                False - 目录不存在，或缺少任意文件
        """
    if resource_type not in ("tokenizer", "llm", "audio_model", "vectorstore"):
        raise ValueError("Arg 'resource_type' must be in (tokenizer, llm, audio_model)", "vectorstore")
    if resource_type == "tokenizer" and check_files_complete("tokenizer", tokenizer_list) is False:
        openi.openi_download_file("enter/QwenTokenizer",
                                  repo_type="dataset" ,
                                  local_dir=get_resources_path(),
                                  max_workers=10)
    elif resource_type == "llm" and check_files_complete("llm", llm_list) is False:
        openi.openi_download_file("enter/QwenModel",
                                  repo_type="model",
                                  local_dir=os.path.join(get_resources_path(), "llm"),
                                  max_workers=10)
    elif resource_type == "audio_model" and check_files_complete("audio_model", audio_model_list) is False:
        openi.openi_download_file("enter/Paraformer",
                                  repo_type="dataset",
                                  local_dir=os.path.join(get_resources_path(), "audio_model"),
                                  max_workers=10)
    elif resource_type == "vectorstore" and check_files_complete("vectorstore", vectorstore_list) is False:
        openi.openi_download_file(
            "enter/vectorstore",
            repo_type="dataset",
            local_dir=os.path.join(get_resources_path(), "vectorstore"),
            max_workers=10)
    else:
        pass


def check_files_complete(resource_type: str, required_files: List[str]) -> bool:
    """
    检查目录是否存在，且目录下是否包含所需的全部文件。

    Args:
        resource_type (str): 要检查的文件类型
        required_files (List[str]): 必须存在的文件名列表（仅文件名，不含路径）

    Returns:
        bool:
            True  - 目录存在且所有文件齐全
            False - 目录不存在，或缺少任意文件
    """
    # 目录是否存在
    if not os.path.isdir(os.path.join(get_resources_path(), resource_type)):
        return False

    # 检查文件是否齐全
    for filename in required_files:
        file_path = os.path.join(os.path.join(get_resources_path(), resource_type), filename)
        if not os.path.isfile(file_path):
            return False

    return True


__all__ = ["download_resources"]
