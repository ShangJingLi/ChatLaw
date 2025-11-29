import os
import openi
from launcher import get_resources_path


def download_resources(resource_type):
    if resource_type not in ("tokenizer", "llm", "video_model"):
        raise ValueError("Arg 'resource_type' must be in (tokenizer, llm, video_model)")
    if resource_type == "tokenizer":
        openi.openi_download_file("enter/QwenTokenizer", repo_type="dataset" , local_dir=get_resources_path(), max_workers=10)
    elif resource_type == "llm":
        openi.openi_download_file("enter/QwenModel", repo_type="model", local_dir=os.path.join(get_resources_path(), "llm"), max_workers=10)
    elif resource_type == "video_model":
        openi.openi_download_file("enter/VoskModel", repo_type="dataset", local_dir=get_resources_path(), max_workers=10)


__all__ = ["download_resources"]
