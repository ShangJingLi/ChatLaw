from openi import openi_download_file

from launcher import get_resources_path


def download_resources(resource_type):
    if resource_type not in ("tokenizer", "llm", "video_model"):
        raise ValueError("Arg 'resource_type' must be in (tokenizer, llm, video_model)")
    if resource_type == "tokenizer":
        openi_download_file("enter/QwenTokenizer", repo_type="dataset" , local_dir=get_resources_path(), max_workers=10)
    elif resource_type == "llm":
        openi_download_file("enter/QwenModel", repo_type="dataset", local_dir=get_resources_path(), max_workers=10)
    elif resource_type == "video":
        openi_download_file("enter/QwenModel", repo_type="dataset", local_dir=get_resources_path(), max_workers=10)


__all__ = ["download_resources"]
