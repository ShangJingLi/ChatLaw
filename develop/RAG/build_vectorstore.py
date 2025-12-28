import os
from utils import build_and_save_vectorstore
from launcher import get_project_root
from launcher import get_development_path


json_path = os.path.join(get_development_path(), "RAG", "resources", "output_json")
save_dir = os.path.join(get_project_root(), "resources", "vectorstore")

if __name__ == "__main__":
    build_and_save_vectorstore(
        json_dir=json_path,
        save_dir=save_dir
    )