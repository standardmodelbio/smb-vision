import argparse

from huggingface_hub import HfApi


def upload_model(folder_path: str, repo_id: str, repo_type: str = "model") -> None:
    """
    Upload a model folder to Hugging Face Hub.

    Args:
        folder_path (str): Path to the local model folder
        repo_id (str): Hugging Face repository ID
        repo_type (str): Type of repository (default: "model")
    """
    api = HfApi()
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=["*.pt", "*.pth", "*.safetensors", "*.json", "*.yaml", "*.txt", "*.md"],
        ignore_patterns=["*checkpoint*", "*pycache*", "*__pycache__*", "*logs*", "*wandb*", "*wandb*"],
    )


def main():
    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub")
    parser.add_argument("--folder-path", required=True, help="Path to the local model folder")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID")
    parser.add_argument("--repo-type", default="model", help="Type of repository (default: model)")

    args = parser.parse_args()
    upload_model(args.folder_path, args.repo_id, args.repo_type)


if __name__ == "__main__":
    main()
