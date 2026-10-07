import os


def resolve_checkpoint(checkpoint: str) -> str:
    """Get the local path of a checkpoint, given either a local directory or a HuggingFace repo id to download."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)
