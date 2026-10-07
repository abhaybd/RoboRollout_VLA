"""
Install openpi's patches to transformers, which openpi's PyTorch models (including MolmoBot-Pi0) require.
This modifies the installed transformers package, so rerun it whenever transformers is reinstalled.
Equivalent to the `cp -r` command in openpi's README, which assumes a source checkout of openpi.
"""

from pathlib import Path
import shutil


def main():
    import openpi.models_pytorch.transformers_replace
    import transformers

    src = Path(openpi.models_pytorch.transformers_replace.__path__[0])
    dst = Path(transformers.__path__[0])
    print(f"Copying {src} to {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    print("Done")


if __name__ == "__main__":
    main()
