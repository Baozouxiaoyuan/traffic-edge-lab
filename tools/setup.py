"""Install a portable CPU or NVIDIA environment from the project root."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-build", choices=("cu126", "cu130", "cu132"), default="cu126")
    parser.add_argument("--collision", action="store_true", help="Also download the optional 4.3 GB collision model")
    parser.add_argument("--examples", action="store_true", help="Download two checksummed Zenodo development clips")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error("Use Python 3.11, 3.12 or 3.13 for the pinned application dependencies.")
    if args.compute == "cuda" and sys.platform == "darwin":
        parser.error("NVIDIA CUDA requires Windows or Linux. Use --compute cpu on macOS.")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)

    def run(*arguments: str) -> None:
        subprocess.run([str(python), *arguments], cwd=root, check=True)

    run("-m", "pip", "install", "--upgrade", "pip")
    torch_command = ["-m", "pip", "install", "--upgrade", "torch", "torchvision"]
    if sys.platform != "darwin":
        build = args.cuda_build if args.compute == "cuda" else "cpu"
        torch_command += ["--index-url", f"https://download.pytorch.org/whl/{build}"]
    run(*torch_command)
    run("-m", "pip", "install", "-r", "requirements.txt")
    run("tools/prepare_models.py")
    if args.examples:
        run("tools/prepare_examples.py")
    if args.collision:
        run("tools/prepare_collision_vlm.py")
    run("-c", "import torch; print('PyTorch:', torch.__version__); print('NVIDIA CUDA available:', torch.cuda.is_available())")
    print("Setup complete. Start with START_CAPSTONE.cmd on Windows or ./START_CAPSTONE.sh on macOS/Linux.")
    if args.compute == "cuda":
        print("The NVIDIA option also requires a compatible GPU and driver. Automatic mode can use the CPU when CUDA is unavailable.")


if __name__ == "__main__":
    main()
