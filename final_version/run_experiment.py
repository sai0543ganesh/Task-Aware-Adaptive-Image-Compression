"""Run one of the five reported handcrafted-feature pipelines in Slurm."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "imagenet-box-resnet18": ("imagenet_box", "resnet18"),
    "imagenet-box-densenet121": ("imagenet_box", "densenet121"),
    "imagenet-bicubic-resnet18": ("imagenet_bicubic", "resnet18"),
    "imagenet-bicubic-densenet121": ("imagenet_bicubic", "densenet121"),
    "food101-bicubic-resnet18": ("food101_bicubic", "resnet18"),
}


def stage_worker(variant, destination):
    for directory in (ROOT / "pipeline" / "shared", ROOT / "pipeline" / variant):
        for module in sorted(directory.glob("*.py")):
            shutil.copyfile(module, destination / module.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS, nargs="?")
    parser.add_argument("--input-dir", type=Path, help="Directory with cross_backbone_inputs.tar.gz")
    parser.add_argument("--output-dir", type=Path, help="Directory in which to write run results")
    parser.add_argument("--list", action="store_true", help="Print the five experiment identifiers")
    args = parser.parse_args()
    if args.list:
        print("\n".join(EXPERIMENTS))
        return
    if not (args.experiment and args.input_dir and args.output_dir):
        parser.error("Provide an experiment, --input-dir and --output-dir, or use --list.")
    if sys.platform != "linux" or not all(os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURMD_NODENAME")):
        parser.error("Run the training worker inside a Linux Slurm allocation.")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "16" or os.environ.get("SLURM_MEM_PER_NODE") != "70656":
        parser.error("Use the reported job setup: 16 CPUs and --mem=69G per node.")

    input_dir = args.input_dir.resolve()
    if not (input_dir / "cross_backbone_inputs.tar.gz").is_file():
        parser.error("cross_backbone_inputs.tar.gz is missing from --input-dir.")
    variant, model = EXPERIMENTS[args.experiment]
    output_dir = args.output_dir.resolve() / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        MTP_RESEARCH_INPUT_DIR=str(input_dir),
        MTP_RESEARCH_OUTPUT_ROOT=str(output_dir),
        MTP_PROJECT_ROOT=str(input_dir),
        MTP_TASK_ARCHITECTURE=model,
        FINAL_RESNET_ARM="proposed",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("MTP_RAM_PREP_RESUME", None)

    with tempfile.TemporaryDirectory(prefix="mtp-worker-") as temporary:
        worker = Path(temporary)
        stage_worker(variant, worker)
        subprocess.run([sys.executable, str(worker / "verify_data_policy.py")], cwd=worker, env=env, check=True)
        subprocess.run([sys.executable, "-u", str(worker / "final_worker.py")], cwd=worker, env=env, check=True)


if __name__ == "__main__":
    main()
