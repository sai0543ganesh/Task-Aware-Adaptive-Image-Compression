# Final handcrafted-feature experiments

This folder contains the code for the five proposed training pipelines in *Adaptive Image Compression for Time- and Memory-Efficient Image Classifier Training*. Each run fits a fresh initial classifier, creates compression-tier labels, fits the 28-input MLP, and continues classifier training on the remaining images. The comparison baselines are archived separately.

| Experiment identifier | Dataset | Classifier | First resize |
| --- | --- | --- | --- |
| `imagenet-box-resnet18` | ImageNet-100 | ResNet-18 | BOX |
| `imagenet-box-densenet121` | ImageNet-100 | DenseNet-121 | BOX |
| `imagenet-bicubic-resnet18` | ImageNet-100 | ResNet-18 | Bicubic |
| `imagenet-bicubic-densenet121` | ImageNet-100 | DenseNet-121 | Bicubic |
| `food101-bicubic-resnet18` | Food-101 | ResNet-18 | Bicubic |

All five map the reduced image to a supported square size with bilinear resizing. `pipeline/shared/` contains the common modules. The dataset and resize-specific directories contain the differing modules. The launcher assembles the correct worker for each identifier. The experiment logic comes from the archived run source, with local cluster paths and the fixed node-name check made portable.

## Prepare the inputs

**ImageNet-100.** These comparisons use the project’s prepared 100-class [ImageNet subset](https://www.image-net.org/download). Use the original `cross_backbone_inputs.tar.gz` and the associated images to reproduce the split. If the files have moved, rebuild only their recorded paths while keeping the original split, labels, and hashes:

```bash
python prepare_imagenet_inputs.py /path/to/original/cross_backbone_inputs.tar.gz /path/to/data /path/to/imagenet_inputs
```

The data directory contains the original `clean/` and `extracted/` subdirectories. The reported split uses 20,000 initial images, 80,000 continuation images, 1,000 validation images, and 5,000 test images. For exact reproduction, obtain the original images and input archive separately. The relocation helper changes file paths while preserving the split and recorded image hashes.

**Food-101.** Download [ETH Zurich’s Food-101 archive](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) separately. Prepare the manifests in a new input directory. This uses the fixed per-class validation selection, duplicate filtering, and the official test split from the reported run.

```bash
mkdir -p /path/to/food_inputs
# Place food-101.tar.gz in /path/to/food_inputs first.
cp optimized_v28_features.csv /path/to/food_inputs/
python prepare_food101.py /path/to/food_inputs
```

The preparation script writes `cross_backbone_inputs.tar.gz` in that directory. The reported split contains 14,645 initial images, 58,572 continuation images, 2,525 validation images, and 25,250 official test images.

## Run on the cluster

The training worker uses Linux with Slurm, an NVIDIA A100, 16 CPUs, and `--mem=69G` per node. Request adequate wall time for each run. The reported environment used Python 3.12, PyTorch 2.5.1 + CUDA 12.4, torchvision 0.20.1 + CUDA 12.4, Pillow 12.3.0, NumPy 2.5.4, and scikit-learn 1.9.1. The code also imports pandas and threadpoolctl. Use an environment with these dependencies already installed; the launcher does not install packages or submit jobs.

From `final_version/` inside the allocation:

```bash
python run_experiment.py --list
python run_experiment.py imagenet-box-resnet18 --input-dir /path/to/imagenet_inputs --output-dir /path/to/results
```

Select one identifier per invocation and pass the directory prepared for that dataset. For example, to run Food-101:

```bash
python run_experiment.py food101-bicubic-resnet18 --input-dir /path/to/food_inputs --output-dir /path/to/results
```

## Read the run outputs

Use a fresh output directory for a rerun. Each worker writes to `<output-dir>/<variant>/job_<id>/<classifier>/proposed/`. The main artifacts are:

- `protocol.json` for the recorded data, model, loader, and hardware settings.
- `results.json` for complete elapsed time, test accuracy, stage timings, and memory measurements.
- `mlp/MLP_EVALUATION.json` for the tier predictor's resolved-holdout metrics.
- `test_predictions.npz` for the classifier predictions used to compute test accuracy.

This launcher runs the proposed route. The [repository overview](../README.md#results-in-the-report) provides the comparison results from the report.
