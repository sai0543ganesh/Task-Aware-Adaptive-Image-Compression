# Adaptive Image Compression for Time- and Memory-Efficient Image Classifier Training

This project selects a compression tier for each training image before classifier continuation. An initial classifier supplies tier labels, and a multilayer perceptron learns to predict those tiers from 28 handcrafted image features. The remaining images are resized, grouped into compatible batches, and reused across training epochs.

## Repository guide

- [Final handcrafted-feature experiments](final_version/README.md) contains the five proposed pipelines, dataset preparation instructions, and Slurm commands.
- [Earlier experiment archive](checkpoint_1/README.md) contains the code and results for a previous predictor that used learned image descriptors. Its measurements belong to that earlier setup.

## Results in the report

| Dataset | Classifier | Downsampling | Complete time saved | Peak host RAM reduction | Test top-1, baseline → proposed |
| --- | --- | --- | ---: | ---: | ---: |
| ImageNet-100 | ResNet-18 | BOX | 36.65% | 71.66% | 80.30% → 76.78% |
| ImageNet-100 | DenseNet-121 | BOX | 24.85% | 71.32% | 84.02% → 78.08% |
| ImageNet-100 | ResNet-18 | Bicubic | 34.63% | 71.04% | 80.30% → 75.66% |
| ImageNet-100 | DenseNet-121 | Bicubic | 24.85% | 72.16% | 84.02% → 77.76% |
| Food-101 | ResNet-18 | Bicubic | 28.19% | 77.06% | 75.52% → 72.03% |

Complete time includes preparation, tier prediction, classifier training, and test evaluation. Peak host RAM is the maximum sampled process-tree proportional set size across the run. These are comparisons of the full pipeline against independent baselines on the same dataset splits. They measure the combined effect of the pipeline's compression, loading, and training choices.

The experiments use the project's prepared [ImageNet](https://www.image-net.org/download) subset and the [Food-101 archive from ETH Zurich](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz). The training images and original per-image manifests are stored separately from the repository.
