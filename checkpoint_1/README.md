# Adaptive Image Compression for Faster Image Classifier Training

This package accompanies the experiment results in the thesis. It includes the frozen Checkpoint 1 implementation and its verified result summaries. The additional bicubic comparisons appear below as reported results.

## Checkpoint 1 — ImageNet-100, job 6176

| Model | Complete runtime, baseline → proposed | Time saved | Test top-1, baseline → proposed | Test top-5, baseline → proposed |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 8,094.28 s → 5,580.52 s | 31.06% | 80.84% → 76.20% (−4.64 pp) | 93.32% → 91.80% |
| DenseNet-121 | 14,542.16 s → 10,839.27 s | 25.46% | 83.72% → 77.84% (−5.88 pp) | 95.16% → 92.92% |

The [ImageNet-100](https://arxiv.org/abs/1409.0575) subset used 100,000 training, 1,000 validation and 5,000 test images. Each baseline/proposed pair ran on the same physical A100-01 GPU with 16 CPUs, 77 GiB requested RAM, classifier batch size 64, four training/evaluation workers, FP32 and TF32 disabled. Each route used the same source-loading policy and its own process, model and timing. The baseline trained for 50 epochs on the full training split. The proposed route trained its own initial classifier for 50 epochs on 20% of the images and continued from its own teacher on the remaining 80% for 48 compressed-input epochs and two resolution-adaptation epochs.

The teacher assigns a preferred retention tier to initial images. A three-seed MLP ensemble uses 222 MobileNet, attention and numerical features to predict one of 11 retention tiers. Checkpoint 1 reduces pixel area with BOX and maps the result to a supported training size with bilinear resizing. The complete runtime includes teacher fitting, labeling, feature extraction, MLP fitting and prediction, image preparation, classifier training, validation, checkpointing and test evaluation as applicable to each route. Archive transport is outside the measured runtime.

| Proposed model | Raw within ±1 tier | Raw exact | Applied ordinal unsafe | Resolved MLP holdout |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 71.67% | 49.05% | 3.94% | 3,576 / 4,000 |
| DenseNet-121 | 72.63% | 47.93% | 5.57% | 3,591 / 4,000 |

The classifier results use one seed, and the evaluations reuse image identities from development. These are whole-pipeline comparisons. Time savings include the effect of the proposed training schedule and preparation, as well as the compressed inputs.

## Additional verified comparisons

**ImageNet-100, job 6483 on A100-02.** This job used the same dataset split as Checkpoint 1 and compared bicubic compression with its own baselines.

| Model | Complete runtime, baseline → bicubic | Time saved | Test top-1, baseline → bicubic | Test top-5, baseline → bicubic |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 2 h 15 min 36 s → 1 h 28 min 29 s | 34.74% | 80.30% → 76.34% (−3.96 pp) | 93.62% → 91.76% |
| DenseNet-121 | 3 h 52 min 48 s → 3 h 02 min 31 s | 21.60% | 84.02% → 77.36% (−6.66 pp) | 94.88% → 92.76% |

**[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), job 6506.** The ResNet-18 pair ran on the same A100 and used the 25,250 official test images.

| Model | Complete runtime, baseline → bicubic | Time saved | Test top-1, baseline → bicubic |
| --- | ---: | ---: | ---: |
| ResNet-18 | 1 h 32 min 55 s → 1 h 12 min 3 s | 22.46% | 75.52% → 71.95% (−3.56 pp) |

Time savings and accuracy changes are calculated before rounding. Each saving is relative to that job's baseline. The ImageNet link documents the parent dataset.

## Code and evidence

| File or folder | Contents |
| --- | --- |
| [code_package/code/](code_package/code/) | Frozen Checkpoint 1 source and its local Python dependencies. Start with [the paired runner](code_package/code/final_phase_models_study.py) and [the per-route worker](code_package/code/final_worker.py). |
| [code_package/inputs/](code_package/inputs/) | Source audit and ordered list of the 222 MLP features. |
| [Verified ResNet-18 reports](verified_reports/results/job_6176/resnet18/) | Baseline and proposed results and protocols, plus the paired comparison. |
| [Verified DenseNet-121 reports](verified_reports/results/job_6176/densenet121/) | Baseline and proposed results and protocols, plus the paired comparison. |
| [Independent final audit](INDEPENDENT_FINAL_AUDIT.json) | Accuracy, MLP and runtime metrics recomputed from the original saved artifacts. |

The Checkpoint 1 code keeps the original Slurm and hardware checks. Re-running it requires the corresponding dataset splits and input archive, the original images, the frozen MobileNetV3-Small checkpoint, and a compatible CUDA environment. The per-route protocols record the software versions and run settings. The published JSON summaries omit local filesystem paths. Dataset files, trained weights, per-image arrays, and full experiment archives are outside this package.
