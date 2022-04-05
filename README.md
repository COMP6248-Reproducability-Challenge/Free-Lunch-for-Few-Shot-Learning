# COMP6248 Reproducibility Challenge - Free Lunch for Few-Shot Learning: Distribution Calibration

Paper: https://openreview.net/forum?id=JWOiYxMG92s

To show that the results from the paper is reproducible, we re-implemented the algorithm mentioned in the paper. The authors' code for creating the dataset and few shot learning task are kept here to ensure we use the same data for comparison. Implementing the distribution calibration algorithm using PyTorch improves the efficiency of the algorithm, so training the classifiers at the end is the bottleneck of the code right now.

## Run using extracted feature
1. Download extracted features from this link: https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n?usp=sharing  
2. Create an empty folder 'cache'
3. Create an empty folder 'checkpoints'
4. Put the downloaded files under the folder './checkpoints/[miniImagenet/CUB]/WideResNet28_10_S2M2_R/last/'.
5. Run ```python evaluate_DC.py```

## TODO
Add more options for running in command line.
Parameter tuning graphs.

## References

```
@inproceedings{
yang2021free,
title={Free Lunch for Few-shot Learning:  Distribution Calibration},
author={Yang, Shuo and Liu, Lu and Xu, Min},
booktitle={International Conference on Learning Representations (ICLR)},
year={2021},
}
```