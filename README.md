# [Value Iteration Networks](https://arxiv.org/abs/1602.02867) in TensorFlow

> Tamar, A., Wu, Y., Thomas, G., Levine, S., and Abbeel, P. _Value Iteration Networks_. Neural Information Processing Systems (NIPS) 2016

This repository was forked from [@TheAbhiKumar](https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks). The detailed information is also given in the above link.

## Training
- Download the 16x16 and 28x28 GridWorld datasets from the [author's repository](https://github.com/avivt/VIN/tree/master/data). This repository contains the 8x8 GridWorld dataset for convenience and its small size.
- Then, 
```
python train.py
```

## Dependencies
- Python 2.7
- TensorFlow 1.0
- SciPy >= 0.18.1 (to load the data)

## Difference
- VIN in TensorFlow is modified by using TF Slim.
- The code is modified to run on Python 2.7.
