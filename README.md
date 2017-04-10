# [Value Iteration Networks](https://arxiv.org/abs/1602.02867) in TensorFlow

> Tamar, A., Wu, Y., Thomas, G., Levine, S., and Abbeel, P. _Value Iteration Networks_. Neural Information Processing Systems (NIPS) 2016

> This repository was forked from [@TheAbhiKumar](https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks)

## Training
- Download the 16x16 and 28x28 GridWorld datasets from the [author's repository](https://github.com/avivt/VIN/tree/master/data). This repository contains the 8x8 GridWorld dataset for convenience and its small size.

```
# Runs the 8x8 Gridworld with default parameters
python train.py
```
The detailed introduction is given in [@TheAbhiKumar](https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks).

## Dependencies
* Python >= 3.6
* TensorFlow >= 1.0
* SciPy >= 0.18.1 (to load the data)

## Datasets
* The GridWorld dataset used is from the author's repository. It also contains Matlab scripts to generate the dataset. The code to process the dataset is from the original repository with minor modifications under this [license](https://github.com/avivt/VIN/blob/master/LICENSE.md)
* The model was also originally tested on three other domains and the author's original code will be [released eventually](https://github.com/avivt/VIN/issues/4)
  * Mars Rover Navigation
  * Continuous control
  * WebNav

## Resources

* [Value Iteration Networks on arXiv](https://arxiv.org/abs/1602.02867)
* [Aviv Tamar's (author) original implementation in Theano](https://github.com/avivt/VIN)
* [NIPS 2016 Supplemental](http://tx.technion.ac.il/~avivt/nips16supp.pdf)
* [ICML Slides](http://technion.ac.il/~danielm/icml_slides/Talk7.pdf)
