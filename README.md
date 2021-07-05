# SNLI Paper Implementation

## Introduction
This code was written as a part on an assignment for a deep learning course. In this assignment we were asked to reproduce results of an SNLI-based paper. Stanford's website contains several papers which attempt to correctly classify samples in the SNLI dataset. The paper I chose to implement is "A Decomposable Attention Model for Natural Language Inference" by Parikh et al. I used the PyTorch package to reproduce its results. You may find the full report of the assignment in this repository.

## Installation Notes

1. Download one of the GloVe embeddings from https://nlp.stanford.edu/projects/glove/

2. Make sure that Pytorch is installed on your machine

## Run
```
python train_model.py <glove_file_path>
```
