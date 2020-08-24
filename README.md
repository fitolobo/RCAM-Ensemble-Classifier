# RCAM-Ensemble-Classifier
Ensemble of Binary Classifiers Combined Using  Recurrent Correlation Associative Memories

In this  work, we propose to combine classifiers using an associative memory model. Precisely, we introduce ensemble methods based on recurrent correlation associative memories (RCAMs) for binary classification problems. We show that an RCAM-based ensemble classifier can be viewed as a majority vote classifier whose weights depend on the similarity between the base classifiers and the resulting ensemble method.

# Getting Started
This repository contain the Python source-codes of the RCAM-Ensemble-Classifier, as described in the paper "Ensemble of Binary Classifiers Combined Using RCAMs" by Rodolfo Lobo and Marcos Eduardo Valle. The Jupyter-notebook of the computational experimens are also available in this repository.

# Usage

First of all, call the AMensemble module using:

```Python
from AMEnsemble import CvECNN, CvICNN, HardRCNN_Ensemble
```
