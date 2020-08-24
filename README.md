# RCAM-Ensemble-Classifier

In this  work, we propose to combine classifiers using an associative memory model. Precisely, we introduce ensemble methods based on recurrent correlation associative memories (RCAMs) for binary classification problems. We show that an RCAM-based ensemble classifier can be viewed as a majority vote classifier whose weights depend on the similarity between the base classifiers and the resulting ensemble method.

# Getting Started
This repository contain the Python source-codes of the RCAM-Ensemble-Classifier for binary classification problems, as described in the paper "Ensemble of Binary Classifiers Combined Using RCAMs" by Rodolfo Lobo and Marcos Eduardo Valle. The Jupyter-notebook of the computational experimens are also available in this repository.

# Usage

First of all, call the AMensemble module using:

```Python
from AMEnsemble import CvECNN, CvICNN, HardRCNN_Ensemble
```

Analogously to the usage of sklearn models we can fit and predict. Firstly, we choose one base estimator
in order to obtain the base classifiers:

```Python
rf = RandomForestClassifier(n_estimators = 30).fit(Xtr,ytr)
y_pred = rf.predict(Xte)
```

and finally, we can fit and predict, initializing using the previous estimators and one associative memory model. In this case, it is possible to use a complex valued exponential model or the identity complex valued model.

```Python
ICAM = HardRCNN_Ensemble(classifiers=rf.estimators_, RCNN = CvICNN).fit(Xtr,ytr)
y_pred = ICAM.predict(Xte)
```
