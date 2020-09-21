# RCAM-Ensemble-Classifier

In this work, we propose to combine classifiers using an associative memory model. Precisely, we introduce ensemble methods based on recurrent correlation associative memories (RCAMs) for binary classification problems. We show that an RCAM-based ensemble classifier can be viewed as a majority vote classifier whose weights depend on the similarity between the base classifiers and the resulting ensemble method.

# Getting Started
This repository contains the Python source-codes of the RCAM-Ensemble-Classifier for binary classification problems, as described in the paper "Ensemble of Binary Classifiers Combined Using RCAMs" by Rodolfo Lobo and Marcos Eduardo Valle. The Jupyter-notebook of the computational experiments is also available in this repository.

# Usage

First of all, call the AMensemble module using:

```Python
from AMEnsemble import CvECNN, CvICNN, HardRCNN_Ensemble
```

Analogously to the usage of sklearn models, we can fit and predict. Firstly, we choose the base estimators
of the ensemble. For example:

```Python
rf = RandomForestClassifier(n_estimators = 30).fit(Xtr,ytr)
y_pred = rf.predict(Xte)
```
The default base classifiers are given by 
```Python
BaseClassifiers = [lin_clf, svm_clf, knn_clf, gnb_clf]
```
Then, we can fit and predict, initializing with the estimators and the associative memory model. In this case, it is possible to use a complex-valued exponential model (CvINN) or the identity complex-valued model (CvECNN):

```Python
ICAM = HardRCNN_Ensemble(classifiers=rf.estimators_, RCNN = CvICNN, alpha=None, max_it=100,tau=1.e-4,verbose = False).fit(Xtr,ytr)
y_pred = ICAM.predict(Xte)
```
where it_max (default is it_max = 100) is the maximum number of iterations and verbose (default is False) informs if the model was previously fitted.
In particular, the associative memory models are:
```Python
CvECNN(U,xin,alpha,max_it,tau,verbose)
CvICNN(U,xin,alpha,max_it,tau,verbose)
```
where it_max (default is it_max = 10) is the maximum number of iterations and verbose (default is true) informs if the maximum number of iterations has been reached.
Another possibility is to find a better <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%5Calpha"> value using grid search

```Python
parameters = {'classifiers':[rf.estimators_], 'RCNN':[CvECNN],
'alpha':[0.01, 0.1, 0.5, 1, 5, 10, 20, 50]}
ECAM_grid = GridSearchCV(HardRCNN_Ensemble(), parameters, cv = 5).fit(Xtr,ytr)
y_pred = ECAM_grid.predict(Xte)
```
