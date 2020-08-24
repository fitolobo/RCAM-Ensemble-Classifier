import numpy as np
from scipy.optimize import fmin

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Base Classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
lin_clf = SVC(kernel="linear", probability=True)

# Support Vector Classifier
svm_clf = SVC(gamma='scale', probability=True)

# K-Nearest Neighbors
knn_clf = KNeighborsClassifier(n_neighbors = 3)
                     
# Gaussian Naive-Bayes Classifier
gnb_clf = GaussianNB()

BaseClassifiers = [lin_clf, svm_clf, knn_clf, gnb_clf]

def Kentropy(alpha,U):
    K = np.exp(alpha*(np.dot(U.T,U)/U.shape[0]-1))
    return np.sum(K*np.log(K))

def CvECNN(U,xin,alpha=10,max_it=10,tau=1.e-4,verbose = True):
    Er = tau+1
    it = 0
    x = np.copy(xin)
    xold = np.copy(x)
    Energy = list()
    while (it<max_it) and (Er>tau):
        it = it+1
        w = np.exp(alpha*(np.dot(np.conj(U).T,x).real/U.shape[0]-1))
        Energy.append(-np.sum(w)/alpha)
        a = np.dot(U,w)
        Abs_a = abs(a)
        x = np.divide(a, Abs_a, out = x, where = (Abs_a>1.e-6))
        Er = np.linalg.norm(x-xold,1)
        xold = np.copy(x)
    w = np.exp(alpha*(np.dot(U.T,x).real/U.shape[0]-1))
    Energy.append(-np.sum(w)/alpha)
    if verbose==True:
        print("Exponential RCNN, with alpha = %.3f, converged in it=%d iterations from max_it=%d. \n" % (alpha, it, max_it))
    return x, np.array(Energy)

def CvICNN(U,xin,alpha=None,max_it=10,tau=1.e-4,verbose = True):
    Er = tau+1
    it = 0
    x = np.copy(xin)
    xold = np.copy(x)
    Energy = list()
    while (it<max_it) and (Er>tau):
        it = it+1
        w = np.dot(np.conj(U).T,x).real/U.shape[0]
        Energy.append(-np.sum(w))
        a = np.dot(U,w)
        Abs_a = abs(a)
        x = np.divide(a, Abs_a, out = x, where = (Abs_a>1.e-6))
        Er = np.linalg.norm(x-xold,1)
        xold = np.copy(x)
    w = np.dot(U.T,x).real/U.shape[0]    
    Energy.append(-np.sum(w))
    if verbose==True:
        print("Identity RCNN converged in it=%d iterations from max_it=%d. \n" % (it, max_it))
    return x, Energy

class HardRCNN_Ensemble(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers = BaseClassifiers, RCNN = CvECNN, alpha=None, max_it=100,tau=1.e-4,verbose = False):
        self.classifiers = classifiers
        self.RCNN = RCNN
        self.alpha = alpha
        self.max_it = max_it
        self.tau = tau
        self.verbose = verbose
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self.le_ = LabelEncoder()
        y = (self.le_).fit_transform(y)
        
        if len((self.le_).classes_)>2:
            print("Hard RCNN Ensemble Classifier can only be used for binary classification.")
        
        try:
            self.Utr_ = 2*np.vstack([clf.predict(X) for clf in self.classifiers]).T-1
        except:
            if self.verbose==True:
                print("The base classifiers have been trained previously!")
            self.Utr_ = 2*np.vstack([(clf.fit(X,y)).predict(X) for clf in self.classifiers]).T-1
            
        self.xtr_ = 2*y-1
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self,attributes="Utr_")
        
        # Input validation
        X = check_array(X)
        
        Ute = 2*np.vstack([clf.predict(X) for clf in self.classifiers]).T-1
        U = np.vstack([self.Utr_,Ute])
        
        xin = np.hstack([self.xtr_,np.zeros((Ute.shape[0],))])
        
        if self.alpha == None:
            if self.verbose==True:
                print("Tunning the parameter alpha...")
            alpha_value = fmin(Kentropy, x0=1, args=(U,), disp=self.verbose)[0]
        else:
            alpha_value = self.alpha
        y, self.Energy = self.RCNN(U, xin, alpha=alpha_value, max_it=self.max_it, tau=self.tau, verbose = self.verbose)
        
        return (self.le_).inverse_transform((y[self.Utr_.shape[0]:]>=0).astype(int))

def BipolarECNN(U,xin,alpha=10,max_it=10,tau=1.e-4,verbose = True):
    Er = tau+1
    it = 0
    x = np.copy(xin)
    xold = np.copy(x)
    while (it<max_it) and (Er>tau):
        it = it+1
        w = np.exp(alpha*(np.dot(U.T,x)/U.shape[0]-1))
        x = np.sign(np.dot(U,w))
        Er = np.linalg.norm(x-xold,1)
        xold = np.copy(x)
    if verbose==True:
        print("Bipolar ECNN, with alpha = %.3f, converged in it=%d iterations from max_it=%d. \n" % (alpha, it, max_it))
    return x

def BipolarICNN(U,xin,alpha=None,max_it=10,tau=1.e-4,verbose = True):
    Er = tau+1
    it = 0
    x = np.copy(xin)
    xold = np.copy(x)
    while (it<max_it) and (Er>tau):
        it = it+1
        w = np.dot(U.T,x)/U.shape[0]
        x = np.sign(np.dot(U,w))
        Er = np.linalg.norm(x-xold,1)
        xold = np.copy(x)
    if verbose==True:
        print("Bipolar Identity RCNN converged in it=%d iterations from max_it=%d. \n" % (it, max_it))
    return x, None
