---
title: "Stochastic Gradient Descent For Sparse Data"
date: "2022-12-04"
author: Kacper Solarski
format: 
  hugo:
    code-fold: true

theme: default
fig-format: jpeg
toc: true
wrap: auto
number-sections: false
lang: en-GB
jupyter: python3
math: true
---



-   <a href="#motivation" id="toc-motivation">Motivation</a>
-   <a href="#why-not-regular-logistic-regression"
    id="toc-why-not-regular-logistic-regression">Why Not Regular Logistic
    Regression?</a>
-   <a href="#building-algorithm" id="toc-building-algorithm">Building
    Algorithm</a>

# Motivation

During my master studies, I was tasked to build a model that will
predict whether a mobile ad will be clicked based on a large dataset
from
[Kaggle](https://www.kaggle.com/competitions/avazu-ctr-prediction/data).
During the class, we learned Stochastic Gradient Descent (SGD) and Naive
Bayes and hence those methods were supposed to be used in the
assignment. We were also told that we're going to struggle with the size
of the dataset and it's easiest if we implement the algorithms from
scratch utilising the sparsity of the data. And this is precisely what
we did. The results were astonishing, as we were able to fit the model
**60x faster**, when comparing our implementation of SGD that made use
of matrices from
[scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
compared to the implementation from
[scikit-learn](https://scikit-learn.org/stable/modules/sgd.html). Let me
walk you through the process of building this algorithm, as it shows how
much efficiency we can gain by utilising sparse matrices.

# Why Not Regular Logistic Regression?

Why exactly did we need to use SGD and couldn't just use [Logistic
Regression
Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)?
The reason is that due to the dimension of the `X` matrix, Logistic
Regression is infeasible since the matrix cannot be inverted. However,
in SGD, we don't need to invert the full matrix and thus we're able to
operate on much larger datasets.

# Building Algorithm

First, let us define the loss that we'll be using in SGD. Since we want
to use Logistic Regression, we're going to use the log-loss:

$$
\begin{equation}
Log Loss =-\left[y_{t} \log \left(p_{t}\right)+\left(1-y_{t}\right) \log \left(1-p_{t}\right)\right]
\end{equation}
$$

Let's then translate this into the code and also add a function that
obtains predictions from weights and features.

``` python
import numpy as np
import scipy.sparse

def log_loss(p, y):
    """Obtain log-loss.

    Parameters
    ----------
    p : np.array with shape (n_samples, 1)
        Matrix of probabilities.
    y : sparse matrix with shape (n_samples, 1)
        Target values.

    Returns
    -------
    log-loss : float
    """
    y = y.A
    y = y.reshape(-1)
    p = p.reshape(-1)
    return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()

def logit(w, X):
    """Obtian predictions.

    Parameters
    ----------
    w : np.array with shape (n_features, 1)
    X : sparse matrix with shape (n_samples, n_features)
        Training data.

    Returns
    -------
    p : np.array with shape (n_samples, 1)
    """
    w = scipy.sparse.csr_matrix(w)
    p = 1 / (1 + np.exp(-((X.dot(w)).A)))
    return p
```

Note that we at least partially utilised that `X` is sparse. Having
log-loss, we can then obtain the gradient, by taking the first
derivative:

$$
\begin{equation}
\nabla Log Loss=\left(p_{t}-y_{t}\right) x_{t}
\end{equation}
$$

Once again, let's turn this into the code:

``` python
def grad(p, y, X, learning_rate):
    """Obtain gradient.

    Parameters
    ----------
    p : np.array with shape (n_samples, 1)
        Matrix of probabilities.
    y : sparse matrix with shape (n_samples, 1)
        Target values.
    X : sparse matrix with shape (n_samples, n_features)
        Training data.
    learning_rate: float or np.array of shape (n_features, )

    Returns
    -------
    gradient: np.array of shape (n_features, 1)
    """
    score = learning_rate * X.transpose().dot(p - y)
    return np.array(score)
```

Same as in previous code chunks, we operated partially on sparse
matrices. Having the gradient, we can move to the core of the algorithm,
which is updating the weights through iterations:

$$
\begin{equation}
w_{t+1} = w_{t}-\eta_{t}\left(p_{t}-y_{t}\right) x_{t}
\end{equation}
$$

I translated this into the code:

``` python
def update_weights(y, X, w, learning_rate):
    """Obtain gradient.

    Parameters
    ----------
    y : sparse matrix with shape (n_samples, 1)
        Target values.
    X : sparse matrix with shape (n_samples, n_features)
        Training data.
    w : np.array with shape (n_features, 1)
        Matrix of weights.
    learning_rate: float or np.array of shape (n_features, )

    Returns
    -------
    p : np.array with shape (n_samples, 1)
        Matrix of probabilities.
    w : np.array with shape (n_features, 1)
        Matrix of weights.
    """
    p = logit(w, X)
    score = grad(p, y, X, learning_rate)
    w = w - score
    return p, w
```

Now that we have all the above functions, let's create a function that
will iterate through the dataset in chunks, obtain predictions, gradient
and then update the weights accordingly. Let the function also print
log-loss at each iteration.

``` python
def sgd_iterative(y, X, learning_rate, chunksize):
    """Obtain gradient.

    Parameters
    ----------
    y : sparse matrix with shape (n_samples, 1)
        Target values.
    X : sparse matrix with shape (n_samples, n_features)
        Training data.
    learning_rate: float or np.array of shape (n_features)
        Learning rate to update weights.
    chunksize: int
        Number of datapoints used in each update

    Returns
    -------
    p : np.array with shape (n_samples, 1)
        Matrix of probabilities.
    w : np.array with shape (n_features, 1)
        Matrix of trained weights.
    """
    # Initialize weights and get total_chunks
    w = np.zeros((X.shape[1], 1))
    total_chunks = int(X.shape[0] / chunksize)

    # iterate through chunks
    for chunk_no in range(total_chunks):

        # slice X and y
        X_chunk = X[chunk_no * chunksize:(chunk_no + 1) * chunksize, :]
        y_chunk = y[chunk_no * chunksize:(chunk_no + 1) * chunksize, :]

        # update weights
        p, w = update_weights(y_chunk, X_chunk, w, learning_rate)

        print(f'Update {chunk_no + 1}')
        print(f'Log-loss {log_loss(p, y_chunk)}')
    
    print(f'Trained weights: {w}')

    return p, w
```

We can test the following implementation by generating the dataset and
then running the function `sgd_iterative`:

``` python
X = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
              1, 1, 1, 0, 1, 0, 1, 0, 0, 1]).reshape(10, 2)

X = scipy.sparse.csr_matrix(np.concatenate([X] * 10000))
beta = np.array([0.1, 0.5]).reshape(2, 1)
beta = scipy.sparse.csr_matrix(beta)
y = X.dot(beta)
p, w = sgd_iterative(y, X, learning_rate=0.0001, chunksize=10000)
```

    Update 1
    Log-loss 0.6931471805599453
    Update 2
    Log-loss 0.6630237709465264
    Update 3
    Log-loss 0.6417298136189502
    Update 4
    Log-loss 0.6263404036898416
    Update 5
    Log-loss 0.6149585705622571
    Update 6
    Log-loss 0.6063549610768965
    Update 7
    Log-loss 0.5997232713097223
    Update 8
    Log-loss 0.5945246559715762
    Update 9
    Log-loss 0.5903909938115283
    Update 10
    Log-loss 0.5870649025730991
    Trained weights: [[-0.94469017]
     [ 0.30482207]]

We can see that log-loss is decreasing over the iterations as the
algorithm is optimizing the weights. The algorithm above doesn't have
many features that are present in sklearn: For example, it only iterates
through the dataset once, it doesn't have any regularization parameters,
it doesn't shuffle the data, and it doesn't have early stopping.

Since I was very surprised by how fast this algorithm is on a very large
dataset, I expanded this implementation into the package
[effCTR](https://github.com/ksolarski/effCTR) and added a couple of
features to it. This
[notebook](https://github.com/ksolarski/effCTR/blob/main/notebooks/demo.ipynb)
also shows how the algorithm from the package can be used and how much
faster it is on large datasets compared to sklearn.
