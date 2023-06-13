# PHMM

## Overview
PHMM (Probabilistic Hidden Markov Model) is a Python package that performs Bayesian inference and missing value imputation on a hidden Markov model. The posterior is computed by MCMC (Markov Chain Monte Carlo). The PHMM package can perform the tasks of parameter estimation, latent sequence prediction, sequence forecasting and missing observations imputation.

## Installation
To install the package, use the command:
```py
pip install PHMM
```
To import the package, use Python command:
```py
import PHMM
```

## Synthetic Data Generation
PHMM package supports users to generate synthetic data from a categorical hidden Markov process. To do this, use
```py
# initial, transition and emission distribution
pi=np.array([0.7,0.2,0.1])
transition=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])
emission=np.array([[0.7,0.2,0.1],
                   [0.1,0.7,0.2],
                   [0.2,0.1,0.7]
    ])

# define the model, generate data
model=PHMM.PHMM(initial=pi,transition=transition,emission=emission)
z,y=model.sampleChains(n,p)
```
The PHMM.PHMM() method initializes a PHMM model object with specified initial, transition and emission distributions. To sample from this hidden Markov process, use model.sample() function. In model.sample(), parameter $n$ stands for the number of sequences and $p$ is the sequence length. The returned $z$ is the generated latent sequences and $y$ the corresponding observed sequences.

Notice that the generated sequences are fully observed. If the user wants to generate incomplete observations, he/she should generate the mask matrix manually.

## Inference I: Training & Prediction
To be continued

## Inference II: Cross Validation Score
To be continued


