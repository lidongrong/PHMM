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
The training interface of the package follows the convention in sklearn. That is, we can directly call a model to initialize it, and then use `train()` to perform training and `predict()` to perform inference and prediction.

```py
import PHMM
# and import other packages, numpy, pandas, etc


# read the data
path='SimulatedData/SampleSize500'
modelPath='Size500ModelCheckPoints'
# specify latent dimension and obs dimension
latentDim=3
obsDim=3
# how many sample you would like to draw from posterior
postNum=75
# number of cores to use
core=2
# number of replicates of the experiments
num=4
# sample size
size=500

# load data
"""
z = latent sequences
y = observed sequences, nan denotes missing values
pi,A and B are true parameters (initial, transition and emission)
"""
z=np.load(f'{path}/z.npy')
y=np.load(f'{path}/y.npy')
pi=np.load(f'{path}/pi.npy')
A=np.load(f'{path}/A.npy')
B=np.load(f'{path}/B.npy')
# some preprocessing
z=z.astype(float)
y=y.astype(float)
z=z[:size,:]
y=y[:size,:]

# prepare some important variables
# rates = missing rate, we are generating some artificial missings manually
rates=[7,9]
summary={}


# run the experiments
for j in range(len(rates)):
    acc=[]
    initialMSE=[]
    transitionMSE=[]
    emissionMSE=[]
    for i in range(num):
        # generate partially observed data, each time a new mask is generated
        r= 0.1 * rates[j]
        mask=np.random.binomial(1,r,y.shape)
        partial_y = y.copy()
        partial_y[mask == 1] = np.nan
        # initialize model
        model=PHMM.PHMM()
        # fit model
        model.fit(y=partial_y, obsDim=obsDim, latentDim=latentDim, postNum=postNum,
                  initZ=None, initInitial=None, initEmission=None, initTransition=None,
                  initialPrior = np.array([1,1,1]),
                  log=True, core=1,
                  transition_alpha = 0.1, transition_delta = 1e-3
                    )
        # evaluate metrics
        # permute latent states to prevent label switching
        model.map(reference=pi)
        # evaluate the metrics to true parameters
        model.metric(trueZ=z, trueInitial=pi, trueTransition=A, trueEmission=B)
        model.summarize()


        # save model
        uniqueID=f'Missing{rates[j]}Exp{i}Model'
        model.saveModel(f'{modelPath}/{uniqueID}')
        acc.append(model.acc)
        initialMSE.append(model.initialError)
        transitionMSE.append(model.transitionError)
        emissionMSE.append(model.emissionError)
        del model
    summary[f'rate{rates[j]}']=[np.mean(acc),np.var(acc),np.mean(initialMSE),np.var(initialMSE),
                                np.mean(transitionMSE),np.var(transitionMSE),
                                np.mean(emissionMSE),np.var(emissionMSE)]
    print(summary)



summary=pd.DataFrame(summary,index=['acc', 'acc Var', 'init MSE', 'init Var', 'trans MSE', 'trans Var', 'em MSE', 'em Var'])
summary=summary.T
print(summary)
summary.to_csv(f'{modelPath}/summary.txt')
```


## Inference II: Cross Validation Score
To be continued


