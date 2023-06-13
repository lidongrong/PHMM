import numpy as np
import joblib
import pandas
import scipy
import itertools
import pickle
import json
from joblib import Parallel,delayed
from tqdm import tqdm



class PHMM:
    def __init__(self,initial=None,transition=None,emission=None):
        '''
        initialize a PHMM class (PHMM: Probabilisitc HMM)
        :param initial: initial distribution
        :param transition: transition matrix
        :param emission: emission probability
        :param xPrior: prior hyperparameters specified for each parameter. Should be parameters of a Dirichlet distribution
        '''
        self.initial=initial
        self.transition=transition
        self.emission=emission
        self.z=None
        self.y=None
        if not (self.initial is None):
            self.latentDim=len(self.initial)
        if not (self.emission is None):
            self.obsDim=len(self.emission[0])
    def sampleSingleChain(self,length):
        '''
        sample latent chain & obs according to parameters
        WARNING: not exposed to user!!! only for testing
        :param length: length of each chain
        :return: latent chain z and observed chain y
        '''
        if (self.initial is None) or (self.transition is None) or (self.emission is None):
            print('Parameters not fully specified, failed to generate chains')
            return
        self.latentDim = len(self.initial)
        self.obsDim = len(self.emission[0])
        # latent chain and observed chain
        latentSequence = []
        observedSequence = []
        # Sample the initial latent state.
        latentState = np.random.choice(self.latentDim, p=self.initial)
        latentSequence.append(latentState)

        # Sample the first observed emission.
        observedEmission = np.random.choice(self.obsDim, p=self.emission[latentState])
        observedSequence.append(observedEmission)

        # Sample the remaining latent states and observed emissions.
        for i in range(length - 1):
            # Sample the next latent state.
            latentState = np.random.choice(self.latentDim, p=self.transition[latentState])
            latentSequence.append(latentState)

            # Sample the next observed emission.
            observedEmission = np.random.choice(self.obsDim, p=self.emission[latentState])
            observedSequence.append(observedEmission)

        return latentSequence, observedSequence

    def sampleChains(self,n,length):
        '''
        sample n chains with common length = length
        :param n: # of chains
        :param length: length of each chain
        :return: an n x length matrix as latent chain and n x length matrix as observed chain
        '''
        latentSequences = []
        observedSequences=[]
        for i in range(n):
            latentSequence, observedSequence = self.sampleSingleChain(length)
            latentSequences.append(latentSequence)
            observedSequences.append(observedSequence)
        z=np.array(latentSequences)
        y=np.array(observedSequences)
        return z,y

    def fit(self,y,obsDim,latentDim,postNum,initInitial=None,initTransition=None,initEmission=None,initZ=None,
            initialPrior=None, transitionPrior=None, emissionPrior=None,
            log=True,core=1):
        '''
        fit the model, get estimations
        :param y: incomplete observed sequences
        :param obsDim: number of observed states
        :param latentDim: number of latent states
        :param initInitial: initial value of initial distribution
        :param initTransition: initial value of transition
        :param initEmission: initial value of emission
        :param initZ: initial values of latent sequences
        :param x prior: hyper parameters placed on priors. If None, all 1 are placed by default
        :param postNum: posterior sample num
        :param log: if display the prog bar. True for displaying the prog bar
        :param core: if use multiprocessing
        :return: return nothing
        '''
        # prior specification
        # if no hyper parameter on prior has been specified, use uniform
        if  (initialPrior is None):
            self.initialPrior = np.ones(latentDim)
        else:
            self.initialPrior=np.array(initialPrior)
        if  (transitionPrior is None):
            self.transitionPrior = np.ones((latentDim,latentDim))
        else:
            self.transitionPrior=np.array(transitionPrior)
        if  (emissionPrior is None):
            self.emissionPrior = np.ones((latentDim,obsDim))
        else:
            self.emissionPrior=np.array(emissionPrior)
        # parameter initialization
        self.obsDim=obsDim
        self.latentDim=latentDim
        self.y=y
        if initTransition is None:
            self.transition=np.random.dirichlet(np.ones(self.latentDim),self.latentDim)
        else:
            self.transition=initTransition
        if initInitial is None:
            self.initial=np.random.dirichlet(np.ones(self.latentDim))
        else:
            self.initial=initInitial
        if initEmission is None:
            self.emission = np.random.dirichlet(np.ones(self.obsDim),self.latentDim)
        else:
            self.emission=initEmission
        if initZ is None:
            self.z=None
        else:
            self.z=initZ
        # prepare posterior
        self.postInitial=np.zeros((postNum,self.latentDim))
        self.postTransition=np.zeros((postNum,self.latentDim,self.latentDim))
        self.postEmission=np.zeros((postNum,self.latentDim,self.obsDim))
        # record the voting of latent states at each iteration
        self.vote=np.zeros((self.y.shape[0],self.y.shape[1],self.latentDim))
        # objective function
        self.objFunc=np.zeros(postNum)
        # start running
        with Parallel(n_jobs=core, backend='loky', max_nbytes='1M') as parallel:
            for s in tqdm(range(postNum)):
                self.gibbsUpdate(parallel)
                self.postInitial[s]=self.initial
                self.postTransition[s]=self.transition
                self.postEmission[s]=self.emission
                self.objFunc[s]=self.func
                # record the voting of each latent site
                # chatGPT told me this works, I don't know why, but it works nicely.
                self.vote[np.arange(self.vote.shape[0]).reshape(-1, 1), np.arange(self.vote.shape[1]), self.z.astype(int)]+=1

        # get estimated z and mean
        self.getMean()
        self.getZ()

    def crossValidationScore(self,y,obsDim,latentDim,postNum,initInitial=None,initTransition=None,initEmission=None,initZ=None,
            initialPrior=None, transitionPrior=None, emissionPrior=None,
            log=True,core=1,
            validationRef=None, predNum=1000,predCore=1,fraction=0.1,folds=10):
        '''
        perform k fold cross validation and report the acc on imputing missing values in obs sequences
        Procedure: random drop some observed entries and try to predict them using the fitted model
        do this in k-fold cross validation
        :param y: same
        :param obsDim: same
        :param latentDim: same
        :param postNum: same
        :param initInitial: same
        :param initTransition: same
        :param initEmission: same
        :param initZ: same
        :param initialPrior: same
        :param transitionPrior: same
        :param emissionPrior: same
        :param log: same
        :param core: same
        :param folds: number of folds, 10 by default
        :param predNum: Number of posterior draw to predict (in cross validation)
        :param predCore: Number of cores used for prediction ( in cross validation )
        :param fraction: fraction of entries set to be missing in predictive evaluation
        :param validationRef: reference for mapping the training to the right position
        :return: cross validated acc on imputing missing values in obs sequences
        '''
        if validationRef is None:
            validationRef=np.eye(max(latentDim,obsDim))
            validationRef=validationRef[:latentDim,:obsDim]
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        y = y[indices]

        # Split the data into n_folds equal parts
        foldSizes = np.full(folds, y.shape[0] // folds, dtype=int)
        foldSizes[:y.shape[0] % folds] += 1

        # Perform cross-validation
        scores = []
        current = 0
        for foldSize in foldSizes:
            start, stop = current, current + foldSize
            y_train = np.concatenate((y[:start], y[stop:]))
            y_test = y[start:stop]

            # get the indices of the observed entries in the test set
            mask = ~np.isnan(y_test)
            # Get the indices of the non-zero elements (True values)
            nonZeroIndices = np.argwhere(mask)
            # Calculate the number of elements to set to 0 (False)
            numElementsToZero = int(fraction * len(nonZeroIndices))
            # Randomly select non-zero indices to set to 0
            selectedIndices = nonZeroIndices[
                np.random.choice(len(nonZeroIndices), numElementsToZero, replace=False)]
            # Update the mask: set the selected indices to 0 (False)
            mask[selectedIndices[:, 0], selectedIndices[:, 1]] = False

            # mask some observations in y test randomly and use this new masked version for evaluation
            masked_y_test=y_test.copy()
            masked_y_test[mask]=np.nan
            # Fit the model
            self.fit(y=y_train,obsDim=obsDim,latentDim=latentDim,postNum=postNum,
                     initInitial=initInitial,initTransition=initTransition,initEmission=initEmission,initZ=initZ,
                     initialPrior=initialPrior,transitionPrior=transitionPrior,emissionPrior=emissionPrior,
                     log=log,core=core)

            self.map(reference=validationRef)
            # Make predictions and compute the score
            z_vote, z_pred, y_vote, y_pred = self.predict(y=masked_y_test,postNum=predNum,core=predCore)
            score = np.sum(np.equal(y_test,y_pred)*mask)/np.sum(mask)
            scores.append(score)

            current = stop

        return np.array(scores)


    def predict(self,y,postNum=1000,core=1):
        '''
        Predict according to new incomplete sequences y, return posterior predictive distribution
        :param y: incomplete sequences
        :param postNum: number of draws from posterior predictive
        :param core: number of cores used for computing
        :return: posterior predictive distribution of z. Include voting result and prediction from majority vote
        '''
        zVote=np.zeros((y.shape[0],y.shape[1],self.latentDim))
        yVote=np.zeros((y.shape[0],y.shape[1],self.latentDim))
        with Parallel(n_jobs=core, backend='loky', max_nbytes='1M') as parallel:
            for s in tqdm(range(postNum)):
                self.gibbsUpdate(parallel)
                # update chains
                ff = parallel(delayed(self.forwardFilter)(y[t]) for t in range(y.shape[0]))
                ff = np.array(ff)
                # evaluate objective function
                newZ = parallel(delayed(self.backwardSampler)(y[t], ff[t]) for t in range(y.shape[0]))
                newZ = np.array(newZ)
                # sample missing entries of y
                newY = np.zeros(y.shape)
                # for each k, generate a mask
                mask = np.array([newZ==k for k in range(self.latentDim)])
                newSample=np.array(
                    [np.random.choice(np.arange(self.obsDim), size=y.shape, p=self.emission[k]) for k in range(self.latentDim)])
                # keep observed entries unchanged
                # chatGPT told me this works, I don't know why, but it works nicely.
                for k in range(len(mask)):
                    newY[mask[k]]=newSample[k][mask[k]]
                newY[~np.isnan(y)]=y[~np.isnan(y)]
                # record the voting result
                yVote[np.arange(yVote.shape[0]).reshape(-1, 1), np.arange(yVote.shape[1]), newY.astype(int)] += 1
                # record the voting of each latent site
                # chatGPT told me this works, I don't know why, but it works nicely.
                zVote[np.arange(zVote.shape[0]).reshape(-1, 1), np.arange(zVote.shape[1]), newZ.astype(int)] += 1

        predZ = np.argmax(zVote, axis=2)
        predY = np.argmax(yVote,axis=2)
        return zVote,predZ, yVote, predY



    def gibbsUpdate(self,parallel):
        '''
        perform a gibbs update, update self.parameters
        :param parallel: parallel object created by joblib.Parallel
        :return: return nothing
        '''
        # update chains
        ff = parallel(delayed(self.forwardFilter)(self.y[t]) for t in range(self.y.shape[0]))
        ff = np.array(ff)
        # evaluate objective funcrtion
        self.func=np.log(np.sum(ff[:,-1,:]))
        newZ=parallel(delayed(self.backwardSampler)(self.y[t],ff[t]) for t in range(self.y.shape[0]))
        self.z=np.array(newZ)
        # update parameters
        newInitial=self.initialSampler()
        self.initial=newInitial
        newTransition=self.transitionSampler()
        self.transition=newTransition
        newEmission=self.emissionSampler()
        self.emission=newEmission


    def initialSampler(self):
        '''
        sample the initial distribution
        WARNING: Private Function, not exposed to users!
        :return: sampled initial
        '''
        self.latentDim=len(self.initial)
        start = self.z[:, 0]
        startNum = np.array([np.sum(start==i) for i in range(self.latentDim)])
        newInitial = np.random.dirichlet(self.initialPrior + startNum)
        return newInitial

    def transitionSampler(self):
        '''
        sample transition matrix
        WARNING: Private Function, not exposed to users!
        :return: return the sampled transition
        '''
        self.latentDim=len(self.initial)
        newTransition = np.zeros((self.latentDim, self.latentDim))
        for j in range(0, newTransition.shape[0]):
            count_of_change = np.zeros(newTransition.shape[1])
            for k in range(0,newTransition.shape[1]):
                # search the pattern of transition from j to k
                search_pattern = [j, k]
                table = (self.z[:, :-1] == search_pattern[0]) & (self.z[:, 1:] == search_pattern[1])
                count_of_change[k] = np.sum(table)
            # Generate dirichlet distribution
            dist = np.random.dirichlet((self.transitionPrior[j] + np.array(count_of_change)))
            newTransition[j, :] = dist
        return newTransition

    def emissionSampler(self):
        '''
        sample the emission matrix
        WARNING: Private Function, not exposed to users!
        :return: return the sampled emission matrix
        '''
        self.latentDim,self.obsDim =self.emission.shape
        newEmission=np.zeros((self.latentDim,self.obsDim))
        for j in range(0, self.latentDim):
            # for j th row of B, calculate the number of each
            # observed states respectively
            obsFreq=np.zeros(self.obsDim)
            for k in range(0, self.obsDim):
                n = np.sum(np.logical_and(self.z == j, self.y == k))
                obsFreq[k]=n
            #obsFreq=np.array(obsFreq)
            newEmission[j, :] = np.random.dirichlet(self.emissionPrior[j] + obsFreq, 1)[0]
        return newEmission

    def forwardFilter(self,y):
        '''
        calculate the forward probability given a single sequence y
        WARNING: Private Function, not exposed to users!
        :param y: the partial observed sequence
        :return: return a length-n array standing for the observed prob
        '''
        # indicator on each site's missingness, 1 for observed and 0 for missing
        indicator = ~np.isnan(y)

        # length of the whole sequence
        T = len(y)
        self.latentDim=len(self.initial)
        self.obsDim=self.emission.shape[1]

        # start to compute alpha recursively
        alpha = np.zeros((T, self.latentDim))
        # calculate the first entry of alpha
        if indicator[0]:
            # the first entry observed
            alpha[0] = self.initial * self.emission[:, int(y[0])]
        else:
            # first entry missing
            alpha[0] = self.initial
        for i in range(1, T):
            # corresponds to the case that y_i is observable
            if indicator[i]:
                alpha[i, :] = np.dot(alpha[i - 1, :], self.transition) * self.emission[:, int(y[i])]
            else:
                alpha[i, :] = np.dot(alpha[i - 1, :], self.transition)

        return alpha


    def backwardSampler(self,y,alpha):
        '''
        backward sampling based on observation y and its forward probability alpha
        WARNING: Private Function, not exposed to users!
        :param y: observed sequence
        :param alpha: forward probability
        :return: sampled latent sequence
        '''
        # initialize the output
        latentSequence = np.zeros(len(y))
        T=len(y)
        self.latentDim=len(self.initial)
        self.obsDim=self.emission.shape[1]

        # First sample the last latent state
        w = alpha[T - 1, :] / np.sum(alpha[T - 1, :])
        latentSequence[T-1]=np.random.choice(self.latentDim,1,p=w)

        # Then sample each latent state in sequence
        for t in range(T-2, -1, -1):
            # compute the index of hidden state z_{t+1}
            next = latentSequence[t+1]
            w = self.transition[:, int(next)] * alpha[t, :]
            w = w/np.sum(w)
            latentSequence[t]=np.random.choice(self.latentDim,1,p=w)
        return latentSequence

    def getZ(self):
        '''
        get an estimated z via mmap (majority vote)
        WARNING: Private function, not exposed to the users!
        :return: return the estimated z hat
        '''
        self.zHat=np.argmax(self.vote,axis=2)
        self.z=self.zHat
        return self.zHat

    def getMean(self):
        '''
        get posterior mean
        WARNING: Private Function, not exposed to the users
        :return: return posterior mean
        '''
        # last 1/3 as posterior draw
        postNum=int(self.postInitial.shape[0]/10)
        self.initialMean=np.mean(self.postInitial[-postNum:],axis=0)
        self.transitionMean=np.mean(self.postTransition[-postNum:],axis=0)
        self.emissionMean=np.mean(self.postEmission[-postNum:],axis=0)
        self.initial=self.initialMean
        self.transition=self.transitionMean
        self.emission=self.emissionMean
        return self.initialMean,self.transitionMean,self.emissionMean

    #############################################################################
    # following are some supporting functions for data analysis or postprocessing
    #############################################################################
    def map(self,reference):
        '''
        tackle the label switching problem: permute estimations according to referenceInitial
        :param referenceInitial: a vector as a reference for initial, or emission
        :return: nothing
        '''
        # last 1/10 as posterior
        if np.array(reference).shape == self.initial.shape:
            postNum = int(self.postEmission.shape[0]/10)
            #postNum=10
            estimatedPostInitial = sum(self.postInitial[-postNum:]) / postNum
            referenceInitial = np.array(reference)
            index = [i for i in range(0, len(referenceInitial))]
            allPermutations = list(itertools.permutations(index))
            # calcualte distance
            distance = []
            for i in range(0, len(allPermutations)):
                pmt = list(allPermutations[i])
                distance.append(np.linalg.norm(estimatedPostInitial[pmt] - referenceInitial))
            # get the map that maps estimation to the right positions
            permute=list(allPermutations[np.argmin(distance)])
            self.permute=permute
            for i in range(0, self.postInitial.shape[0]):
                # permute est_A
                self.postTransition[i] = self.postTransition[i][permute]
                self.postTransition[i] = self.postTransition[i][:, permute]
                # permute est_pi
                self.postInitial[i] = self.postInitial[i][permute]
                # permute est emission
                self.postEmission[i]=self.postEmission[i][permute, :]
                # permute votings
            self.vote=self.vote[:,:, permute]
            self.getMean()
            self.getZ()
        elif self.emission.shape == np.array(reference).shape:
            postNum = int(self.postEmission.shape[0] / 10)
            # postNum=10
            estimatedPostEmission = sum(self.postEmission[-postNum:]) / postNum
            reference = np.array(reference)
            index = [i for i in range(0, len(reference))]
            allPermutations = list(itertools.permutations(index))
            # calcualte distance
            distance = []
            for i in range(0, len(allPermutations)):
                pmt = list(allPermutations[i])
                distance.append(np.linalg.norm(estimatedPostEmission[pmt,:] - reference))
            # get the map that maps estimation to the right positions
            permute = list(allPermutations[np.argmin(distance)])
            self.permute = permute
            for i in range(0, self.postInitial.shape[0]):
                # permute est_A
                self.postTransition[i] = self.postTransition[i][permute,:]
                self.postTransition[i] = self.postTransition[i][:, permute]
                # permute est_pi
                self.postInitial[i] = self.postInitial[i][permute]
                # permute est emission
                self.postEmission[i] = self.postEmission[i][permute, :]
                # permute votings
            self.vote = self.vote[:, :, permute]
            self.getMean()
            self.getZ()
        else:
            # unidentified input
            pass

    def metric(self,trueZ,trueInitial,trueTransition,trueEmission):
        '''
        evaluate metrics on true data (if applicable)
        :param trueZ: true latent sequence
        :param trueInitial: true Initial
        :param trueTransition: true Transition
        :param trueEmission: true emission
        :return: acc, error on initial, transition and emission
        '''
        self.getMean()
        self.getZ()
        self.acc= np.sum(np.equal(self.zHat,trueZ))/np.size(self.zHat)
        self.initialError=np.mean(np.square(self.initialMean-trueInitial))
        self.transitionError=np.mean(np.square(self.transitionMean-trueTransition))
        self.emissionError=np.mean(np.square(self.emissionMean-trueEmission))
        # AIC and BIC
        self.AIC=-2*self.objFunc + 2*(self.initial.size + self.transition.size + self.emission.size)
        self.BIC=-2*self.objFunc + np.log(self.z.size)*(self.initial.size + self.transition.size + self.emission.size)
        return self.acc, self.initialError, self.transitionError, self.emissionError

    def summarize(self):
        self.getMean()
        # get posterior variance
        # last 1/3 as posterior draw
        postNum = int(self.postInitial.shape[0] / 3)
        self.initialVar = np.var(self.postInitial[-postNum:], axis=0)
        self.transitionVar = np.var(self.postTransition[-postNum:], axis=0)
        self.emissionVar = np.var(self.postEmission[-postNum:], axis=0)

    def saveModel(self, modelName):
        '''
        pickle the model
        :return: return nothing
        '''
        self.name=modelName
        with open(f"{modelName}.pkl", "wb") as f:
            pickle.dump(self, f)

    def loadModel(self,modelName):
        '''
        load a PHMM model and return it
        :param modelName: name of the model, should be a .pkl file
        :return: the loaded model
        '''
        with open(f"{modelName}.pkl", "rb") as f:
            model = pickle.load(f)
        return model