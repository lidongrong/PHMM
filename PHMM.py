import joblib
import pandas
import scipy.stats as ss
import itertools
import pickle
import json
from joblib import Parallel,delayed
from tqdm import tqdm
from scipy.stats import dirichlet
import torch
from torch.distributions import Dirichlet
import numpy as np
import time

class PHMM:
    """
    Stands for Integrated Probabilistic Hidden Markov Model
    Sample from a collapsed posterior whose latent states are integrated out
    """
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
            log=True,core=1,transition_alpha = 10.0, transition_delta = 0.001):
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
        :param transition_alpha: alpha used in sampling the transition matrix
        :param transition_epsilon: correcting term used in sampling the transition matrix
        :return: return nothing
        '''
        # set some parameters in sampling
        self.transition_alpha = transition_alpha
        self.transition_delta = transition_delta
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
        # observation mask, 1 for observed, 0 for missing
        self.obs_mask = ~np.isnan(self.y)
        # observed indices for each sequence
        self.obs_idx = [np.where(indicator)[0] for indicator in self.obs_mask]
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
        start = time.time()
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
                # set nan values to 0, anyway they are not important
                #print(f'A: {self.transition}')
                #print(f'B: {self.emission}')
                #print(f'initial: {self.initial}')
                self.vote[np.arange(self.vote.shape[0]).reshape(-1, 1), np.arange(self.vote.shape[1]), np.nan_to_num(self.z, nan=0).astype(int)]+=1
        end = time.time()
        self.total_time = end - start
        # AIC and BIC
        self.AIC = -2 * self.objFunc + 2 * (self.initial.size + self.transition.size + self.emission.size)
        self.BIC = -2 * self.objFunc + np.log(self.z.size) * (self.initial.size + self.transition.size + self.emission.size)
        # get estimated z and mean
        self.getMean()
        self.getZ()

    def crossValidationScore(self,y,obsDim,latentDim,postNum,initInitial=None,initTransition=None,initEmission=None,initZ=None,
            initialPrior=None, transitionPrior=None, emissionPrior=None,
            log=True,core=1,
            validationRef=None, predNum=1000,predCore=1,fraction=0.1,folds=10,drop='random'):
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
        :param drop: 'random':drop the observations randomly. 'forecast':drop the last fraction percent of data.
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
            if drop=='forecast':
                window= int(np.ceil(y_test.shape[1]*fraction))
                # set all observations in window as missing
                mask[:,-window:]=False
            # if drop can't be specified, treat it as dropping randomly
            else:
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


    def predict(self,y,postNum=1000,core=1,posterior = False):
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
                #print('emission matrix: ',self.emission)
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
        # if to return the posterior sampling result
        if posterior:
            return zVote, yVote
        else:
            return predZ, predY

    def gibbsUpdate(self,parallel):
        '''
        perform a gibbs update, update self.parameters
        :param parallel: parallel object created by joblib.Parallel
        :return: return nothing
        '''
        # update chains
        ff = parallel(delayed(self.forwardFilter)(self.y[t],self.obs_idx[t]) for t in range(self.y.shape[0]))
        ff = np.array(ff)
        # evaluate objective funcrtion
        self.func=np.log(np.sum(ff[:,-1,:]))
        newZ=parallel(delayed(self.backwardSampler)(self.y[t],ff[t],self.obs_idx[t]) for t in range(self.y.shape[0]))
        self.z=np.array(newZ)
        # update parameters
        newInitial=self.initialSampler()
        self.initial=newInitial
        newTransition=self.transitionSampler(alpha = self.transition_alpha, delta = self.transition_delta)
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
        #print(newInitial)
        return newInitial

    def transitionSampler(self, alpha=0.1, delta=0.001):
        """
        Sample the transition matrix using a Metropolis-Hastings step on the simplex
        for partially observed chains. Instead of using a Dirichlet proposal, we use a
        random walk in the unconstrained (ALR-transformed) space. Each row gets its own scaling
        factor, and that row-specific scale is adapted dynamically so that its acceptance rate
        is approximately 0.234.

        It is assumed that the following members are defined as class attributes
        prior to or during the first call to this method:

          - self.scaling: NumPy array (length latentDim) of initial noise scales for the random walk.
          - self.total_acceptance: NumPy array (length latentDim) for accepted proposals.
          - self.total_attempts: NumPy array (length latentDim) for proposal attempts.
          - self.gamma: adaptation rate constant (suggested default is 0.1)
          - self.target_accept: target acceptance rate for each row (default: 0.234)

        :param alpha: initial scaling for each row (used for the first call)
        :param delta: (unused here) retained for compatibility
        :return: The updated transition matrix.
        """

        # Initialize class-level members if needed.
        if not hasattr(self, 'scaling'):
            self.scaling = np.full(self.latentDim, alpha, dtype=float)
        if not hasattr(self, 'total_acceptance'):
            self.total_acceptance = np.zeros(self.latentDim, dtype=float)
        if not hasattr(self, 'total_attempts'):
            self.total_attempts = np.zeros(self.latentDim, dtype=float)
        if not hasattr(self, 'gamma'):
            self.gamma = 0.1
        if not hasattr(self, 'target_accept'):
            self.target_accept = 0.234

        newTransition = np.copy(self.transition)
        accepted_count_total = 0  # for debug printing

        # --- Helper functions for the ALR random walk proposal ---
        def alr_transform(x, eps=1e-10):
            """
            Map a probability vector x on the simplex (length K) to R^(K-1)
            using the additive log-ratio (ALR) transform.
            (Uses the last component as the reference.)
            """
            x = np.clip(x, eps, None)
            return np.log(x[:-1]) - np.log(x[-1])

        def inverse_alr_transform(y):
            """
            Inverse of the ALR transform: maps a point y in R^(K-1)
            back to a point x on the K-dimensional simplex.
            """
            exp_y = np.exp(y)
            x = np.concatenate([exp_y, [1.0]])
            return x / np.sum(x)

        def jacobian_inverse_alr(y):
            """
            Compute the absolute determinant of the Jacobian for the inverse ALR transform.

            For the ALR transformation:
              y_i = log(x_i) - log(x_K),  i = 1,...,K-1,
            the inverse is:
              x_i = exp(y_i) / (1 + sum(exp(y)))   for i=1,...,K-1, and
              x_K = 1 / (1 + sum(exp(y))).
            The Jacobian determinant is given by:
              |J| = (∏ exp(y_i)) / (1 + sum(exp(y)))^K,
            where K = len(y) + 1.
            """
            exp_y = np.exp(y)
            K = len(y) + 1
            return np.prod(exp_y) / ((1 + np.sum(exp_y)) ** K)

        # --- Likelihood and Prior functions (unchanged from the original code) ---
        def compute_likelihood(trans_matrix):
            # Get dimensions: n_chains x T (observations per chain)
            n_chains, T = self.z.shape

            # Boolean mask for observed (non-NaN) entries.
            observed_mask = ~np.isnan(self.z)

            # Valid consecutive observed pairs.
            valid_pairs_mask = observed_mask[:, :-1] & observed_mask[:, 1:]
            if not np.any(valid_pairs_mask):
                return 0.0

            start_states = self.z[:, :-1][valid_pairs_mask].astype(int)
            end_states = self.z[:, 1:][valid_pairs_mask].astype(int)

            time_indices = np.arange(T)
            start_times = time_indices[:-1].reshape(1, -1).repeat(n_chains, axis=0)[valid_pairs_mask]
            end_times = time_indices[1:].reshape(1, -1).repeat(n_chains, axis=0)[valid_pairs_mask]
            gaps = end_times - start_times

            unique_gaps = np.unique(gaps)
            gap_powers = {gap: np.linalg.matrix_power(trans_matrix, gap) for gap in unique_gaps}

            log_lik = 0.0
            for gap in unique_gaps:
                gap_mask = (gaps == gap)
                if np.any(gap_mask):
                    power_mat = gap_powers[gap]
                    log_lik += np.sum(np.log(power_mat[start_states[gap_mask], end_states[gap_mask]] + 1e-300))
            return log_lik

        def compute_prior(row, prior):
            return dirichlet.logpdf(row, prior)

        # --- MH update for each row of the transition matrix ---
        for j in range(self.latentDim):
            current_row = newTransition[j, :]

            # Use the row-specific scaling factor.
            scale_j = self.scaling[j]

            # --- Random Walk Proposal in ALR space ---
            y_current = alr_transform(current_row)
            y_proposed = y_current + np.random.normal(0, scale_j, size=y_current.shape)
            proposed_row = inverse_alr_transform(y_proposed)

            jac_current = jacobian_inverse_alr(y_current)
            jac_proposed = jacobian_inverse_alr(y_proposed)

            # --- Compute likelihoods and priors ---
            current_loglik = compute_likelihood(newTransition)
            current_logprior = compute_prior(current_row, self.transitionPrior[j])
            temp_trans = np.copy(newTransition)
            temp_trans[j, :] = proposed_row
            proposed_loglik = compute_likelihood(temp_trans)
            proposed_logprior = compute_prior(proposed_row, self.transitionPrior[j])

            # --- Acceptance Ratio with Jacobian Correction ---
            log_ratio = (proposed_loglik + proposed_logprior + np.log(jac_proposed)) - \
                        (current_loglik + current_logprior + np.log(jac_current))

            if np.log(np.random.random()) < log_ratio:
                newTransition[j, :] = proposed_row
                accepted = True
                accepted_count_total += 1
            else:
                accepted = False

            # --- Update row-specific cumulative counters ---
            self.total_attempts[j] += 1
            if accepted:
                self.total_acceptance[j] += 1

            # Compute row-specific acceptance rate and adapt the scaling factor.
            acc_rate_j = self.total_acceptance[j] / self.total_attempts[j]
            self.scaling[j] *= np.exp(self.gamma * (acc_rate_j - self.target_accept))

        # --- Optional global debug output ---
        global_attempts = np.sum(self.total_attempts)
        global_acceptance = np.sum(self.total_acceptance)
        global_accept_rate = global_acceptance / global_attempts if global_attempts > 0 else 0.0

        '''
        print(f"Accepted in this call: {accepted_count_total} out of {self.latentDim} rows")
        print(f"Cumulative acceptance (per row):")
        for j in range(self.latentDim):
            print(f"  Row {j}: {self.total_acceptance[j]:.1f}/{self.total_attempts[j]:.1f} "
                  f"(rate = {self.total_acceptance[j] / self.total_attempts[j]:.3f}), scaling = {self.scaling[j]:.3f}")
        print(f"Global acceptance: {global_acceptance:.1f}/{global_attempts:.1f} ({global_accept_rate:.3f})")
        print(f"Updated transition matrix:\n{self.transition}")
        '''
        #print(f'transition: {self.transition}')

        # Update the model's transition matrix, then return.
        self.transition = newTransition

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

    def forwardFilter(self,y,obs_idx):
        '''
        calculate the forward probability given a single sequence y
        WARNING: Private Function, not exposed to users!
        :param y: the partial observed sequence
        :return: return a length-n array standing for the observed prob
        '''
        # indicator on each site's missingness, 1 for observed and 0 for missing
        #indicator = ~np.isnan(y)
        obs_idx = obs_idx
        T = len(y)

        if len(obs_idx) == 0:
            return np.zeros((T, self.latentDim))

        # Initialize alpha matrix (all zeros)
        alpha = np.zeros((T, self.latentDim))

        # Handle first observed position
        first_obs_idx = obs_idx[0]
        if first_obs_idx == 0:
            alpha[0] = self.initial * self.emission[:, int(y[0])]
        else:
            # If first observation isn't at t=0, need to propagate initial distribution
            power_matrix = np.linalg.matrix_power(self.transition, first_obs_idx)
            alpha[first_obs_idx] = np.dot(self.initial, power_matrix) * self.emission[:, int(y[first_obs_idx])]
            alpha[0] = self.initial

        # Process subsequent observations
        for i in range(1, len(obs_idx)):
            curr_idx = obs_idx[i]
            prev_idx = obs_idx[i - 1]
            gap = curr_idx - prev_idx

            # Compute transition over gap
            if gap == 1:
                trans_matrix = self.transition
            else:
                trans_matrix = np.linalg.matrix_power(self.transition, gap)

            # Update alpha
            alpha[curr_idx] = np.dot(alpha[prev_idx], trans_matrix) * self.emission[:, int(y[curr_idx])]

        return alpha

    def backwardSampler(self, y, alpha, obs_idx):
        '''
        backward sampling based on observation y and its forward probability alpha
        WARNING: Private Function, not exposed to users!
        :param y: observed sequence
        :param alpha: forward probability
        :param obs_idx: indices of observed positions
        :return: sampled latent sequence
        '''
        # initialize the output
        T = len(y)
        self.latentDim = len(self.initial)
        self.obsDim = self.emission.shape[1]
        latentSequence = np.full(T, np.nan)

        if len(obs_idx) == 0:
            # If no observations, only sample initial state
            latentSequence[0] = np.random.choice(self.latentDim, 1, p=self.initial)
            return latentSequence

        # Sample the last observed state
        last_obs_idx = obs_idx[-1]
        w = alpha[last_obs_idx, :] / np.sum(alpha[last_obs_idx, :])
        latentSequence[last_obs_idx] = np.random.choice(self.latentDim, 1, p=w)

        # Backward sample through observed positions
        for i in range(len(obs_idx) - 2, -1, -1):
            curr_idx = obs_idx[i]
            next_idx = obs_idx[i + 1]
            gap = next_idx - curr_idx

            # Compute transition over gap
            if gap == 1:
                trans_matrix = self.transition
            else:
                trans_matrix = np.linalg.matrix_power(self.transition, gap)

            # Sample state
            next_state = latentSequence[next_idx]
            w = trans_matrix[:, int(next_state)] * alpha[curr_idx, :]
            w = w / np.sum(w)
            latentSequence[curr_idx] = np.random.choice(self.latentDim, 1, p=w)

        # If first observation isn't at t=0, sample initial state
        if obs_idx[0] != 0:
            first_obs_idx = obs_idx[0]
            first_state = latentSequence[first_obs_idx]
            power_matrix = np.linalg.matrix_power(self.transition, first_obs_idx)
            w = power_matrix[:, int(first_state)] * alpha[0, :]
            w = w / np.sum(w)
            latentSequence[0] = np.random.choice(self.latentDim, 1, p=w)

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
        self.initialErrorStd = np.std(np.square(self.initialMean-trueInitial))
        self.initialError=np.mean(np.square(self.initialMean-trueInitial))
        self.transitionError=np.mean(np.square(self.transitionMean-trueTransition))
        self.transitionErrorStd = np.std(np.square(self.transitionMean-trueTransition))
        self.emissionError=np.mean(np.square(self.emissionMean-trueEmission))
        self.emissionErrorStd = np.std(np.square(self.emissionMean-trueEmission))
        return self.acc, self.initialError, self.transitionError, self.emissionError, self.initialErrorStd, self.transitionErrorStd, self.emissionErrorStd

    def trivialImputer(self,y,method='mode'):
        '''
        Impute y using trivial methods
        :param y: incomplete sequences
        :param method: Imputing methods. 'mode': Impute with modes. 'forward':Impute forwardly. 'backward': impute backwardly.
        'random': random imputing
        :return:
        '''
        if method=='forward':
            # Forward imputation
            matrix = np.nan_to_num(y, nan=np.nan, copy=True)
            matrix = np.roll(matrix, shift=1, axis=1)
            matrix[:, 0] = np.nan_to_num(matrix[:, 0], nan=np.nan, copy=True)
            matrix[:, 0] = np.roll(matrix[:, 0], shift=1, axis=0)
            matrix[:, 0][np.isnan(matrix[:, 0])] = 0
        elif method=='backward':
            # Backward imputation
            matrix = np.nan_to_num(y, nan=np.nan, copy=True)
            matrix = np.roll(matrix, shift=-1, axis=1)
            matrix[:, -1] = np.nan_to_num(matrix[:, -1], nan=np.nan, copy=True)
            matrix[:, -1] = np.roll(matrix[:, -1], shift=-1, axis=0)
            matrix[:, -1][np.isnan(matrix[:, -1])] = 0
        elif method=='mode':
            # Mode imputation
            matrix = y.copy()
            # Calculate modes for each row and store them in a new array
            row_modes = np.array(
                [ss.mode(row[~np.isnan(row)])[0][0] if not np.all(np.isnan(row)) else np.nan for row in
                 matrix])

            # Create a mask for missing entries
            missing_mask = np.isnan(matrix)

            # Create a broadcasted version of the row modes
            row_modes_broadcasted = np.broadcast_to(row_modes[:, np.newaxis], matrix.shape)

            # Fill missing entries with the modes using the mask
            matrix[missing_mask] = row_modes_broadcasted[missing_mask]
        else:
            # random imputation if can't identify the method variable
            # Create a copy of the matrix to avoid modifying the original
            matrix = np.copy(y)

            # Create a mask of observed entries
            mask = ~np.isnan(matrix)

            # Randomly sample values from observed entries for missing entries
            matrix[np.isnan(matrix)] = np.random.choice(matrix[mask], size=np.isnan(matrix).sum())
        return matrix

    def trivialImputerEvaluation(self,y,drop='random',fraction=0.1):
        '''
        evaluate trivial imputers by dropping some observations and try to predict them
        :param y: data
        :param drop: dropping method. similar to crossValidationScore(), drop randomly by 'random' and drop last by 'forecast'
        :param fraction: percent of dropping
        :return: score of forward imputer, backward imputer, mode imputer and random imputer
        '''
        # get the indices of the observed entries in the test set
        mask = ~np.isnan(y)
        if drop == 'forecast':
            window = int(np.ceil(y.shape[1] * fraction))
            # set all observations in window as missing
            mask[:, -window:] = False
        # if drop can't be specified, treat it as dropping randomly
        else:
            # Get the indices of the non-zero elements (True values)
            nonZeroIndices = np.argwhere(mask)
            # Calculate the number of elements to set to 0 (False)
            numElementsToZero = int(fraction * len(nonZeroIndices))
            # Randomly select non-zero indices to set to 0
            selectedIndices = nonZeroIndices[
                np.random.choice(len(nonZeroIndices), numElementsToZero, replace=False)]
            # Update the mask: set the selected indices to 0 (False)
            mask[selectedIndices[:, 0], selectedIndices[:, 1]] = False
        # start imputing
        methods = ['mode', 'forward', 'backward', 'random']
        scores = {}
        # mask some observations in y according to mask and use this new masked version for evaluation
        masked_y_test = y.copy()
        masked_y_test[mask] = np.nan

        for i in range(len(methods)):
            method = methods[i]
            y_pred = self.trivialImputer(masked_y_test, method=method)
            score = np.sum(np.equal(y, y_pred) * mask) / np.sum(mask)
            scores[method]=score
        return scores

    def summarize(self):
        self.getMean()
        # get posterior variance
        # last 1/3 as posterior draw
        postNum = int(self.postInitial.shape[0] / 3)
        self.initialVar = np.var(self.postInitial[-postNum:], axis=0)
        self.transitionVar = np.var(self.postTransition[-postNum:], axis=0)
        self.emissionVar = np.var(self.postEmission[-postNum:], axis=0)


    def evaluateESS(self):
        """
        Evaluate the performance of the MCMC sampler using stored posterior samples.

        Computes and returns:
          1. Time per 1000 iterations.
          2. Effective Sample Size (ESS) per iteration for each component in each set of parameters.
          3. ESS per second for each component in each set of parameters.

        Posterior samples are expected to be stored in:
          - self.postInitial   (shape: T x init_dim)
          - self.postTransition (shape: T x transition_shape...)
          - self.postEmission   (shape: T x emission_shape...)

        :param total_run_time: Total time in seconds for the T iterations.
        :return: A dictionary containing evaluation metrics for each parameter set.
                 Example structure:
                 {
                   "time_per_1000": float,
                   "Initial": {
                         "abs_ess": numpy.array,
                         "ess_per_iter": numpy.array,
                         "ess_per_sec": numpy.array,
                         "summary": {"min": float, "median": float, "max": float}
                   },
                   "Transition": { ... },
                   "Emission": { ... }
                 }
        """
        # Total number of MCMC iterations.
        total_run_time = self.total_time
        T = self.postInitial.shape[0]
        time_per_1000 = (total_run_time / T) * 1000

        #####################
        # Helper functions. #
        #####################

        def compute_autocorrelation(x):
            """
            Compute the autocorrelation function for a 1D array x.

            Returns an array of autocorrelations for lags 0,1,2,...,len(x)-1.
            """
            n = len(x)
            x = np.asarray(x)
            x = x - np.mean(x)
            # Full autocorrelation (only nonnegative lags).
            result = np.correlate(x, x, mode='full')[n - 1:]
            result = result / np.arange(n, 0, -1)
            # Normalize by lag-0.
            if result[0] == 0:
                return np.zeros_like(result)
            return result / result[0]

        def ess_1d(x):
            """
            Compute the effective sample size (ESS) for a 1D chain x.

            Uses the initial monotone sequence estimator: sums consecutive pairs
            of autocorrelations until a pair becomes negative.
            """
            T_local = len(x)
            acf = compute_autocorrelation(x)
            sum_r = 0.0
            # Sum pairs of autocorrelations: (lag1 + lag2), (lag3 + lag4), ...
            for lag in range(1, T_local - 1, 2):
                pair_sum = acf[lag] + acf[lag + 1]
                if pair_sum < 0:
                    break
                sum_r += pair_sum
            # Guard against degenerate cases.
            if (1 + 2 * sum_r) <= 0:
                return T_local
            return T_local / (1 + 2 * sum_r)

        def compute_ess_metrics(samples):
            """
            Compute ESS metrics for each component of the samples.

            samples: array-like with shape (iterations, parameter_shape...)

            Returns a dictionary with:
              - "abs_ess": ESS for each component (absolute numbers).
              - "ess_per_iter": ESS per iteration.
              - "ess_per_sec": ESS per second.
              - "summary": summary statistics (min, median, max) of the ESS.
            """
            # Compute ESS along the first axis (iterations) for each component.
            abs_ess = np.apply_along_axis(ess_1d, 0, samples)
            ess_per_iter = abs_ess / T
            ess_per_sec = abs_ess / total_run_time
            summary = {
                "min": float(np.min(abs_ess)),
                "median": float(np.median(abs_ess)),
                "max": float(np.max(abs_ess))
            }
            return {
                "abs_ess": abs_ess,
                "ess_per_iter": ess_per_iter,
                "ess_per_sec": ess_per_sec,
                "summary": summary
            }

        #########################
        # Compute the metrics.  #
        #########################
        results = {}
        results["time_per_1000"] = time_per_1000

        # Compute ESS metrics for each set of parameters.
        results["Initial"] = compute_ess_metrics(self.postInitial)
        results["Transition"] = compute_ess_metrics(self.postTransition)
        results["Emission"] = compute_ess_metrics(self.postEmission)

        return results

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
