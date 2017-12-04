import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_n_components = self.min_n_components
        best_score = float("inf")
        num_states = self.min_n_components

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                logL = model.score(self.X, self.lengths)
                n = n_components
                d = len(self.X[0])
                n_parameters = n * n + 2 * n * d - 1
                n_dataPoints = len(self.X)
                BIC_score = -1 * logL + n_parameters * np.log(n_dataPoints)
            except ValueError:
                n_components += 1
                continue

            if best_score > BIC_score:
                best_score = BIC_score
                best_n_components = n_components

            n_components += 1

        return self.base_model(best_n_components)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_n = self.random_state
        best_DIC = float("-inf")
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components, n_iter=1000).fit(self.X, self.lengths)
                original_prob = model.score(self.X, self.lengths)

                sum_prob_others = 0.0
                word_count = 0

                for word in self.words:
                    if word == self.this_word:
                        continue

                    word_count += 1

                    X_other, lengths_other = self.hwords[word]

                    logL = model.score(X_other, lengths_other)
                    sum_prob_others += logL

                avg_prob_others = sum_prob_others / len(word_count)
                DIC = original_prob - avg_prob_others

                if DIC > best_DIC:
                    best_DIC = DIC
                    best_n = n_components

            except:
                pass

        best_model = GaussianHMM(best_n, n_iter=1000).fit(self.X, self.lengths)

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_avg_score = float("-inf")
        sumScore = 0
        best_n = None
        best_model = None
        n_splits = min(len(self.lengths), 3)
        split_method = KFold(n_splits)

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            counter = 0

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                try:
                    counter += 1

                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                    model = GaussianHMM(n_components, n_iter=1000).fit(X_train, lengths_train)
                    logL = model.score(X_test, lengths_test)

                    sumScore += logL
                except:
                    pass

            avgScore = sumScore / counter

            if avgScore > best_avg_score:
                best_avg_score= avgScore
                best_n = n_components

        best_model = GaussianHMM(best_n, n_iter=1000).fit(self.X, self.lengths)

        return best_model
