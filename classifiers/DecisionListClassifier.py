#
#   An implementation of Yarowsky's Decision List Classifier, useful for Word Sense Disambiguation
#
#   @author Gleb Promokhov gleb.promokhov@gmail.com
#

import re
import string
import math
from nltk.probability import (ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist)
from nltk import pos_tag

class DecisionListClassifier(object):
    """
    The decision list is a list of the following form:
        [(sense, feature, log-likelihood), ...]
    sorted by log-likelihood.

    Sense is one of two word senses.
    Seature are collocated tokens joined by _. Currently done with collocated
        words and collocated parts of speech.
    Log-likelihood is how likely that feature maps to the sense.

    Can be used to make a binary word sense disambiguation.
    """

    def __init__(self, target_word, sense_pairs, max_collocation=3):
        """
        Construct a new decision list classifier from a list of tuples
        of the form:
            (sense, [context_tokens])
        where the sense is a unquie identifier of the sense of the target word.
        Example:
            [('bass', ['the','large','bass','swam','down','river']),
             ('*bass', ['the', 'bass', 'was', 'very', 'loud']), ...
            ]

        :type target_word: a string
        :param target_word: word do model word senses for
        :type sense_pairs: A list
        :param sense_pairs: A list of word sense and context words
            in the above form.
        :type max_collocation: An integer
        :param max_collocation: When training the model, use this value
            as the maximum number of tokens to generate ngrams for.
            Example: 0 means only train with unigrams, 1 means train with
            bigrams 1 left and right of target word and unigrams, etc...
        :raise ValueError: If there is anything but 2 word senses in
            ``sense_pair``.
        """

        # check number of word senses
        word_senses = list(set([w[0] for w in sense_pairs]))
        if len(word_senses) != 2:
            raise ValueError('There can only be 2 word senses')

        # find majority word sense (can use as baseline)
        word_sense_count = 0
        word_sense_star_count = 0
        word_sense_majority = ''
        for w in sense_pairs:
            if w == word_senses[0]:
                word_sense_count +=1
            else:
                word_sense_star_count +=1

        if word_sense_count/(word_sense_star_count+word_sense_count) > \
            word_sense_star_count/(word_sense_star_count+word_sense_count) :
            word_sense_majority = word_senses[0]
        else:
            word_sense_majority = word_senses[1]

        # set word senses
        self.word = target_word
        self.word_sense = word_senses[0]
        self.word_sense_star = word_senses[1]
        self.majority_word_sense = word_sense_majority
        # make the decision list
        dl = self.train(target_word, sense_pairs, max_collocation)
        # set the decision list
        self._decision_list = dl
        # evaluation result object
        self.evaluation_result = {}

    def __repr__(self):
        return '<DecisionListClassifier: %s rules>' % (len(self._decision_list))

    def __getitem__(self, i):
        return self._decision_list[i]

    def predict(self, target_word, tokens, max_collocation=3):
        """
        Predict word sense of ``target_word`` in ``tokens``.

        :type target_word: a string
        :params target_word: target word to predict sense of
        :type tokens: a list
        :params tokens: tokens to predict from
        :type max_collocation: an integer
        :params max_collocation: max collocation distance
        :return sense: the predicted word sense
        """
        # generate features for given tokens identically to training features
        features = []
        features += self.generate_collocations(target_word, tokens, max_collocation)

        pos = self.parts_of_speech(target_word, tokens)
        for g in self.generate_collocations(target_word, pos, max_collocation):
            if g != target_word: #TODO: a double count
                features += [g]

        # get maximum sense for feature
        max_log = 0.0
        sense = self.majority_word_sense
        for f in features:
            for rule in self._decision_list:
                if rule[1] == f:
                    if rule[2] > max_log:
                        max_log = rule[2]
                        sense = rule[0]

        return sense

    def train(self, target_word, sense_pairs, max_collocation):
        """
        Returns a decision list using Yarowsky's log-likelihood decision list
        algorithm.

        :type target_word: a string:
        :param target_word: word to model word senses for
        :type sense_pairs: A list
        :param sense_pairs: A list of tuples containing word sense and context
            tokens.
        :type max_collocation: An integer
        :param max_collocation: When training the model, use this value
            as the max collocation distance.
            Example: 0 means only train with unigrams, 1 means train with
            bigrams 1 left and right of target word and unigrams, etc...
        :return dl: the decision list
        """

        # # matrix of features * sense and their counts
        # # fc[feature][sense] => count
        freqs = ConditionalFreqDist()

        # count stuff
        for pair in sense_pairs:

            # ngrams
            for g in self.generate_collocations(target_word, pair[1], max_collocation):
                freqs[g][pair[0]] += 1

            # parts of speech ngrams
            pos = self.parts_of_speech(target_word, pair[1])
            for g in self.generate_collocations(target_word, pos, max_collocation):
                if g != self.word: #TODO: a double count
                    freqs[g][pair[0]] += 1

        # smooth out model
        probs = ConditionalProbDist(freqs, LidstoneProbDist, 0.2)

        # make the decision list
        dl = []
        for feature in probs.conditions():
            self.add_to_decision_list(dl, probs, feature)

        # sort by log-likelihood
        dl.sort(key=lambda r: r[2], reverse=True)

        return dl

    def add_to_decision_list(self, dl, probs, feature):
        """
        Calculate log-likelihood of each word sense given a `feature``
        and append to ``dl``.

        :type dl: a list
        :param dl: decision list of format [(sense, feature, log-likelihood), ...]
        :type probs: nltk.probability.ConditionalProbDist
        :param probs: prob dist of collocations
        :type feature: a string
        :param feature: a collocation
        :return :
        """
        prob = probs[feature].prob(self.word_sense) # Pr(sense1|feature)
        prob_star = probs[feature].prob(self.word_sense_star) # Pr(sense2|feature)
        d = math.log(prob/prob_star)
        if d == 0:
            dl.append((self.majority_word_sense, feature, 0))
        else:
            # can tell what sense feature is for be +/- value
            sense = self.word if d > 0 else self.word_sense_star
            dl.append((sense, feature, abs(d)))

    def parts_of_speech(self, target_word, tokens):
        """
        Convert word tokens to parts of speech tokens. Keep ``target_word``.

        :type target_word: a string
        :param target_word: target word
        :type tokens: a list of strings
        :param tokens: tokens to convert
        :return pos: list of parts of speech tag except ``target_word``.
        """
        b = tokens.index(target_word)
        pos = [w[1] for w in pos_tag(tokens)]
        pos[b] = target_word
        return pos

    #TODO: add flag to genrate list of lists, (need to join to prevent duplicates, how to prevent duplicates lists?)
    #NOTE: could also just use nlkt.utils.ngrams
    #NOTE: dynamic programming matrix??
    def generate_collocations(self, word, tokens, max_c):
        """
        Generate all possible ngrams of max_collocation distance around target
        word.
            >>> generate_collocations('bass', ['the', 'large', 'bass', 'went', 'down'], 2)
            ['the_large_bass', 'large_bass_went', ... 'bass_went','bass']

        :type word: a string
        :param word: target word
        :type tokens: a list
        :param tokens: tokens to make ngrams
        :type max_c: an int
        :param max_c: span of ngrams to make
        :raise ValueError: if ``word`` not in ``tokens``
        """
        def ngrams_h(c, l, r, acc):
            if c == 0:
                return acc
            acc.append('_'.join(tokens[l:r]))
            return ngrams_h(c-1, l+1, r+1, acc)

        w = tokens.index(word) #ValueError
        ngrams = []
        for c in range(max_c, 0, -1):
            ngrams = ngrams_h(c+1, w-c, w+1, [])

        ngrams += [word]
        return ngrams

__all__ = ['DecisionListClassifier']
