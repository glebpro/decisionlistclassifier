#
#   An implementation of Yarowsky's Decision List Classifier, useful for Word Sense Disambiguation
#
#   @author Gleb Promokhov gxp5819
#

import re
import sys
import argparse
import string
import math
import random
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import LidstoneProbDist
from nltk.stem import WordNetLemmatizer
from nltk.metrics import ConfusionMatrix
from nltk import pos_tag

class DecisionListClassifier(object):
    """
    The decision list is a sorted list of tuples of the following form:
        (sense, feature, log-likelihood)
    sorted by log-likelihood.

    Can be used to make a binary word sense disambiguation.
    """

    def __init__(self, sense_pairs, max_collocation=3):
        """
        Construct a new decision list classifier from a list of tuples
        of the form:
            (sense, [context_tokens])
        where the sense is a unquie identifier of the sense of the target word.
        Example:
            [('bass', ['the','large','bass','swam','down','river']),
             ('*bass', ['the', 'bass', 'was', 'very', 'loud']), ...
            ]

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
        word_senses = set([w[0] for w in sense_pairs])
        if len(word_senses) != 2:
            raise ValueError('There can only be 2 word senses')

        # find majority word sense (can use as basline)
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

        # make the decision list
        dl = self.train(sense_pairs, max_collocation)

        # set word senses
        self.word = word_senses[0]
        self.word_star = word_senses[1]
        self.majority_word = word_sense_majority
        # set the decision list
        self._decision_list = dl

    def report(self):
        print("\nMajority baseline sense label: {}".format(self.result["majority_baseline"]))
        print("\nMajority baseline accuracy: {}%".format(self.result["majority_baseline_percent_correct"]))
        print("\nDecision list classifier accuracy: {}%".format(self.result["percent_correct"]))
        print("\tImprovement over baseline: {}%".format(self.result["percent_correct"]-self.result["majority_baseline_percent_correct"]))
        print("\tMacro precision: {}%".format(round(self.result["macro_precision"]*100, 2)))
        print("\tMacro recall: {}%".format(round(self.result["macro_recall"]*100, 2)))
        print("\t For sense: {}".format(self.word))
        print("\t\t Precision: {}%".format(round(self.result["precision_word"]*100, 2)))
        print("\t\t Recall: {}%".format(round(self.result["recall_word"]*100, 2)))
        print("\t For sense: {}".format(self.word_star))
        print("\t\t Precision: {}%".format(round(self.result["precision_word_star"]*100, 2)))
        print("\t\t Recall: {}%".format(round(self.result["recall_word_star"]*100, 2)))
        print("\nConfusion matrix: \n{}".format(self.result["confusion_matrix"]))
        print("Top 10 Decision Rules:")
        for i in range(0, 10):
            print("\t{}".format(self.decision_list[i]))
        print("\nThree example correctly guessed contexts:")
        for i in range(self.sample_size):
            print("\t{}".format(' '.join(self.result["correct_guess_sample"][i])))
        print("\nThree example incorrectly guessed contexts:")
        for i in range(self.sample_size):
            print("\t{}".format(' '.join(self.result["incorrect_guess_sample"][i])))
        print("")

    def evaluate(self, test_corp):

        # [(sense, [tokens]), ...]
        test_corpus = self.preprocess_corpus(test_corp)

        # baseline testing
        baseline_correct = 0
        self.result["majority_baseline"] = self.majority_baseline
        for row in test_corpus:
            if row[0] == self.majority_baseline:
                baseline_correct += 1
        self.result["majority_baseline_percent_correct"] = round(baseline_correct/len(test_corpus)*100, 2)

        # analyze using model prediction
        guesses = [] # what we guessed
        actual = [] # what acutally was
        correctly_guessed = []
        incorrectly_guessed = []
        for row in test_corpus:
            g = self.predict(row[1])
            guesses.append(g)
            actual.append(row[0])
            if g == row[0]:
                self.result["correct_count"] += 1
                correctly_guessed.append(row[1])
            else:
                self.result["incorrect_count"] += 1
                incorrectly_guessed.append(row[1])

        self.result["correct_guess_sample"] = random.sample(correctly_guessed, self.sample_size)
        self.result["incorrect_guess_sample"] = random.sample(incorrectly_guessed, self.sample_size)
        self.result["percent_correct"] = round((self.result["correct_count"]/len(test_corpus))*100, 2)

        # confusion
        self.result["confusion_matrix"] = ConfusionMatrix(actual, guesses)

        # calculate true/false_positive/negatives for both senses
        true_pos = Counter()
        false_neg = Counter()
        false_pos = Counter()
        for i in [self.word, self.word_star]:
            for j in [self.word, self.word_star]:
                if i == j:
                    true_pos[i] += self.result["confusion_matrix"][i,j]
                else:
                    false_neg[i] += self.result["confusion_matrix"][i,j]
                    false_pos[j] += self.result["confusion_matrix"][i,j]

        # precision
        self.result["precision_word"] = true_pos[self.word] / float(true_pos[self.word]+false_pos[self.word])
        if float(true_pos[self.word_star]+false_pos[self.word_star]) == 0:
            self.result["precision_word_star"] = 0
        else:
            self.result["precision_word_star"] = true_pos[self.word_star] / float(true_pos[self.word_star]+false_pos[self.word_star])

        # recall
        self.result["recall_word"] = true_pos[self.word] / float(true_pos[self.word]+false_neg[self.word])
        self.result["recall_word_star"] = true_pos[self.word_star] / float(true_pos[self.word_star]+false_neg[self.word_star])

        # macros
        self.result["macro_precision"] = (float(self.result["recall_word"]) + float(self.result["recall_word_star"])) / 2.0
        self.result["macro_recall"] = (float(self.result["recall_word"]) + float(self.result["recall_word_star"])) / 2.0

    def predict(self, context_tokens):

        # break apart context tokens into same feature set as trained on
        features = []
        features += [self.collocate(self.word, context_tokens, 0, 0)] # unigrams
        features += [self.collocate(self.word, context_tokens, 1, 0)] # bigrams 1 token left
        features += [self.collocate(self.word, context_tokens, 0, 1)] # bigrams 1 token right
        features += [self.collocate(self.word, context_tokens, 2, 0)] # trigrams 2 token left
        features += [self.collocate(self.word, context_tokens, 1, 1)] # trigrams 1 token left 1 token right
        features += [self.collocate(self.word, context_tokens, 0, 2)] # trigrams 2 token right

        pos = self.parts_of_speech(context_tokens)
        features += [self.collocate(self.word, pos, 0, 0)] # unigrams
        features += [self.collocate(self.word, pos, 1, 0)] # bigrams 1 token left
        features += [self.collocate(self.word, pos, 0, 1)] # bigrams 1 token right
        features += [self.collocate(self.word, pos, 2, 0)] # trigrams 2 token left
        features += [self.collocate(self.word, pos, 1, 1)] # trigrams 1 token left 1 token right
        features += [self.collocate(self.word, pos, 0, 2)] # trigrams 2 token right

        # match with best matching feature to predict the sense
        max_log = 0.0
        sense = self.majority_baseline
        for f in features:
            for rule in self.decision_list:
                if rule[1] == f:
                    if rule[2] > max_log:
                        max_log = rule[2]
                        sense = rule[0]

        return sense

    def train(self, sense_pairs, max_collocation):
        """
        Returns a decision list using Yarowsky's log-likelihood decision list
        algorithm.

        :type sense_pairs: A list
        :param sense_pairs: A list of tuples containing word sense and context
            tokens.
        :type max_collocation: An integer
        :param max_collocation: When training the model, use this value
            as the maximum number of tokens to generate ngrams for.
            Example: 0 means only train with unigrams, 1 means train with
            bigrams 1 left and right of target word and unigrams, etc...
        """
        # matrix of features * sense and their counts
        # fc[feature][sense] => count
        freqs = ConditionalFreqDist()

        # count stuff
        for pair in sense_pairs:

            freqs[feature][sense] += 1

            # count the n-grams
            self.add_feature(freqs, row[0], self.collocate(self.word, row[1], 0, 0)) # unigrams
            self.add_feature(freqs, row[0], self.collocate(self.word, row[1], 1, 0)) # bigrams 1 token left
            self.add_feature(freqs, row[0], self.collocate(self.word, row[1], 0, 1)) # bigrams 1 token right
            self.add_feature(freqs, row[0], self.collocate(self.word, row[1], 2, 0)) # trigrams 2 token left
            self.add_feature(freqs, row[0], self.collocate(self.word, row[1], 1, 1)) # trigrams 1 token left 1 token right
            self.add_feature(freqs, row[0], self.collocate(self.word, row[1], 0, 2)) # trigrams 2 token right

            # count parts of speach grams
            pos = self.parts_of_speech(row[1])
            self.add_feature(freqs, row[0], self.collocate(self.word, pos, 0, 0)) # unigrams
            self.add_feature(freqs, row[0], self.collocate(self.word, pos, 1, 0)) # bigrams 1 token left
            self.add_feature(freqs, row[0], self.collocate(self.word, pos, 0, 1)) # bigrams 1 token right
            self.add_feature(freqs, row[0], self.collocate(self.word, pos, 2, 0)) # trigrams 2 token left
            self.add_feature(freqs, row[0], self.collocate(self.word, pos, 1, 1)) # trigrams 1 token left 1 token right
            self.add_feature(freqs, row[0], self.collocate(self.word, pos, 0, 2)) # trigrams 2 token right

        # data for baseline
        self.word_count_total = self.word_count + self.word_star_count
        word_pr = self.word_count / self.word_count_total
        word_star_pr = self.word_star_count / self.word_count_total
        if word_pr > word_star_pr:
            self.majority_baseline = self.word
        else:
            self.majority_baseline = self.word_star

        # smooth out model
        probs = ConditionalProbDist(freqs, LidstoneProbDist, 0.2)

        # make the decision list
        for feature in probs.conditions():
            self.add_to_decision_list(probs, feature)
        self.decision_list.sort(key=lambda row: row[2], reverse=True) # sort by log-likelihood

    def add_to_decision_list(self, probs, feature):
        prob = probs[feature].prob(self.word) # Pr(sense1|feature)
        prob_star = probs[feature].prob(self.word_star) # Pr(sense2|feature)
        d = math.log(prob/prob_star)
        if d == 0:
            self.decision_list.append(('', feature, 0))
        else:
            sense = self.word if d > 0 else self.word_star # can tell what sense feature is for be -/+ value
            self.decision_list.append((sense, feature, abs(d)))

    def add_feature(self, freqs, sense, feature):

    def parts_of_speech(self, tokens):
        b = tokens.index(self.word)
        c = [w[1] for w in pos_tag(tokens)]
        c[b] = self.word
        return c

    def generate_ngrams(self, word, tokens, max_collocations):
        """

        :raise ValueError: if ``word`` not in ``tokens``.
        """
        ngrams = []
        w = tokens.index(word)
        for left in range(wmax_collocations):
            ngrams.append('_'.join(tokens[p-left:p] + [word] + tokens[p:p+right]))
    def collocate(self, word, tokens, left, right):
        p = tokens.index(word)
        return '_'.join(tokens[p-left:p] + [word] + tokens[p:p+right])

    # format corpus
    # sets self.word
    # sets self.senses
    # returns: [(*?word, [tokens]), ...]
    def preprocess_corpus(self, corpus):

        c = []
        stops = set(stopwords.words('english'))
        corpus = [l for l in corpus.split("\n") if len(l)]
        senses = []
        lm = WordNetLemmatizer()

        for line in corpus:

            line = line.strip()

            # (sense, tokens)
            s = line.index(":")
            line = (line[:s], line[s:])

            # tokenize
            line = (line[0], word_tokenize(line[1]))

            # lemmatize
            #line = (line[0], [lm.lemmatize(w) for w in line[1]])

            # remove stop words, they muck up semantic context
            line = (line[0], [w for w in line[1] if w not in stops])

            # remove punctuation
            line = (line[0], [re.sub('['+string.punctuation+']', '', w) for w in line[1]])

            # remove left overs from contractions ( " they're -> 're ") and other short words
            line = (line[0], [w for w in line[1] if len(w) > 2])

            # lower
            line = (line[0], [w.lower() for w in line[1]])

            senses.append(line[0])
            c.append(line)

        self.word = re.sub('\*', '', c[0][0])
        self.word_star = '*'+self.word
        return c

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training_data", help="training data")
    parser.add_argument("-s", "--testing_data", help="testing data")
    args = parser.parse_args()

    training_corpus = open(args.training_data, 'r').read()
    testing_corpus = open(args.testing_data, 'r').read()

    classifier = DecisionListClassifier(training_corpus)

    classifier.evaluate(testing_corpus)

    classifier.report()

if __name__ == "__main__":
    main()
