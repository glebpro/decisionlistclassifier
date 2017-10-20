#
#   Some utilities and tests for classifiers.DecisionListClassifier
#
#   @author Gleb Promokhov gleb.promokhov@gmail.com
#

from classifiers.DecisionListClassifier import DecisionListClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.metrics import ConfusionMatrix
from collections import Counter
import re
import string
import random

def preprocess_corpus(corpus):
    c = []
    stops = set(stopwords.words('english'))
    corpus = [l for l in corpus.split("\n") if len(l)]
    senses = []

    for line in corpus:

        line = line.strip()

        # (sense, tokens)
        s = line.index(":")
        line = (line[:s], line[s:])

        # tokenize
        line = (line[0], word_tokenize(line[1]))

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

    return c

def evaluate(word, dlc, test_corpus, sample_size=3):

    result = {}

    # do some baseline counting
    word_senses = list(set([w[0] for w in test_corpus]))
    word_sense, word_sense_star = word_senses[0], word_senses[1]
    word_sense_count = 0
    word_sense_star_count = 0
    word_sense_majority = ''
    for w in test_corpus:
        if w[0] == word_sense:
            word_sense_count +=1
        else:
            word_sense_star_count +=1

    if word_sense_count/(word_sense_star_count+word_sense_count) > \
        word_sense_star_count/(word_sense_star_count+word_sense_count) :
        word_sense_majority = word_senses[0]
    else:
        word_sense_majority = word_senses[1]

    # baseline testing
    baseline_correct = 0
    result["majority_baseline"] = word_sense_majority
    for row in test_corpus:
        if row[0] == word_sense_majority:
            baseline_correct += 1
    result["majority_baseline_percent_correct"] = round(baseline_correct/len(test_corpus)*100, 2)

    # analyze using model prediction
    result["correct_count"], result["incorrect_count"] = 0, 0
    guesses = [] # what we guessed
    actual = [] # what acutally was
    correctly_guessed = []
    incorrectly_guessed = []
    for row in test_corpus:
        g = dlc.predict(word, row[1])
        guesses.append(g)
        actual.append(row[0])
        if g == row[0]:
            result["correct_count"] += 1
            correctly_guessed.append(row[1])
        else:
            result["incorrect_count"] += 1
            incorrectly_guessed.append(row[1])

    result["correct_guess_sample"] = random.sample(correctly_guessed, sample_size)
    result["incorrect_guess_sample"] = random.sample(incorrectly_guessed, sample_size)
    result["percent_correct"] = round((result["correct_count"]/len(test_corpus))*100, 2)

    # confusion
    result["confusion_matrix"] = ConfusionMatrix(actual, guesses)

    # calculate true/false_positive/negatives for both senses
    true_pos = Counter()
    false_neg = Counter()
    false_pos = Counter()
    for i in [word_sense, word_sense_star]:
        for j in [word_sense, word_sense_star]:
            if i == j:
                true_pos[i] += result["confusion_matrix"][i,j]
            else:
                false_neg[i] += result["confusion_matrix"][i,j]
                false_pos[j] += result["confusion_matrix"][i,j]

    # # precision
    # result["precision_word"] = true_pos[word_sense] / float(true_pos[word_sense]+false_pos[word_sense])
    # if float(true_pos[word_sense_star]+false_pos[word_sense_star]) == 0:
    #     result["precision_word_star"] = 0
    # else:
    #     result["precision_word_star"] = true_pos[word_sense_star] / float(true_pos[word_sense_star]+false_pos[word_sense_star])
    #
    # # recall
    # result["recall_word"] = true_pos[word_sense] / float(true_pos[word_sense]+false_neg[word_sense])
    # result["recall_word_star"] = true_pos[word_sense_star] / float(true_pos[word_sense_star]+false_neg[word_sense_star])
    #
    # # macros
    # result["macro_precision"] = (float(result["recall_word"]) + float(result["recall_word_star"])) / 2.0
    # result["macro_recall"] = (float(result["recall_word"]) + float(result["recall_word_star"])) / 2.0
    #
    # result["word_sense"] = word_sense
    # result["word_sense_star"] = word_sense_star

    return result

def report(dlc, result, sample_size=3):
    print("\nMajority baseline sense label: {}".format(result["majority_baseline"]))
    print("\nMajority baseline accuracy: {}%".format(result["majority_baseline_percent_correct"]))
    print("\nDecision list classifier accuracy: {}%".format(result["percent_correct"]))
    print("\tImprovement over baseline: {}%".format(result["percent_correct"]-result["majority_baseline_percent_correct"]))
    # print("\tMacro precision: {}%".format(round(result["macro_precision"]*100, 2)))
    # print("\tMacro recall: {}%".format(round(result["macro_recall"]*100, 2)))
    # print("\t For sense: {}".format(result["word_sense"]))
    # print("\t\t Precision: {}%".format(round(result["precision_word"]*100, 2)))
    # print("\t\t Recall: {}%".format(round(result["recall_word"]*100, 2)))
    # print("\t For sense: {}".format(result["word_sense_star"]))
    # print("\t\t Precision: {}%".format(round(result["precision_word_star"]*100, 2)))
    # print("\t\t Recall: {}%".format(round(result["recall_word_star"]*100, 2)))
    print("\nConfusion matrix: \n{}".format(result["confusion_matrix"]))
    print("Top 10 Decision Rules:")
    for i in range(0, 10):
        print("\t{}".format(dlc[i]))
    print("\nThree example correctly guessed contexts:")
    for i in range(sample_size):
        print("\t{}".format(' '.join(result["correct_guess_sample"][i])))
    print("\nThree example incorrectly guessed contexts:")
    for i in range(sample_size):
        print("\t{}".format(' '.join(result["incorrect_guess_sample"][i])))
    print("")

def test0():
    dlc = DecisionListClassifier('bass', [('bass', ['loud', 'bass', 'sound']), ('*bass',['large', 'bass', 'fish'])])
    print(dlc[0])
    for d in dlc:
        print(d)
    print(dlc.predict('bass', ['loud', 'bass']))

def test1():
    train = [('bass', ['stephan', 'weidner', 'composer', 'bass', 'player', 'boehse', 'onkelz']), \
            ('bass', ['valued', '250000', 'another', 'double', 'bass', 'trapped', 'room']), \
            ('*bass', ['portion', 'shrimp', 'mussels', 'sea', 'bass', 'whatnot', 'spicy'] \
            )]
    test = [('bass', ['frantic', 'drums', 'pulsing', 'bass', 'unleashed', 'roni', 'size']), \
            ('bass', ['timbres', 'single', 'muddy', 'bass', 'line', 'the', 'treble', 'fared']), \
            ('*bass', ['instead', 'playing', 'bass', 'pick', 'avery']
            )]

    dlc = DecisionListClassifier('bass', train)
    result = evaluate('bass', dlc, test, 0)

    report(dlc, result)

def test2():

    train = preprocess_corpus(open('data/bass.trn', 'r').read())
    test = preprocess_corpus(open('data/bass.tst', 'r').read())

    dlc = DecisionListClassifier('bass', train)
    result = evaluate('bass', dlc, test)

    report(dlc, result)

if __name__ == "__main__":
    # test0()
    # test1()
    test2()
