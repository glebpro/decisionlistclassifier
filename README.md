# DecisionListClassifier

An implementation of Yarowsky's[[1](#1)] log-likelihood algorithm for decision lists.

```
>>> from classifiers import DecisionListClassifier
>>> dlc = DecisionListClassifier('bass', [('bass', ['loud', 'bass', 'sound']), ('*bass',['large', 'bass', 'fish'])])
>>> dlc[0]
('bass', 'loud_bass', 1.791759469228055)
>>> dlc.predict('bass', ['large', 'bass'])
bass
```

Run `$ python test.py` to see example result + how it works.

## Download
`git clone https://github.com/glebpro/decisionlistclassifier.git`

## Requirements
Python >= 3.4

## License
MIT licensed. See the bundled [LICENSE](/LICENSE) file for more details.

<hr>
###### 1
<i>Yarowsky, D. (1994, June). Decision lists for lexical ambiguity resolution: Application to accent restoration in Spanish and French. In Proceedings of the 32nd annual meeting on Association for Computational Linguistics (pp. 88-95). Association for Computational Linguistics.</i>
