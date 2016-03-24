Implementation of a attention model for entailment (for fun with keras) from [this paper](http://arxiv.org/abs/1509.06664).
The code is research quality -- which means there is no quality. It works, but somewhat poorly than the numbers in the paper.

To train,

* Download [snli dataset](http://nlp.stanford.edu/projects/snli/).
* Create train, dev, test files with tab separated text, hypothesis and label (example file train10.txt). You can find some snippet in `reader.py` for this, if you are lazy.
* Train!

You should be able to get >70% test and dev accuracy (I did no tuning).

Log is written out in *.log file with callback for accuracy.

For comments, improvements, bug-reports and suggestions for tuning, email shyamupa@gmail.com