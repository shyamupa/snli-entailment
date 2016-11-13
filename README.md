Implementations of a attention model for entailment from [this paper](http://arxiv.org/abs/1509.06664) in keras and tensorflow.

Compatible with keras v1.0.6 and tensorflow 0.11.0rc2

I implemented the model to learn the APIs for keras and tensorflow, so I have not really tuned on the performance. The models implemented in keras is a little different, as keras does not expose a method to set a LSTMs state.

To train,

* Download [snli dataset](http://nlp.stanford.edu/projects/snli/).
* Create train, dev, test files with tab separated text, hypothesis and label (example file train10.txt). You can find some snippet in `reader.py` for this, if you are lazy.
* Train by either running,

```
python amodel.py -train <TRAIN> -dev <DEV> -test <TEST>
```
for using the keras implementation, or 
```
python tf_model.py -train <TRAIN> -dev <DEV> -test <TEST>
```
for using the tensorflow implementation. Look at the `get_params()` method in both scripts to see how to specify different parameters.


Log is written out in *.log file with callback for accuracy.

For comments, improvements, bug-reports and suggestions for tuning, email shyamupa@gmail.com