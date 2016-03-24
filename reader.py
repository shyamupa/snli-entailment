# coding: utf-8
import json
from myutils import *
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

def get_data(data,vocab):
    for d in data:
        prem=map_to_idx(tokenize(d["sentence1"]),vocab)
        hyp=map_to_idx(tokenize(d["sentence2"]),vocab)
        label=d["gold_label"]
        yield prem, hyp , label

def load_data(train,vocab,labels={'neutral':0,'entailment':1,'contradiction':2}):
    X,Y,Z=[],[],[]
    for p,h,l in train:
        p=map_to_idx(tokenize(p),vocab)
        h=map_to_idx(tokenize(h),vocab)
        # print 'P:',map_to_txt(p,vocab)
        # print 'H:',map_to_txt(h,vocab)
        # print p+" DELIMITER "+h
        # ph=map_to_idx(tokenize(p+" delimiter "+h),vocab)
        # print 'PH:',map_to_txt(ph,vocab)
        # print 'L:',l
        if l in labels:         # get rid of '-'
            X+=[p]
            Y+=[h]
            Z+=[labels[l]]
    return X,Y,Z


def get_vocab(data):
    vocab=Counter()
    for ex in data:
        tokens=tokenize(ex[0])
        tokens+=tokenize(ex[1])
        vocab.update(tokens)
    lst = ["unk", "delimiter"] + [ x for x, y in vocab.iteritems() if y > 0]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    return vocab

def convert2simple(data,out):
    '''
    get the good stuff out of json into a tsv file
    '''
    for d in data:
        print>>out, d["sentence1"]+"\t"+d["sentence2"]+"\t"+d["gold_label"]
    out.close()


if __name__=="__main__":

    train=[l.strip().split('\t') for l in open('train.txt')][:20000]
    dev=[l.strip().split('\t') for l in open('dev.txt')]
    test=[l.strip().split('\t') for l in open('test.txt')]
    labels={'contradiction':-1,'neutral':0,'entailment':1}

    vocab=get_vocab(train)
    X_train,Y_train,Z_train=load_data(train,vocab)
    X_dev,Y_dev,Z_dev=load_data(dev,vocab)
    print len(X_train),X_train[0]
    print len(X_dev),X_dev[0]
