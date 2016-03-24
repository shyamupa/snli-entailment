import sys
import re
import numpy as np
import argparse
import random
# from keras import backend as K
def tokenize(sent):
    '''
    data_reader.tokenize('a#b')
    ['a', '#', 'b']
    '''
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]

def map_to_idx(x, vocab):
    '''
    x is a sequence of tokens
    '''
    # 0 is for UNK
    return [ vocab[w] if w in vocab else 0 for w in x  ]

def map_to_txt(x,vocab):
    textify=map_to_idx(x,inverse_map(vocab))
    return ' '.join(textify)
    
def inverse_map(vocab):
    return {v: k for k, v in vocab.items()}

def inverse_ids2txt(X_inp,Y_inp,vocabx,vocaby,outp=None):
    '''
    takes x,y int seqs and maps them back to strings
    '''
    inv_map_x = inverse_map(vocabx)
    inv_map_y = inverse_map(vocaby)
    if outp:
        for x,y,z in zip(X_inp,Y_inp,outp):
            print(' '.join(map_to_idx(x,inv_map_x)))
            print(' '.join(map_to_idx(y,inv_map_y)))
            print(z)
    else:
        for x,y in zip(X_inp,Y_inp):
            print(' '.join(map_to_idx(x,inv_map_x)))
            print(' '.join(map_to_idx(y,inv_map_y)))


def create_train_examples(X, Y, yspace, num=-1, balanced=True):
    '''
    :param X: X seq
    :param Y: Y seq
    :param yspace: from which to sample
    :param num: how many negs, -1 means all of it
    :return: x,y,z such that if x,y in X,Y then z=1 else 0
    '''
    X_inp = []
    Y_inp = []
    outp = []
    for x, y in zip(X, Y):
        neg_samples=yspace[:]  # copy
        neg_samples.remove(y)
        if num == -1:
            pass
        else:
            neg_samples=[i for i in random.sample(neg_samples,num)]

        if not balanced:
            X_inp.append(x)
            Y_inp.append(y)
            outp.append([1.0, 0.0])

        for yn in neg_samples:
            if balanced:
                X_inp.append(x)
                Y_inp.append(y)
                outp.append([1.0, 0.0])
            X_inp.append(x)
            Y_inp.append(yn)
            outp.append([0.0, 1.0])

    return X_inp, Y_inp, outp

# def load_word2vec_embeddings(vocab_dim,index_dict,word_vectors,output_dim):
#     vocab_dim = 300 # dimensionality of your word vectors
#     n_symbols = len(index_dict) + 1 # adding 1 to account for 0th index (for masking)
#     embedding_weights = np.zeros((n_symbols+1,vocab_dim))
#     for word,index in index_dict.items():
#         embedding_weights[index,:] = word_vectors[word]
#
#     return Embedding(output_dim=output_dim, input_dim=n_symbols + 1, mask_zero=True, weights=[embedding_weights]) # note you have to put embedding weights in a list by convention

def check_layer_output_shape(layer, input_data):
    ndim = len(input_data.shape)
    layer.input = K.placeholder(ndim=ndim)
    layer.set_input_shape(input_data.shape)
    expected_output_shape = layer.output_shape[1:]

    function = K.function([layer.input], [layer.get_output()])
    output = function([input_data])[0]
    print(output.shape,expected_output_shape)
    assert output.shape[1:] == expected_output_shape

def mytest():
    input_data = np.random.random((10, 142, 200))
    Y = np.random.random((10, 142, 200))
    alpha = np.random.random((10, 142))
    print(input_data.shape)
    # layer = Reshape(dims=(2, 3))
    # layer = Lambda(get_H_n, output_shape=(200,))
    # Y = Layer()
    # alpha= Layer()
    # Y.set_input_shape((None,142,200))
    # alpha.set_input_shape((None,142))
    # ll=Merge([Y,alpha],mode='join')
    # layer=Lambda(get_R, output_shape=(200,1))
    # layer.set_previous(ll)
    # print(layer.input)
    # func = K.function([layer.input], [layer.get_output()])
    # layer=Lambda(get_Y, output_shape=(110, 200))
    # check_layer_output_shape(layer, input_data)
    sys.exit(0)

def show_weights(model,node_name,indices=[0]):
    Wr=model.nodes[node_name].get_weights()
    for i in indices:
        print(Wr[i][0:5])


def show_output(model,node_name,input_data_dict):
    lout= K.function([model.inputs[i].input for i in model.input_order],
                     [model.nodes[node_name].get_output(train=False)])
    output= lout([input_data_dict[i] for i in model.input_order])[0]
    print('input', input_data_dict['input'][0][0:10])
    print(node_name, output[0][0:5])
    print('input', input_data_dict['input'][1][0:10])
    print(node_name, output[1][0:5])


def categorize(ll):
    new_y_train = []
    for y in ll:
        if y == 1:
            new_y_train += [[0, 1]]
        else:
            new_y_train += [[1, 0]]
    return np.asarray(new_y_train)

if __name__=="__main__":
    pass
