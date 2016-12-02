"""
Source Code for project of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof.  Zoran Kostic

"""
import os
import sys
import numpy
import scipy.io
import tarfile
import theano
import theano.tensor as T
import cPickle, gzip, numpy


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_data():
    dataset = 'mnist.pkl.gz'
    from six.moves import urllib
    origin = ('http://deeplearning.net/data/mnist/' + dataset)
    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin, dataset)
    f = gzip.open('mnist.pkl.gz', 'rb')
    
    train_set, valid_set, test_set = cPickle.load(f)
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    
    return rval
    