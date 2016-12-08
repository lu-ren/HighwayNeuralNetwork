from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy
import scipy.io

import theano
import theano.tensor as T


from layer import HighwayLayer, HiddenLayer, LogisticRegression
from project_utils import load_data

class myHighway(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, n_layers, activation, Highway):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        self.layer0 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation
        )
        
        #self.n_layers = n_layers
        self.hidden_list = [self.layer0]
        
        for i in xrange(n_layers -1):
            if Highway is False:
                x = HiddenLayer(
                    rng=rng,
                    input=self.hidden_list[-1].output,
                    n_in=n_hidden,
                    n_out=n_hidden,
                    activation=activation
                )
            else :
                x = HighwayLayer(
                    rng=rng,
                    input=self.hidden_list[-1].output,
                    n_in=n_hidden,
                    n_out=n_hidden,
                    activation=activation
                )
            self.hidden_list.append(x)
        
        
        self.logRegressionLayer = LogisticRegression(
            input=self.hidden_list[-1].output,
            n_in=n_hidden,
            n_out=n_out
        )
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.logRegressionLayer.W).sum()
        for x in self.hidden_list:
            self.L1 += abs(x.W).sum()
        
        

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = abs(self.logRegressionLayer.W **2).sum()
        for x in self.hidden_list:
            self.L2_sqr += abs(x.W **2).sum()
        

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        
        #self.crossent = self.logRegressionLayer.cross_entropy

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = []
        
        for x in self.hidden_list:
            self.params += x.params
        self.params += self.logRegressionLayer.params
        
        # keep track of model input
        self.input = input






def test_network(n_layers, activation = T.tanh, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             batch_size=20, n_hidden=500, verbose=False, Highway = False):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # load the dataset; download the dataset if it is not present
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    #rng = numpy.random.RandomState(277164956)
    rng = numpy.random.RandomState(100)
    """
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_out=10,
    )
    """
    #TODO: use your MLP and comment out the classifier object above
    classifier = myHighway(
        rng=rng,
        input=x,
        n_in=784,
        n_hidden=n_hidden,
        n_out=10,
        n_layers = n_layers,
        activation = activation,
        Highway = Highway
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
   
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        #outputs=classifier.crossent(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        },
        allow_input_downcast=True
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        #outputs=classifier.crossent(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        },
        allow_input_downcast=True
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    """
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    """
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    lr = T.fscalar('lr')
    
    momentum =theano.shared(numpy.cast[theano.config.floatX](0.7683542317733868), name='momentum')
    updates = []
    for param, gparam in  zip(classifier.params, gparams):
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - lr*param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))
    
    
    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index, lr],
        outputs=[cost, classifier.negative_log_likelihood(y)],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        allow_input_downcast=True
    )
    """
    train_loss = theano.function(
        inputs=[index,lr],
        outputs=classifier.negative_log_likelihood(y),
        updates = updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        allow_input_downcast=True
    )
    """
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    #patience = 10000  # look as this many examples regardless
    #patience_increase = 3 
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_avg_cost = numpy.inf
    best_training_loss = numpy.inf
    avg_training_loss = numpy.zeros(n_train_batches)
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    train_matrix = numpy.zeros(n_epochs)
    val_matrix = numpy.zeros(n_epochs)
    test_matrix = numpy.zeros(n_epochs)
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #if(epoch > 1):
            #learning_rate = numpy.exp(-0.9839659844716131)*learning_rate
            #if(learning_rate < 0.00001):
            #    learning_rate = 0.00001
        #print(learning_rate)
        for minibatch_index in range(n_train_batches):
            
            #average instead of lowest? 
            minibatch_avg_cost, avg_training_loss[minibatch_index] = train_model(minibatch_index, learning_rate)
            #if(minibatch_avg_cost < best_avg_cost):
            #    best_avg_cost = minibatch_avg_cost
            #    best_training_loss = train_loss(minibatch_index, learning_rate)
            #train_matrix[epoch-1] = best_training_loss
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                train_matrix[epoch-1] = numpy.mean(avg_training_loss)
                print(train_matrix[epoch-1])
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                val_matrix[epoch-1] = this_validation_loss
                if verbose:
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                
                test_matrix[epoch -1] = test_score
                
            #if patience <= iter:
                #done_looping = True
                #break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
    numpy.savetxt("train" + str(end_time) + ".csv", train_matrix, delimiter=",")
    numpy.savetxt("validation" + str(end_time) + ".csv", val_matrix, delimiter=",")
    numpy.savetxt("test" + str(end_time) + ".csv", test_matrix, delimiter=",")
    
    return train_matrix, val_matrix, test_matrix

if __name__ == '__main__':
    
#test_network(n_layers, activation=T.Tanh, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
#            batch_size=20, n_hidden=500, verbose=False, Highway=False)0.1776272776349125
    test_network(n_layers = 10, activation = T.nnet.relu, learning_rate=0.01, L2_reg=0.0001, n_epochs = 400, batch_size=20, n_hidden=50, verbose=True, Highway = True)
    #test_network(n_layers = 20, n_epochs = 400, batch_size=20, n_hidden=50, verbose=True,
    #                                  Highway = True)
    #test_rnnslu2()
