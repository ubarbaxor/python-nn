import numpy as np
np.set_printoptions(linewidth=400)

def sigmoid(x, derivative=False):
    if (derivative==True):
        return x * ( 1-x )

    return 1 / ( 1+np.exp( -x ) )

def linear(x, derivative=False):
    if (derivative == True):
        return 1
    return x

# Model :
# input layer size 4 binary, hidden layer size 4, output layer size 1 int

# known input/outputs for training
# These are binary values
X = np.array([
    [1,1,1,1], # 15
    [1,0,0,1], # 9
    [1,1,0,1], # 13
    [0,1,0,1], # 5
    [0,0,0,1], # 1
    [1,0,1,0], # 10
    [0,1,1,0], # 6
    [0,0,1,0], # 2
    [0,0,1,1], # 3
    [0,1,0,0], # 4
])

y = np.array([
    [15.],
    [9.],
    [13.],
    [5.],
    [1.],
    [10.],
    [6.],
    [2.],
    [3.],
    [4.],
]) / float(16) # Normalize by max value -> 0-16

# seed random for synapses
np.random.seed(42) # because 1337 was already taken

# returns a randomly initiated matrix of weights
# centered in [-1,1]
def synapse(input_size, output_size):
    return (2 * np.random.random((input_size, output_size)) - 1)

def forward(layer, weights, activation=sigmoid):
    return activation(np.dot(layer, weights))

syn0 = synapse(4,16)
syn1 = synapse(16,1)
syn2 = [16]

for i in xrange(100*1000):
    # Compute layers
    l1 = forward(X, syn0) # neuron / hidden layer activation
    l2 = forward(l1, syn1) # output layer
    l3 = forward(l2, syn2, linear) # linear transform to human

    # Error calculation
    l2_error = y - l2 # difference matrix btwn known and computed outputs
    l2_delta = l2_error * sigmoid(l2, derivative=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1, derivative=True)

    # update synapses weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += np.dot(X.T, l1_delta)

    # Print every 10k rounds
    if (i % 1e4 == 0):
        print '=' * 100
        print('Iteration ' + str(i))
        print '/' * 100
        print('Layers :')
        print('Input\n' + str(X))
        print('Output\n' + str(l2))
        print('L3\n' + str(l3))
        print "!" * 100
        print('Errors')
        print('l2\n' + str(l2_error))
        print('l1\n' + str(l1_error))
        print '/\\' * 50
        print('deltas')
        print('l2\n' + str(l2_delta))
        print('l1\n' + str(l1_delta))
        print '=' * 100
print( 'Output after training :\n' + str(l3) )

#TODO : Test the model after training
Xtest = np.array([
    [1,1,1,1],
    [0,0,0,0],
    [0,0,1,1],
    [1,1,0,0],
    [0,1,0,1]
])

Ytest = np.array([
    [15],
    [0],
    [3],
    [12],
    [5]
])

Ztest = forward(forward(forward(Xtest, syn0), syn1), syn2, linear)
Test_Error = Ztest - Ytest
print(str(Ztest) + '\n' + str(Test_Error))
print(str(np.rint(Ztest)))
