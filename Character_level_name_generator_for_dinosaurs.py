import numpy as np
from utils import *
import random
import pprint

# Dataset and Preprocessing

# Read the dataset of dinosaur names.
# Create a list of unique characters (such as a-z)
# Compute the dataset and vocabulary size.

data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

# The characters are a-z (26 characters) plus the "\n" (or newline character).
# The newline character "\n" plays a role similar to the `<EOS>` (or "End of sentence") token.
# char_to_ix: Create a python dictionary (i.e., a hash table) to map each character to an index from 0-26.
# ix_to_char: Create a second python dictionary that maps each index back to the corresponding character.

chars = sorted(chars)
print(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ix_to_char)


# Overview of the model
#
# The model has the following structure:
#
# - Initialize parameters
# - Run the optimization loop
#     - Forward propagation to compute the loss function
#     - Backward propagation to compute the gradients with respect to the loss function
#     - Clip the gradients to avoid exploding gradients
#     - Using the gradients, update your parameters with the gradient descent update rule.
# - Return the learned parameters
#
# * At each time-step, the RNN tries to predict what is the next character given the previous characters.

# Building blocks of the model
#
# Gradient clipping: to avoid exploding gradients
# Sampling: a technique used to generate characters

# Clipping the gradients in the optimization loop
#
#
#
# #### gradient clipping
# Implement a function `clip` that takes in a dictionary of gradients and returns a clipped version of gradients if needed.
#

def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out= gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients

# Sampling

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []

    # idx is the index of the one-hot vector x that is set to 1
    idx = -1

    # Loop over time-steps t. At each time-step:
    # sample a character from a probability distribution
    # and append its index (`idx`) to the list "indices".
    # We'll stop if we reach 50 characters
    # Setting the maximum number of characters helps with debugging and prevents infinite loops.
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):

        # Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Waa, a_prev)+np.dot(Wax, x)+b)
        z = np.dot(Wya, a)+by
        y = softmax(z)

        # Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(range(len(y.ravel())), p=y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Overwrite the input x with one that corresponds to the sampled index `idx`.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices


np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:\n", indices)
print("list of sampled characters:\n", [ix_to_char[i] for i in indices])

# Building the language model
#
# Build the character-level language model for text generation.
#
# Gradient descent
#
# Implement a function performing one step of stochastic gradient descent (with clipped gradients).
#
# The following functions are used from utils:
#
# def rnn_forward(X, Y, a_prev, parameters):
#     """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
#     It returns the loss' value as well as a "cache" storing values to be used in backpropagation."""
#     ....
#     return loss, cache
#
# def rnn_backward(X, Y, parameters, cache):
#     """ Performs the backward propagation through time to compute the gradients of the loss with respect
#     to the parameters. It returns also all the hidden states."""
#     ...
#     return gradients, a
#
# def update_parameters(parameters, gradients, learning_rate):
#     """ Updates parameters using the Gradient Descent Update Rule."""
#     ...
#     return parameters
# ```
#

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    # Forward propagate through time
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # Backpropagate through time
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Clip the gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)

    # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X)-1]



# Training the model

# Given the dataset of dinosaur names, use each line of the dataset (one name) as one training example.
#

def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text (size of the vocabulary)

    Returns:
    parameters -- learned parameters
    """

    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of the LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):

        idx = j%len(examples)

        single_example = examples[idx]
        single_example_chars = [i for i in single_example]
        single_example_ix = [char_to_ix[i] for i in single_example_chars]
        X = [None]+single_example_ix

        ix_newline = char_to_ix['\n']
        Y = single_example_ix + [ix_newline]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)

        if verbose and j in [0, len(examples) -1, len(examples)]:
            print("j = " , j, "idx = ", idx,)
        if verbose and j in [0]:
            print("single_example =", single_example)
            print("single_example_chars", single_example_chars)
            print("single_example_ix", single_example_ix)
            print(" X = ", X, "\n", "Y =       ", Y, "\n")

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):

                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)

            print('\n')

    return parameters


parameters = model(data, ix_to_char, char_to_ix, verbose = True)
