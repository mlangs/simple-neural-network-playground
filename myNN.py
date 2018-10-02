import numpy as np
from scipy.special import expit
from math import ceil


def create(input, output, *args, scaling = 1):
    # Inputs:   input: number of input nodes (integer)
    #           output: number of output nodes (integer)
    #           *args: (optional) number of nodes for every hidden layer (list)
    #           scaling: maximum values of the weights
    # Output:   array consisting of the weights arrays

    weights = []

    # args are tuples, but we call the function with args as a list
    # so we take the list out of the tuple
    if args != ():
        args = args[0]

    # adding all inputs in one list
    # the for loop is empty if there are no *args
    all_args = [input]
    for i in range(len(args)):
        all_args.append(args[i])
    all_args.append(output)

    # generating all the weight arrays and adding them to an array
    # the values are between -1 and 1
    # the +1 in the np.random.rand() function is needed for the weights related
    # to the bias
    for i in range(1,len(all_args)):
        w = scaling *(2 * np.random.rand(all_args[i], all_args[i-1] + 1) - 1)
        weights += [w]

    return weights


def eval(input, weights):
    # Inputs:  input: value of the input nodes (list)
    #          weights: array consisting of weight arrays
    # Output:  value of the output nodes (list)

    # add 1 to bias
    input = [[1]] + [[i] for i in input]

    # first multiplication: weights * inputs
    # use of the sigmoid function for scaling:
    # expit(x) = 1/(1+exp(-x))
    temp = expit(np.matmul(weights[0], input))

    # add 1 again for bias
    temp = [[1]] + [[i] for i in temp]

    # for loop is only relevant if there are hidden layers
    for i in range(1, len(weights)):
        temp = expit(np.matmul(weights[i], temp))
        temp = [[1]] + [[i] for i in temp]

    # transforming the list of arrays into a regular list
    # without the 1 in the beginning
    output = [ temp[i][0][0] for i in range(1, len(temp))]

    return output


def eval_fitness(real_result, estimate, OUTPUT):
    # Inputs:  real_result: correct output values (list)
    #          estimate: estimated output values (list)
    #          OUTPUT: number of output layers
    # Output:  fitness: returns fitness value (float)
    #          please note: the value is good if it is low!

    if type(real_result) != list or len(real_result) != OUTPUT:
        print('Error: Type or lenght of the List is not correct!')
        return

    fitness = 0
    for i in range(OUTPUT):
        fitness += abs(real_result[i] - estimate[i])**2
    return round(fitness, 5)


def mutate(weights, mutation_rate = 0.1):
    # Inputs:  weights: array consisting of weight arrays
    #          mutation_rate: maximum change of the weight values
    # Output:  weights: array consisting of weight arrays

    # for every array in weights mask is created
    # mask consists of boolean values (True, False)
    # for every array in weights a random array r in the same size
    # is created (scaling with mutation_rate)
    # r arrays are added to the weight arrays,
    # but only the values are added where the mask says True
    for i in range(len(weights)):
        mask = np.random.randint(0,2,size=weights[i].shape).astype(np.bool)
        r =  mutation_rate * (2 * np.random.rand(*weights[i].shape) - 1)
        weights[i][mask] = weights[i][mask]+r[mask]

    return weights


def recombine(weights1, weights2):
    # Inputs:  weights1: array consisting of weight arrays
    #          weights2: array consisting of weight arrays
    # Output:  weights1: new array consisting of combined weight arrays

    # for every array in weights mask is created
    # mask consists of boolean values (True, False)
    # if the boolean value is True, the weight value in weights1
    # gets replace by the weight value of weights2
    for i in range(len(weights1)):
        mask = np.random.randint(0,2,size=weights1[i].shape).astype(np.bool)
        weights1[i][mask] = weights2[i][mask]

    return weights1


def select(id_list, fitness_list, sel_rate = 0.7, probability_scale = 2):
    # Inputs:  id: IDs of individuals (list)
    #          fitness: fitness values of individuals (list)
    #          please note: lower value means more fitness
    #          sel_rate: percentage of surviving individuals
    #          probability_scale: a higher value makes the selection
    #          less sensitive
    # Output:  id_list list of surviving individuals
    #          this list is shortended according to sel_rate

    surviving = []

    l = len(id_list)
    p = [1/(i+probability_scale) for i in fitness_list]
    s = sum(p)
    probability = [i/s for i in p]

    surviving = np.random.choice(l, l, replace = True, p = probability)
    surviving = [id_list[i] for i in surviving]

    return surviving[:ceil(l*sel_rate)]



if __name__ == "__main__":
    print('Please do not run as __main__')
