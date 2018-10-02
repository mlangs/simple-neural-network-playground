### here i will give an example problem
### this file is just a placeholder for now

from myNNClass import NeuralNet


NeuralNet.POPULATION = 20 #population size
NeuralNet.GENERATIONS = 5 #number of generations
NeuralNet.INPUT = 2
NeuralNet.OUTPUT = 3
NeuralNet.HIDDEN = [2, 2, 2]
NeuralNet.M_RATE = 0.1
NeuralNet.SEL_RATE = 0.7 #how many survive
NeuralNet.PROBABILITY_SCALE = 2 # a higher value makes the selection less sensitive


#playground
a = [25, 6]

NeuralNet.POPULATION = 4
lst = NeuralNet.from_list_generate_pop()

for i in lst:
    i.eval(a)
    print(id(i), i.weights)
