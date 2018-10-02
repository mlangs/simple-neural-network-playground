import myNN as NN


class NeuralNet:

    POPULATION = 20 #population size
    GENERATIONS = 5 #number of generations
    INPUT = 2
    OUTPUT = 3
    HIDDEN = [2, 2, 2]
    SCALING = 1 #scaling for creating a NN
    M_RATE = 0.1
    SEL_RATE = 0.7 #how many survive
    PROBABILITY_SCALE = 2 # a higher value makes the selection less sensitive


    def __init__(self):
        self.weights = NN.create(NeuralNet.INPUT, NeuralNet.OUTPUT,
                                NeuralNet.HIDDEN, scaling = NeuralNet.SCALING)
        self.fitness = 0


    def reset_weights(self):
        self.weights = NN.create(NeuralNet.INPUT, NeuralNet.OUTPUT,
                                NeuralNet.HIDDEN, scaling = NeuralNet.SCALING)


    def eval(self, inputList):
        return NN.eval(inputList, self.weights)


    def converted_result(self, inputList):
        # similar to eval, but with rounded results
        return [1 if i >=0.5 else 0 for i in NN.eval(inputList, self.weights)]


    def eval_fitness(self, real_result, estimate):
        self.fitness = NN.eval_fitness(real_result, estimate, NeuralNet.OUTPUT)
        # self.fitness = NN.eval_fitness(real_result, self.eval(inputList), NeuralNet.OUTPUT)
        # maybe better? estimate in function would be redundant


    def mutate(self):
        self.weights = NN.mutate(self.weights, mutation_rate = NeuralNet.M_RATE)


    def recombine(self, weights1, weights2):    # NOTE: probably not the best way? evolve function?
        self.weights = NN.recombine(weights1, weights2)




    @classmethod
    def select(cls, id_list, fitness_list):
        return NN.select(id_list, fitness_list, sel_rate = cls.SEL_RATE,
                        probability_scale = cls.PROBABILITY_SCALE)



    # additional constructor

    @classmethod
    def from_list_generate_pop(cls):
        # generates a whole generation
        return [cls() for i in range(NeuralNet.POPULATION)]




if __name__ == "__main__": # following is just for testing
    # Posibillity to set values
    NeuralNet.POPULATION = 4

    #genereate generation
    lst = NeuralNet.from_list_generate_pop()



    # for i in lst:
    #     print(id(i), i.weights)
        # if id(i) not in select(id_list, fitness_list):
        #     i.reset_weights()


    # for k in lst:
    #     for i in lst:
    #         for j in lst:
    #             if i.fitness == value_a and j.fitness == value_b and k.fitness == value_c:
    #             k.recombine(i.weights, j.weights)
