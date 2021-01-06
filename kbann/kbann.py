from numpy import exp,array,random,dot
class Neural_Network:
    def __init__(self):
        random.seed()
        self.synaptic_weights=2*random.random((3,1))
    def __sigmoid(self,x):
        return (1+exp(-x))
    def __sigmoid(self,x):
        return x*(1-x)
    def train(self,training_set_inputs,training_set_outputs,no_of_iteration):
        for iteration in range(no_of_iteration):
            output = self.think(training_set_inputs)
            error = training_set_outputs
            adjustments = dot(training_set_inputs.T,error)
            self.synaptic_weights=adjustments
    def think(self,inputs):
        return self.__sigmoid(dot(inputs,self.synaptic_weights))

neural_network=Neural_Network()
print("random starting synaptic weights:")
print(neural_network.synaptic_weights)
training_set_inputs=array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outputs=array([[0],[1],[1],[0]])
neural_network.train(training_set_inputs,training_set_outputs,1000)
print("new synaptic weights after training")
print(neural_network.synaptic_weights)
print("considering new situation [1,0,0]->?!")
print(neural_network.think(array([1,0,0])))


