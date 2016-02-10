# # # # # # # # # # # # #
# XOR NeuralNet Example #
# # # # # # # # # # # # #

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure.modules import TanhLayer

def build_simple_XOR_network():
	net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
	return net

def test_simple_XOR_network(network):
	print(network.activate([0, 0]))
	print(network.activate([0, 1]))
	print(network.activate([1, 0]))
	print(network.activate([1, 1]))
	
def create_XOR_dataset():
	ds = SupervisedDataSet(2, 1)
	ds.addSample((0, 0), (0,))
	ds.addSample((0, 1), (1,))
	ds.addSample((1, 0), (1,))
	ds.addSample((1, 1), (0,))
	return ds
	
network = build_simple_XOR_network()
test_simple_XOR_network(network)
print()

dataSet = create_XOR_dataset()
trainer = BackpropTrainer(network, dataSet)

for i in range(1000):
	print("Epoch: " + str(i))
	trainer.train()

test_simple_XOR_network(network)
