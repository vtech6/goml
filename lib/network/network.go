package network

import (
	"fmt"
	"math/rand"
)

type Input struct {
	x []float64
	y []float64
}

type Network struct {
	epochs        int
	batchSize     int
	layers        []Layer
	loss          float64
	learningRate  float64
	nHiddenLayers int
	output        []float64
}

//MLP stands for Multilayer Perceptron which is the model that we will be
//creating and using. It consists of n-Inputs, m-Hidden-Layers and an Output.

type MLP struct {
	layers []*Layer
}

//To initialize MLP we pass it a shape. This is an array that describes how
//many layers of how many neurons will be in our model.
//For example shape [2,3,1] would mean 2 inputs, 3 hidden layers and an output.

func (m *MLP) initNetwork(shape []int) {
	var layer Layer
	m.layers = make([]*Layer, len(shape))
	for i := 1; i < len(shape); i++ {
		newLayer := layer.initLayer(shape[i-1], shape[i])
		m.layers[i-1] = newLayer
	}
}

//We iterate through each layer and feed forward the data. What that means, is
//each layer passes the x-input to each of its neurons and multiplies its value
//by corresponding weights. For example - in a network of shape [2,3,1], an
//input [x1, x2] would be multiplied by weights [w1, w2] of 3 neurons of the
//hidden layer and return one output Y.

func (m *MLP) calculateOutput(networkInput []float64) []*Value {
	for i := 0; i < len(networkInput); i++ {
		if i == 0 {
			m.layers[i].feedForward(networkInput)
		} else {
			m.layers[i].feedForwardDeep(m.layers[i-1].output)
		}
	}
	return m.layers[len(m.layers)-2].output
}

func RunNetwork() {
	inputs := [][]float64{{2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}}
	targets := []float64{1.0, -1.0, -1.0, 1.0}
	network := MLP{}
	network.initNetwork([]int{3, 4, 4, 1})
	//For step in steps
	for step := 0; step < 10; step++ {
		var value Value
		loss := value.init(0.0)
		outputs := make([]*Value, len(inputs))
		//For input of inputs
		for inputIndex := 0; inputIndex < len(inputs); inputIndex++ {
			//Calculate output and add its value to outputs
			output := network.calculateOutput(inputs[inputIndex])
			outputs[inputIndex] = output[0]
		}

		for outputIndex := 0; outputIndex < len(outputs); outputIndex++ {
			targetValue := value.init(targets[outputIndex])
			negativeOutput := outputs[outputIndex].negative()
			yDifference := negativeOutput.add(targetValue)
			loss = loss.add(yDifference.tanh())
			fmt.Println("Prediction", outputIndex, " ", outputs[outputIndex].value)

		}

		loss.gradient = 1.0
		loss.backward()
		loss.calculateGradients()
		for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
			layer := network.layers[layerIndex]
			for neuronIndex := 0; neuronIndex < len(layer.neurons); neuronIndex++ {
				neuron := layer.neurons[neuronIndex]
				for valueIndex := 0; valueIndex < len(neuron.weights); valueIndex++ {
					//Replace line below with gradient
					weight := neuron.weights[valueIndex].value * rand.Float64()
					weight += neuron.weights[valueIndex].value
					neuron.weights[valueIndex].value = weight
				}

				neuron.bias.value += neuron.bias.value * neuron.bias.gradient
			}
		}
		fmt.Println("---------")
		fmt.Println("Step", step, ", Loss:", loss.value)
		fmt.Println("---------")
	}
	output := network.calculateOutput([]float64{2.5, 2.5, -0.5})
	fmt.Println("VALIDATION: ", output[0].value)
}
