package network

import "fmt"

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
	fmt.Println(m)
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
	fmt.Println(m.layers[len(m.layers)-2])
	return m.layers[len(m.layers)-2].output
}

//The following function allows us to gather all the parameters of our nodes
//so that we can multiply their values by their corresponding gradients.

func (m *MLP) parameters() [][][]*Value {
	parameters := make([][][]*Value, len(m.layers)-1)
	for i := 0; i < len(m.layers)-1; i++ {
		parameters[i] = *m.layers[i].parameters()

	}
	return parameters
}
