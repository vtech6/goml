package network

import (
	"fmt"
)

type Input struct {
	x []float64
	y []float64
}

//MLP stands for Multilayer Perceptron which is the model that we will be
//creating and using. It consists of n-Inputs, m-Hidden-Layers and an Output.

type MLP struct {
	layers []*Layer
}

type NetworkParams struct {
	trainX       [][]float64
	trainY       []float64
	shape        []int
	learningRate float64
	steps        int
}

//To initialize MLP we pass it a shape. This is an array that describes how
//many layers of how many neurons will be in our model.
//For example shape [2,3,1] would mean 2 inputs, 3 hidden layers and an output.

func (m *MLP) initNetwork(shape []int, learningRate float64, steps int) {
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

func (m *MLP) parameters() []*Value {
	parameters := make([]*Value, 0)
	for i := 0; i < len(m.layers)-1; i++ {
		layerParams := m.layers[i].parameters()
		for j := 0; j < len(layerParams); j++ {
			parameters = append(parameters, layerParams[j])

		}

	}
	return parameters
}

func runNetwork(params NetworkParams) {
	inputs := params.trainX
	targets := params.trainY
	shape := params.shape
	steps := params.steps
	learningRate := params.learningRate
	fmt.Println(learningRate)
	network := MLP{}
	network.initNetwork(shape, learningRate, steps)
	//For step in steps we repeat the following process:
	//1. Feed forward the data
	//2. Calculate the output
	//3. Calculate target - predicted output difference
	//4. Calculate and accumulate loss for each prediction
	//5. Calculate gradient for each element of our network
	//6. Move the variable in the opposite direction of its gradient to
	//   decrease loss

	//Missing: batches, epochs

	for step := 0; step < steps; step++ {

		var value Value
		loss := value.init(0.0)
		outputs := make([]*Value, len(inputs))
		//For input of inputs
		for inputIndex := 0; inputIndex < len(inputs); inputIndex++ {
			//Calculate output and add its value to outputs
			output := network.calculateOutput(inputs[inputIndex])
			outputs[inputIndex] = output[0]
		}

		//Calculate loss for each x from the training set and add it to
		//the previously accumulated loss.

		for outputIndex := 0; outputIndex < len(outputs); outputIndex++ {
			targetValue := value.init(targets[outputIndex])
			negativeOutput := outputs[outputIndex].multiply(value.init(-1.0))
			yDifference := negativeOutput.add(targetValue)
			loss = yDifference.square().add(loss)
			fmt.Println("Prediction", outputIndex, " ", outputs[outputIndex].value)
		}

		parameters := network.parameters()
		for paramIndex := 0; paramIndex < len(parameters); paramIndex++ {
			parameters[paramIndex].gradient = 0.0
		}
		//Gradient at the end is always 1.0
		loss.gradient = 1.0

		loss.backward()
		loss.calculateGradients()

		for paramIndex := 0; paramIndex < len(parameters); paramIndex++ {
			parameter := parameters[paramIndex]
			parameter.value += parameter.gradient * -learningRate
		}
		//Parse our network and adjust each variable by gradient

		fmt.Println("---------")
		fmt.Println("Step", step, ", Loss:", loss.value)
		fmt.Println("---------")
	}
	//Below we test if the network can generalize on a data sample. This should
	//be expanded to accept testX and testY, then produce the accuracy of our
	//network. For now we have this minimalistic test.
	val1 := network.calculateOutput([]float64{2.5, 2.5, -2.0})
	fmt.Println("Expected output: ~(1)")
	fmt.Println("VALIDATION: ", val1[0].value)

	val2 := network.calculateOutput([]float64{3, -1, 0})
	fmt.Println("Expected output: ~(-1)")
	fmt.Println("VALIDATION: ", val2[0].value)
}
