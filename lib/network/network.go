package network

import (
	"fmt"
	"math"
)

type Network struct {
	layers []*Layer
}

type NetworkParams struct {
	nEpochs          int
	batchSize        int
	trainX           [][]float64
	trainY           [][]float64
	testX            [][]float64
	testY            [][]float64
	shape            []int
	learningRate     float64
	steps            int
	verbose          bool
	costFunction     string
	neuronActivation string
	saveOutput       bool
}

//To initialize MLP we pass it a shape. This is an array that describes how
//many layers of how many neurons will be in our model.
//For example shape [2,3,1] would mean 2 inputs, 3 hidden layers and an output.

func (m *Network) initNetwork(shape []int, learningRate float64, steps int, neuronActivation string) {
	var layer Layer
	m.layers = make([]*Layer, len(shape))
	for i := 1; i < len(shape); i++ {
		newLayer := layer.initLayer(shape[i-1], shape[i], neuronActivation)
		m.layers[i-1] = newLayer
	}
}

//We iterate through each layer and feed forward the data. What that means, is
//each layer passes the x-input to each of its neurons and multiplies its value
//by corresponding weights. For example - in a network of shape [2,3,1], an
//input [x1, x2] would be multiplied by weights [w1, w2] of 3 neurons of the
//hidden layer and return one output Y.

func (m *Network) calculateOutput(networkInput []float64) []*Value {
	for i := 0; i < len(m.layers)-1; i++ {
		if i == 0 {
			m.layers[i].feedForward(networkInput)
		} else {
			m.layers[i].feedForwardDeep(m.layers[i-1].output)
		}
	}
	return m.layers[len(m.layers)-2].output
}

func (m *Network) parameters() []*Value {
	parameters := make([]*Value, 0)
	for i := 0; i < len(m.layers)-1; i++ {
		layerParams := m.layers[i].parameters()
		for j := 0; j < len(layerParams); j++ {
			parameters = append(parameters, layerParams[j])

		}

	}
	return parameters
}

func runNetwork(params NetworkParams) *SavedData {
	nEpochs := params.nEpochs
	batchSize := params.batchSize
	trainX := params.trainX
	trainY := params.trainY
	testX := params.testX
	testY := params.testY
	shape := params.shape
	steps := params.steps
	verbose := params.verbose
	learningRate := params.learningRate
	costFunction := params.costFunction
	network := Network{}
	network.initNetwork(shape, learningRate, steps, params.neuronActivation)
	saveOutput := params.saveOutput
	var output SavedData
	//For step in steps we repeat the following process:
	//1. Feed forward the data
	//2. Calculate the output
	//3. Calculate target - predicted output difference
	//4. Calculate and accumulate loss for each prediction
	//5. Calculate gradient for each element of our network
	//6. Move the variable in the opposite direction of its gradient to
	//   decrease loss

	//Missing: batches, epochs
	xBatches, yBatches := getMiniBatches(batchSize, trainX, trainY)
	minValue := 0.0
	maxValue := 0.0

	runEpochs(nEpochs, func(epochIndex int) {
		var value Value
		if verbose {
			fmt.Println("---------")
			fmt.Println("Epoch", epochIndex+1)
			fmt.Println("---------")
		}
		var epochOutputs []float64
		for batchIndex := range xBatches {
			inputs := xBatches[batchIndex]
			targets := yBatches[batchIndex]
			for step := 0; step < steps; step++ {
				loss := value.init(0.0)
				for outputIndex := 0; outputIndex < len(inputs); outputIndex++ {
					outputs := network.calculateOutput(inputs[outputIndex])
					if len(targets[outputIndex]) == 1 {
						for valueIndex := range targets[0] {
							if outputs[valueIndex].value < minValue {
								minValue = outputs[valueIndex].value
							}
							if outputs[valueIndex].value > maxValue {
								maxValue = outputs[valueIndex].value
							}
							_output := outputs[valueIndex]
							switch costFunction {
							case "mse":
								loss = Mse(_output, targets[outputIndex][valueIndex]).add(loss)
							default:
								loss = Bce(_output, len(targets), targets[outputIndex][valueIndex]).add(loss)
							}
						}
					} else {
						softMaxed := make([]*Value, len(outputs))
						for i := range softMaxed {
							softMaxed[i] = outputs[i].softmax(outputs, i)
						}

						loss = categoricalCrossEntropy(softMaxed, targets[outputIndex]).add(loss)
						//When trying different loss function, make sure to
						//change the activation functions (for example from
						//tanh to sigmoid for Binary Crossentropy etc.)
					}

					if saveOutput && epochIndex == nEpochs-1 {
						_val := math.Round(((outputs[0].value - minValue) / (maxValue - minValue)))
						epochOutputs = append(epochOutputs, _val)
					}
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
				if verbose {
					fmt.Println("Batch", batchIndex+1, ", Step", step+1, ", Loss:", loss.value)
				}
			}
		}
		if saveOutput {
			output.PredictionsTrain = epochOutputs
		}

	})
	fmt.Println("---------")
	fmt.Println("VALIDATION:")
	fmt.Println("---------")
	minMaxScale := maxValue - minValue
	//Below we test if the network can generalize on a data sample. This should
	//be expanded to accept testX and testY, then produce the accuracy of our
	//network. For now we have this minimalistic test.
	accuracy := 0.0
	switch costFunction {
	case "mse":
		accuracy = regressionMetrics(network, testX, minValue, minMaxScale, testY, &output, saveOutput)
	default:
		accuracy = bceMetrics(network, testX, minValue, minMaxScale, testY, &output, saveOutput)

	}
	fmt.Println("--------")

	fmt.Printf("Test accuracy:%.2f%s", accuracy, "%")
	if saveOutput {
		output.TestX = testX
		output.TestY = testY
		output.TrainX = trainX
		output.TrainY = trainY

		saveData(output)
	}

	return &output
}
