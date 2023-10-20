package network

import (
	"fmt"
)

type Input struct {
	x     []float64
	y     []float64
	shape interface{}
}

func Run() {

	//	minimalNetworkInput := []Input{{x: []float64{0, 0}, y: []float64{0}},
	//		{x: []float64{0, 1}, y: []float64{1}},
	//		{x: []float64{1, 0}, y: []float64{1}},
	//		{x: []float64{1, 1}, y: []float64{0}}}

	trainSet, testSet := generateLinearData(100)
	l := train(trainSet, 10)

	l.test(testSet)
}

func train(networkInput []Input, epochs int) *Layer {
	layer := Layer{neurons: []Neuron{}}
	learningRate := 0.01

	layerOutput := make([]float64, len(networkInput))
	layer.initLayer(networkInput)

	for epoch := 0; epoch < epochs; epoch++ {
		accuracy := 0.0
		losses := make([]float64, len(networkInput))
		for i := 0; i < len(networkInput); i++ {
			layer.feedForward(networkInput[i].x)
			layer.outputFunc()
			layerOutput[i] = average(layer.output)
			losses[i] = meanSquaredError(layer.output, networkInput[i].y[0])
			accuracy += (layerOutput[i] / networkInput[i].y[0])
		}
		loss := average(losses)

		accuracy = accuracy / float64(len(networkInput))
		fmt.Println("Loss (MSE):", loss, ", MSE: ", accuracy)

		layer.backpropagate(1 - (loss * learningRate))
		//for i := 0; i < len(networkInput); i++ {
		//	fmt.Println("Target: ", networkInput[i].y, ", Prediction: ", layerOutput[i])
		//	}
	}
	return &layer
}
