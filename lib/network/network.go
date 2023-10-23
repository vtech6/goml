package network

import (
	"fmt"
)

type Input struct {
	x     []float64
	y     []float64
	shape interface{}
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

func Run() {

	//	minimalNetworkInput := []Input{{x: []float64{0, 0}, y: []float64{0}},
	//		{x: []float64{0, 1}, y: []float64{1}},
	//		{x: []float64{1, 0}, y: []float64{1}},
	//		{x: []float64{1, 1}, y: []float64{0}}}
	network := initNetwork()
	trainSet := generateInput(100)

	network.trainNetwork(trainSet)
	network.networkPredict([]float64{1.3, 1.2, 1.3, 2, 2.5, 8, 2.3})
}

func (n *Network) networkTest(testSet []Input) {
	fmt.Println("-------")
	n.networkFeedForward(testSet)
}

func initNetwork() Network {
	var network Network
	network.setupNetwork(0.001, 2)
	return network
}

func (n *Network) setupNetwork(learningRate float64, nHiddenLayers int) {
	n.epochs = 10
	n.learningRate = learningRate
	n.nHiddenLayers = nHiddenLayers
}

func (n *Network) trainNetwork(networkInput []Input) {
	n.output = make([]float64, len(networkInput))
	n.initLayers(n.nHiddenLayers)
	for i := 0; i < n.epochs; i++ {
		n.networkFeedForward(networkInput)
		n.networkValidate()
		n.networkBackpropagate()
	}
	for i := 0; i < len(networkInput); i++ {
		fmt.Println("Source: ", networkInput[i].x, ", Target: ", networkInput[i].y, ", Prediction: ", n.output[i])
	}
}

func (n *Network) initLayers(nHiddenLayers int) {
	layers := make([]Layer, nHiddenLayers)
	for i := 0; i < nHiddenLayers; i++ {
		layer := Layer{neurons: []Neuron{}}
		layer.initLayer()
		layers[i] = layer
	}
	n.layers = layers
}

func (n *Network) networkPredict(networkInputX []float64) {
	predictions := make([]float64, len(networkInputX))
	for i := 0; i < len(networkInputX); i++ {
		for layerIndex := 0; layerIndex < len(n.layers); layerIndex++ {
			if layerIndex == 0 {
				//If layer is first after input, process the input
				n.layers[layerIndex].feedForward([]float64{networkInputX[i]})
			} else {
				//Else process the output of the previous layer
				n.layers[layerIndex].feedForward(n.layers[layerIndex-1].output)
			}
			n.layers[layerIndex].outputFunc()
		}
		predictions[i] = average(n.layers[len(n.layers)-1].output)
		fmt.Println("Source: ", networkInputX[i], ", Prediction: ", predictions[i])
	}
}

func (n *Network) networkFeedForward(networkInput []Input) {
	for i := 0; i < len(networkInput); i++ {
		for layerIndex := 0; layerIndex < len(n.layers); layerIndex++ {
			if layerIndex == 0 {
				//If layer is first after input, process the input
				n.layers[layerIndex].feedForward(networkInput[i].x)
			} else {
				//Else process the output of the previous layer
				n.layers[layerIndex].feedForward(n.layers[layerIndex-1].output)
			}
			n.layers[layerIndex].outputFunc()
			n.layers[layerIndex].losses = make([]float64, len(networkInput))
			n.layers[layerIndex].losses[i] = meanSquaredError(n.layers[layerIndex].output, networkInput[i].y[0])
		}
		n.output[i] = average(n.layers[len(n.layers)-1].output)
	}
}

func (n *Network) networkValidate() {
	networkLosses := make([]float64, len(n.layers))
	for layerIndex := 0; layerIndex < len(n.layers); layerIndex++ {
		networkLosses[layerIndex] = average(n.layers[layerIndex].losses)
	}
	n.loss = average(networkLosses)
}

func (n *Network) networkBackpropagate() {
	for i := 0; i < len(n.layers); i++ {
		activations := n.layers[i].output
		delta = 1
		n.layers[i-1].backpropagate(1 - (n.loss * n.learningRate))
	}
}

func train(networkInput []Input, epochs int) *Layer {
	layer := Layer{neurons: []Neuron{}}
	learningRate := 0.001

	layerOutput := make([]float64, len(networkInput))
	layer.initLayer()

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
