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

func Run() {

	network := initNetwork()
	trainSet := generateInput(2)

	network.trainNetwork(normalizeData(trainSet))
	network.networkPredict([]float64{0.5, 0.1, 0.5, 0.8, 0.9, trainSet[0].x[0]})

	testFunc()
}

func testFunc() {
	neuron := Neuron{}
	neuron.initNeuron(2)
	neuron.calculateOutput([]float64{0.5, 0.8})
	fmt.Println(neuron.activation)
}

func initNetwork() Network {
	var network Network
	network.setupNetwork(0.001, 2)
	return network
}

func (n *Network) setupNetwork(learningRate float64, nHiddenLayers int) {
	n.epochs = 1
	n.learningRate = learningRate
	n.nHiddenLayers = nHiddenLayers
}

func (n *Network) initLayers(nHiddenLayers int) {
	layers := make([]Layer, nHiddenLayers)
	for i := 0; i < nHiddenLayers; i++ {
		layer := Layer{neurons: []Neuron{}}
		if i == 0 {
			layer.initLayer(1, 2)
		} else {
			layer.initLayer(2, 2)
		}
		layers[i] = layer
	}
	n.layers = layers
}

func (n *Network) trainNetwork(networkInput []Input) {
	n.output = make([]float64, len(networkInput))
	n.initLayers(n.nHiddenLayers)
	for i := 0; i < n.epochs; i++ {
		n.networkFeedForward(networkInput)
		n.networkValidate()
		if i < n.epochs-1 {
			n.networkBackpropagate(networkInput)
		}
	}
	//	for i := 0; i < len(networkInput); i++ {
	//		fmt.Println("Source: ", networkInput[i].x, ", Target: ", networkInput[i].y, ", Prediction: ", n.output[i])
	//	}
}

func (n *Network) networkFeedForward(networkInput []Input) {
	for i := 0; i < len(networkInput); i++ {
		for layerIndex := 0; layerIndex < len(n.layers); layerIndex++ {
			if layerIndex == 0 {

				//If layer is first after input, process the input
				n.layers[layerIndex].feedForward(networkInput[i].x)
			} else {

				//Else process the output of the previous layer
			}
			n.layers[layerIndex].outputFunc()
		}

	}
}
