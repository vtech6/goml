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
	var value Value
	x1 := value.init(2.0)
	x2 := value.init(0.0)

	w1 := value.init(-3.0)
	w2 := value.init(1.0)

	bias := value.init(6.8813735870195432)
	x1w1 := x1.multiply(w1)
	x2w2 := x2.multiply(w2)
	x1w1x2w2 := x1w1.add(x2w2)
	n := x1w1x2w2.add(bias)
	output := n.tanh()
	output.gradient = 1.0
	output.backward()
	output.calculateGradients()
	fmt.Println(output.gradient, n.gradient)
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
				n.layers[layerIndex].feedForward(n.layers[layerIndex-1].output)
			}
			n.layers[layerIndex].outputFunc()
		}

		n.output[i] = average(n.layers[len(n.layers)-1].output)
	}
}
