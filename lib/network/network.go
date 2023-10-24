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

	testFunc()
}

func testFunc() {
	inputs := [][]float64{{2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}}
	targets := []float64{1.0, -1.0, -1.0, 1.0}
	network := MLP{}
	network.initNetwork([]int{3, 4, 4, 1})

	for k := 0; k < 20; k++ {
		loss := Value{value: 0.0}
		predictions := make([]*Value, len(inputs))
		for i := 0; i < len(predictions); i++ {
			output := network.calculateOutput(inputs[i])[0]
			fmt.Println(output, "OUTPUT")
			predictions[i] = output
		}
		for i := 0; i < len(predictions); i++ {
			negativePred := predictions[i].negative()
			targetValue := Value{value: targets[i]}
			loss = *loss.add(targetValue.add(&negativePred))
		}

		fmt.Println("PREDICTION 1", predictions[0].value)
		fmt.Println(loss, "LOSS")
	}
	network.calculateOutput([]float64{0.5, 0.8})
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

func (n *Network) networkFeedForward(networkInput []Input) {
	for i := 0; i < len(networkInput); i++ {
		for layerIndex := 0; layerIndex < len(n.layers); layerIndex++ {
			if layerIndex == 0 {

				//If layer is first after input, process the input
				n.layers[layerIndex].feedForward(networkInput[i].x)
			} else {

				//Else process the output of the previous layer
			}
		}

	}
}
