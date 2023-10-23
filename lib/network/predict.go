package network

import "fmt"

func (n *Network) networkPredict(networkInputX []float64) {
	predictions := make([]float64, len(networkInputX))
	for i := 0; i < len(networkInputX); i++ {
		for layerIndex := 0; layerIndex < len(n.layers); layerIndex++ {
			if layerIndex == 0 {
				//If layer is first after input, process the input
				n.layers[layerIndex].feedForward([]float64{networkInputX[i]})
			} else {
				//Else process the output of the previous layer
			}
		}
		fmt.Println("Source: ", networkInputX[i], ", Prediction: ", predictions[i])
	}
}
