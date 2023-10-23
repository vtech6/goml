package network

func (n *Network) networkValidate() {
	networkLosses := make([]float64, len(n.layers))
	for layerIndex := 0; layerIndex < len(n.layers); layerIndex++ {
		networkLosses[layerIndex] = average(n.layers[layerIndex].losses)
	}
	n.loss = 0
}

func (n *Network) networkBackpropagate(networkInput []Input) {
	lenLayers := len(n.layers)
	for j := 0; j < len(networkInput); j++ {
		for i := lenLayers - 1; i > 0; i-- {

			//		activations := n.layers[i].output
			//		prevActivations := n.layers[i].output
			comparison := n.layers[i]
			if i != lenLayers-1 {
				comparison = n.layers[i+1]
			}
			n.layers[i].backpropagate(networkInput[j], i == (lenLayers-1), &comparison)
		}
	}
}

func (l *Layer) backpropagate(networkInput Input, lastLayer bool, nextLayer *Layer) {

}
