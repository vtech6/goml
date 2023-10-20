package network

type Neuron struct {
	weight float64
	bias   float64
	output []float64
}

func (n *Neuron) initNeuron(neuronInput Input) {
	n.weight = randomFloatStd()
	n.bias = generateBias()
}

func (n *Neuron) calculateOutput(neuronInput []float64) {
	output := make([]float64, len(neuronInput))
	for i := 0; i < len(neuronInput); i++ {
		output[i] = (n.weight * neuronInput[i]) + n.bias
	}
	n.output = output
}
