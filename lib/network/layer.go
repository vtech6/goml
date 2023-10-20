package network

type Layer struct {
	//	activationFunction func()
	neurons []Neuron
	output  []float64
}

func (l *Layer) initLayer(networkInput []Input) {
	// For the shape of input generate a neuron
	// Each Neuron initializes with random weight
	// and biase defined in generateBias
	nNeurons := 4
	neurons := make([]Neuron, nNeurons)
	for i := 0; i < nNeurons; i++ {
		neuron := Neuron{}
		neuron.initNeuron(networkInput[i])
		neurons[i] = neuron
	}
	l.neurons = neurons
	l.output = make([]float64, nNeurons)
}

func (l *Layer) feedForward(layerInput []float64) {
	for i := 0; i < len(l.neurons); i++ {
		l.neurons[i].calculateOutput(layerInput)
	}

}

func (l *Layer) outputFunc() {
	output := make([]float64, len(l.neurons))
	for i := 0; i < len(l.neurons); i++ {
		output[i] = average(l.neurons[i].output)
	}
	l.output = output
}

func (l *Layer) backpropagate(loss float64) {
	for i := 0; i < len(l.neurons); i++ {
		l.neurons[i].weight = l.neurons[i].weight * loss
		l.neurons[i].bias = l.neurons[i].bias * loss
	}
}
