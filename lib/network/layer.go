package network

type Layer struct {
	//	activationFunction func()
	neurons     []Neuron
	output      []float64
	losses      []float64
	derivatives [][]float64
}

func (l *Layer) initLayer(nNeurons int, nWeights int) {
	// For the shape of input generate a neuron
	// Each Neuron initializes with random weights
	// and bias defined in generateBias
	neurons := make([]Neuron, nNeurons)
	l.output = make([]float64, nNeurons)
	for i := 0; i < nNeurons; i++ {
		neuron := Neuron{}
		neuron.initNeuron(nWeights)
		neurons[i] = neuron
	}
	l.neurons = neurons
}

func (l *Layer) feedForward(layerInput []float64) {
	for i := 0; i < len(l.neurons); i++ {

		l.neurons[i].calculateOutput(layerInput)
		l.output[i] = l.neurons[i].activation
	}

}

func (l *Layer) outputFunc() {
	output := make([]float64, len(l.neurons))
	for i := 0; i < len(l.neurons); i++ {
		output[i] = average(l.neurons[i].output)

	}
	l.output = output
}
