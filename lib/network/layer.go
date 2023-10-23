package network

type Layer struct {
	//	activationFunction func()
	neurons     []Neuron
	output      []*Value
	losses      []float64
	derivatives [][]float64
}

func (l *Layer) initLayer(nNeurons int, nWeights int) {
	// For the shape of input generate a neuron
	// Each Neuron initializes with random weights
	// and bias defined in generateBias
	neurons := make([]Neuron, nNeurons)
	l.output = make([]*Value, nNeurons)
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

func (l *Layer) outputFunc() {}
