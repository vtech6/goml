package network

type Layer struct {
	//	activationFunction func()
	neurons     []*Neuron
	output      []*Value
	losses      []float64
	derivatives [][]float64
}

func (l *Layer) initLayer(nInputs int, nNeurons int) *Layer {
	var layer Layer
	// For the shape of input generate a neuron
	// Each Neuron initializes with random weights
	// and bias defined in generateBias
	neurons := make([]*Neuron, nNeurons)
	layer.output = make([]*Value, nNeurons)
	for i := 0; i < nNeurons; i++ {
		neuron := Neuron{}
		neuron.initNeuron(nInputs)
		neurons[i] = &neuron
	}
	layer.neurons = neurons
	return &layer
}

//We pass the layer input of shape []float64 to our neurons if it's the first
//hidden layer

func (l *Layer) feedForward(layerInput []float64) {
	for i := 0; i < len(l.neurons); i++ {
		l.neurons[i].calculateOutput(layerInput)
		l.output[i] = l.neurons[i].activation
	}
}

//For hidden layers, the input is the output of the previous layer, which means
//we should accept an array of Values.

func (l *Layer) feedForwardDeep(layerInput []*Value) {
	for i := 0; i < len(l.neurons); i++ {
		l.neurons[i].calculateOutputDeep(layerInput)
		l.output[i] = l.neurons[i].activation
	}
}

func (l *Layer) parameters() *[][]*Value {
	params := make([][]*Value, len(l.neurons))
	for i := 0; i < len(l.neurons); i++ {
		params[i] = l.neurons[i].parameters()
	}
	return &params
}
