package network

import "fmt"

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

func (l *Layer) feedForward(layerInput []float64) {
	for i := 0; i < len(l.neurons); i++ {
		l.neurons[i].calculateOutput(layerInput)
		l.output[i] = l.neurons[i].activation
	}
}

func (l *Layer) feedForwardDeep(layerInput []*Value) {
	for i := 0; i < len(l.neurons); i++ {
		l.neurons[i].calculateOutputDeep(layerInput)
		l.output[i] = l.neurons[i].activation
	}
}

func (l *Layer) parameters() [][]*Value {
	params := make([][]*Value, len(l.neurons))
	for i := 0; i < len(l.neurons); i++ {
		params[i] = l.neurons[i].parameters()
	}
	return params
}

type MLP struct {
	layers []*Layer
}

func (m *MLP) initNetwork(shape []int) {
	var layer Layer
	m.layers = make([]*Layer, len(shape))
	for i := 1; i < len(shape); i++ {
		newLayer := layer.initLayer(shape[i-1], shape[i])
		m.layers[i-1] = newLayer
	}
}

func (m *MLP) calculateOutput(networkInput []float64) []*Value {
	for i := 0; i < len(networkInput); i++ {
		if i == 0 {
			m.layers[i].feedForward(networkInput)
		} else {
			m.layers[i].feedForwardDeep(m.layers[i-1].output)
		}
	}
	fmt.Println(m.layers[len(m.layers)-2])
	return m.layers[len(m.layers)-2].output
}

func (m *MLP) parameters() [][][]*Value {
	parameters := make([][][]*Value, len(m.layers)-1)
	for i := 0; i < len(m.layers)-1; i++ {
		parameters[i] = m.layers[i].parameters()

	}
	return parameters
}
