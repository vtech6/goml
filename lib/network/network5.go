package network

import (
	"fmt"
	"math/rand"
)

type Neuron struct {
	weights []float64
	biases  []float64
	output  []float64
}

func (n *Neuron) initNeuron(neuronInput Input) {
	weights := make([]float64, len(neuronInput.x))
	for i := 0; i < len(neuronInput.x); i++ {
		weights[i] = randomFloatStd()
	}
	n.weights = weights
}

type Input struct {
	x     []float64
	y     []float64
	shape interface{}
}

type Layer struct {
	//	activationFunction func()
	neurons []Neuron
}

func randomFloatStd() float64 {
	return rand.Float64()*2 - 1
}

func (l *Layer) initLayer(networkInput []Input) {
	neurons := make([]Neuron, len(networkInput))
	for i := 0; i < len(networkInput); i++ {
		neuron := Neuron{}
		neuron.initNeuron(networkInput[i])
		neurons[i] = neuron
	}
	l.neurons = neurons
}

func Run() {

	networkInput := []Input{{x: []float64{0, 0}, y: []float64{0}},
		{x: []float64{0, 1}, y: []float64{1}},
		{x: []float64{1, 0}, y: []float64{1}},
		{x: []float64{1, 1}, y: []float64{0}}}

	layer := Layer{neurons: []Neuron{}}

	layer.initLayer(networkInput)

	fmt.Println(layer.neurons)

}
