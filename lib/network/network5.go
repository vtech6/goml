package network

import (
	"fmt"
	"math/rand"
)

type Neuron struct {
	weight float64
	bias   float64
	output []float64
}

type Input struct {
	x     []float64
	y     []float64
	shape interface{}
}

type Layer struct {
	//	activationFunction func()
	neurons []Neuron
	output  []float64
}

func (n *Neuron) initNeuron(neuronInput Input) {
	n.weight = randomFloatStd()
	n.bias = generateBias()
}

func randomFloatStd() float64 {
	return rand.Float64()*2 - 1
}

func generateBias() float64 {
	return 1
}

func (n *Neuron) calculateOutput(neuronInput []float64) {
	output := make([]float64, len(neuronInput))
	for i := 0; i < len(neuronInput); i++ {
		output[i] = (n.weight * neuronInput[i]) + n.bias
	}
	n.output = output
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

func average(arr []float64) float64 {
	accumulator := 0.0
	for i := 0; i < len(arr); i++ {
		accumulator += arr[i]
	}
	accumulator = accumulator / float64(len(arr))
	return accumulator
}

func sum(arr []float64) float64 {
	accumulator := 0.0
	for i := 0; i < len(arr); i++ {
		accumulator += arr[i]
	}
	return accumulator
}

func Run() {

	networkInput := []Input{{x: []float64{0, 0}, y: []float64{0}},
		{x: []float64{0, 1}, y: []float64{1}},
		{x: []float64{1, 0}, y: []float64{1}},
		{x: []float64{1, 1}, y: []float64{0}}}

	layer := Layer{neurons: []Neuron{}}

	output := make([]float64, len(networkInput))
	layer.initLayer(networkInput)
	for i := 0; i < len(networkInput); i++ {
		layer.feedForward(networkInput[i].x)
		layer.outputFunc()
		output[i] = average(layer.output)
	}
	fmt.Println(output)

}
