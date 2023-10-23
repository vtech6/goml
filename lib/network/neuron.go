package network

import (
	"fmt"
	"math"
)

type Neuron struct {
	weights    []float64
	bias       float64
	output     []float64
	delta      float64
	activation float64
}

func (n *Neuron) initNeuron(inputLength int) {
	n.weights = make([]float64, inputLength)
	for i := 0; i < inputLength; i++ {
		n.weights[i] = randomFloatStd()
	}
	n.bias = generateBias()
	n.output = make([]float64, inputLength)
}

func (n *Neuron) calculateOutput(neuronInput []float64) {
	output := n.bias
	zipped, maxLen := zip(n.weights, neuronInput)
	for i := 0; i < maxLen; i++ {
		output += zipped[i][0] * zipped[i][1]
		n.output[i] = sigmoid(output)

	}
	n.activation = sigmoid(output)

}

type Value struct {
	value     float64
	gradient  float64
	backward  func()
	operation string
	children  []*Value
}

func (v *Value) init(value float64) *Value {
	newValue := Value{value: value}
	newValue.operation = "INIT"
	newValue.gradient = 0.0
	newValue.backward = func() {
		fmt.Println("NOTHING")
	}
	return &newValue

}

func (v *Value) multiply(input *Value) *Value {
	output := Value{value: v.value * input.value, children: []*Value{v, input}}
	output.backward = func() {
		v.gradient += input.value * output.gradient
		input.gradient += v.value * output.gradient
		fmt.Println("MULTIPLICATION")
	}
	v.operation = "multiplication"
	return &output
}

func (v *Value) add(input *Value) *Value {
	output := Value{value: v.value + input.value, children: []*Value{v, input}}
	output.backward = func() {
		v.gradient += 1.0 * output.gradient
		input.gradient += 1.0 * output.gradient
		fmt.Println(v.gradient, "ADDITION Gradient")
	}
	v.operation = "addition"
	return &output
}

func (v *Value) tanh() *Value {
	x := v.value
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	output := Value{value: t, children: []*Value{v}}
	output.backward = func() {
		v.gradient += (1 - (t * t)) * output.gradient
		fmt.Println(v.gradient, "TANH GRAD")
		fmt.Println("TANH")
	}
	v.operation = "activation"
	return &output
}

func (v *Value) calculateGradients() {
	topo := []*Value{}
	buildTopo(v, &topo)
	fmt.Println(topo)
	v.gradient = 1.0
	for nodeIndex := 0; nodeIndex < len(topo); nodeIndex++ {
		topo[nodeIndex].backward()
		fmt.Println(topo[nodeIndex])
	}

}

func buildTopo(v *Value, topo *[]*Value) {
	for i := 0; i < len(v.children); i++ {
		*topo = append(*topo, v.children[i])
		buildTopo(v.children[i], topo)
	}
}
