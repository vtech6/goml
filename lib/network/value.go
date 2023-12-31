package network

import (
	"math"
)

//This is the Value model recreated from Karpathy's tutorial
//Link: https://github.com/karpathy/nn-zero-to-hero/tree/master

type Value struct {
	value     float64
	gradient  float64
	backward  func()
	operation string
	children  []*Value
}

//This is the value initializer. Most functions return pointers, which means
//we don't always return new objects - most of them modify the entities that
//already exist inside our memory. It's also crucial for our case despite
//being more performant, as it allows us to tweak values of our nodes using
//the calculated gradient.

func (v *Value) init(value float64) *Value {
	newValue := Value{value: value}
	newValue.operation = "INIT"
	newValue.gradient = 0.0
	newValue.backward = func() {
	}
	return &newValue
}

//The operations below allow us to do arithmetic on our custom Value type.
//They also assign the newly created values a specific backwards function which
//calculates the gradients using derivatives.

func (v *Value) add(input *Value) *Value {
	output := Value{value: v.value + input.value, children: []*Value{v, input}}
	output.backward = func() {
		v.gradient += 1.0 * output.gradient
		input.gradient += 1.0 * output.gradient
	}
	v.operation = "addition"
	return &output
}

func (v *Value) multiply(input *Value) *Value {
	output := Value{value: v.value * input.value, children: []*Value{v, input}}
	output.backward = func() {
		v.gradient += input.value * output.gradient
		input.gradient += v.value * output.gradient
	}
	v.operation = "multiplication"
	return &output
}

func (v *Value) pow(input float64) *Value {
	output := Value{value: math.Pow(v.value, input), children: []*Value{v}}
	output.backward = func() {
		v.gradient += (input * (math.Pow(v.value, input-1)) * output.gradient)
	}
	v.operation = "pow"
	return &output
}

func (v *Value) log() *Value {
	output := Value{value: math.Log10(v.value), children: []*Value{v}}
	output.backward = func() {
		v.gradient += ((1 / (v.value * math.Ln10)) * output.gradient)
	}
	return &output
}

func (v *Value) exp() *Value {
	output := Value{value: math.Exp(v.value), children: []*Value{v}, operation: "EXP"}
	output.backward = func() {
		v.gradient += (output.value * output.gradient)
	}
	return &output
}

//The function below builds the tree of nodes, then runs backward propagation
//by multiplying the values by their respective gradient and learning rate.
//Learning rate to be implemented (for now we can think of it as value 1).

func (v *Value) calculateGradients() {
	topo := []*Value{}
	buildTopo(v, &topo)
	v.gradient = 1.0
	for nodeIndex := 0; nodeIndex < len(topo); nodeIndex++ {
		if topo[nodeIndex].backward != nil {
			topo[nodeIndex].backward()
		}
	}
}
