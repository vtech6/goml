package network

import (
	"fmt"
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

//Tanh is an activation function, so it does not require any other input than
//the Value itself. It returns a new Value that wraps the original.

func (v *Value) tanh() *Value {
	x := v.value
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	output := Value{value: t, children: []*Value{v}}
	output.backward = func() {
		v.gradient += (1 - (t * t)) * output.gradient
	}
	v.operation = "activation"
	return &output
}

func (v *Value) relu() *Value {
	loss := math.Max(0, v.value)
	output := Value{children: []*Value{v}, operation: "RELU", value: loss}
	output.backward = func() {
		derivative := 0.0
		if v.value > 0 {
			derivative = 1
		}
		v.gradient += derivative * output.gradient
	}
	return &output
}

func (v *Value) sigmoid() *Value {
	activation := (1.0 / (1.0 + math.Exp(-v.value)))
	output := Value{value: activation, children: []*Value{v}, operation: "SIGMOID"}
	output.backward = func() {
		v.gradient += (activation * (1 - activation)) * output.gradient
	}
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

func (v *Value) softmax(inputs []*Value, inputIndex int) *Value {
	sumExp := 0.0
	for i := range inputs {
		sumExp += math.Exp(inputs[i].value)
	}
	e := math.Exp(v.value) / sumExp
	output := Value{value: e, children: []*Value{v}}
	output.backward = func() {
		for i := range inputs {
			for j := range inputs {
				if i == j {
					inputs[i].gradient += inputs[i].value * (1.0 - inputs[j].value) * output.gradient
				} else {
					inputs[i].gradient += (-inputs[j].value) * inputs[i].value * output.gradient
				}
			}
		}
	}
	return &output
}

func (v *Value) mse(target float64) *Value {
	var value Value
	targetValue := value.init(target)
	negativeOutput := v.multiply(value.init(-1.0))
	yDifference := negativeOutput.add(targetValue)
	powDifference := yDifference.pow(2)
	return powDifference
}

func Mse(input *Value, target float64) *Value {
	return input.mse(target)
}

func (v *Value) binaryCrossEntropy(dataLength int, target float64) *Value {
	loss := 0.0
	ratio := 1.0 / float64(dataLength)
	log := math.Log(v.value)
	loss = -(ratio * log)
	output := Value{children: []*Value{v}, value: loss}
	output.backward = func() {
		derivative := 0.0
		if target == 1 {
			derivative = -1 / v.value
		} else {
			derivative = 1 / (1 - v.value)
		}

		v.gradient += derivative * output.gradient
	}

	return &output
}
func Bce(value *Value, dataLength int, target float64) *Value {
	return value.binaryCrossEntropy(dataLength, target)
}

func categoricalCrossEntropy(outputs []*Value, targets []float64) *Value {
	loss := 0.0
	for i := range outputs {
		loss += -(targets[i] * (math.Log(outputs[i].value) + math.Pow(10, -100)))
	}
	output := Value{value: loss, children: outputs, operation: "CCE"}
	output.backward = func() {
		for i := range outputs {
			if outputs[i].value != 0 {
				outputs[i].gradient += (-(targets[i] / (outputs[i].value + (math.Pow(10, -100))))) * output.gradient
			}
		}
	}
	return &output

}

func softmax2(outputs []*Value) []*Value {
	var value Value
	sumExp := 0.0
	newOutputs := make([]*Value, len(outputs))

	for i := range outputs {
		sumExp += math.Exp(outputs[i].value)
	}

	for i := range outputs {
		exp := math.Exp(outputs[i].value)
		softMaxed := exp / sumExp
		newOutputs[i] = value.init(0.0)
		newOutputs[i].value = softMaxed
		newOutputs[i].children = outputs
		newOutputs[i].backward = func() {
			for j := range outputs {
				if i == j {
					outputs[i].gradient += outputs[i].value * (1.0 - outputs[j].value) * newOutputs[i].gradient
				} else {
					outputs[i].gradient += (-outputs[j].value) * outputs[i].value * newOutputs[j].gradient
				}
			}

		}
	}
	fmt.Println(newOutputs[0].gradient)
	return newOutputs
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
