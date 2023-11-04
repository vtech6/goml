package network

import (
	"fmt"
	"math"
)

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
