package network

import "math"

func (v *Value) tanh() *Value {
	x := v.value
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	output := Value{value: t, children: []*Value{v}}
	output.backward = func() {
		v.gradient += (1 - (t * t)) * output.gradient
	}
	v.operation = "TANH"
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
