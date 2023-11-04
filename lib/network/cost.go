package network

import "math"

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
