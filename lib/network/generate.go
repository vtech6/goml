package network

import "math"

func generateLinearData(nInputs int) ([]Input, []Input) {
	data := make([]Input, nInputs)
	for i := 0; i < nInputs; i++ {
		x := 1 * float64(i)
		y := x + 0.5
		data[i] = Input{x: []float64{x}, y: []float64{y}}

	}
	trainSize := int(math.Floor(float64(nInputs) * 0.8))
	return data[:trainSize], data[trainSize:]
}

func generateInput(nInputs int) []Input {
	generatedInput := make([]Input, nInputs)
	for i := 0; i < nInputs; i++ {
		generatedInput[i] = Input{x: []float64{randomFloatStd()*2 + 3}, y: []float64{randomFloatStd()*5 + 2}}
	}
	return generatedInput
}
