package network

import (
	"math"
	"math/rand"
)

func generateLinearData(nInputs int) ([]Input, []Input) {
	data := make([]Input, nInputs)
	for i := 0; i < nInputs; i++ {
		x := 1 * float64(i)
		y := x + rand.Float64()*1
		data[i] = Input{x: []float64{x}, y: []float64{y}}
	}

	data = normalizeData(data)

	trainSize := int(math.Floor(float64(nInputs) * 0.8))
	for i := 0; i < nInputs*10; i++ {
		randomIndex := rand.Intn(nInputs - 1)
		randomIndex2 := rand.Intn(nInputs - 1)
		copy := data[randomIndex]
		data[randomIndex] = data[randomIndex2]
		data[randomIndex2] = copy
	}
	return data[:trainSize], data[trainSize:]
}

func generateInput(nInputs int) []Input {
	generatedInput := make([]Input, nInputs)
	for i := 0; i < nInputs; i++ {
		generatedInput[i] = Input{x: []float64{rand.Float64()*5 + 3}, y: []float64{rand.Float64()*5 + 2}}
	}
	return generatedInput
}
