package network

import (
	"fmt"
)

type Input struct {
	x     []float64
	y     []float64
	shape interface{}
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

		loss := meanSquaredError(layer.output, networkInput[i].y[0])
		fmt.Println(loss)
	}

	fmt.Println(output)

}
