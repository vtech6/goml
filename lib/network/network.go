package network

import (
	"fmt"
	"math/rand"
)

type NeuralNetwork struct {
	numberOfLayers int8
	sizes          []int8
	biases         [][]float32
	weights        [][]float32
}

func NewNeuralNetwork(sizes []int8) *NeuralNetwork {
	numberOfLayers := int8(len(sizes))
	biases := make([][][]float32, numberOfLayers)
	weights := make([][][]float32, numberOfLayers)
	for i := 0; i < int(numberOfLayers); i++ {
		if i > 0 {
			for size := 0; size < int(sizes[i]); size++ {
				biases[i] = append(biases[i], []float32{rand.Float32()})
				_weights := make([]float32, sizes[i-1])
				for j := 0; j < int(sizes[i-1]); j++ {
					_weights[j] = rand.Float32()
				}
				weights[i] = append(weights[i], _weights)

			}
		}

	}

	weights = weights[1:]
	biases = biases[1:]
	fmt.Println(biases)
	fmt.Println(weights)

	return &NeuralNetwork{
		numberOfLayers: numberOfLayers,
		sizes:          sizes}
}
