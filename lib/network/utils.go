package network

import "math/rand"

func randomFloatStd() float64 {
	return 1
}

func generateBias() float64 {
	// return randomFloatStd()
	return (rand.Float64() * 2) - 1
}

func average(arr []float64) float64 {
	accumulator := 0.0
	for i := 0; i < len(arr); i++ {
		accumulator += arr[i]
	}
	accumulator = accumulator / float64(len(arr))
	return accumulator
}

func sum(arr []float64) float64 {
	accumulator := 0.0
	for i := 0; i < len(arr); i++ {
		accumulator += arr[i]
	}
	return accumulator
}
