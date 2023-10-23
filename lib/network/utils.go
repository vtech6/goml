package network

import (
	"math"
	"math/rand"
)

const (
	TestMode = false
)

func randomFloatStd() float64 {
	if TestMode == true {
		return 1
	} else {
		return rand.Float64()
	}
}

func generateBias() float64 {
	if TestMode == true {

		return 0
	} else {
		return rand.Float64()
	}
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

func reshape(arr []float64) [][]float64 {
	newShape := make([][]float64, len(arr))
	for i := 0; i < len(arr); i++ {
		newShape[i][0] = arr[i]
	}
	return newShape
}

func sigmoid(input float64) float64 {
	return 1.0 / (1.0 + math.Exp(-input))
}

func normalizeData(arr []Input) []Input {
	normalizedArr := make([]Input, len(arr))
	maxX := 0.0
	maxY := 0.0
	for i := 0; i < len(arr); i++ {
		if arr[i].x[0] > maxX {
			maxX = arr[i].x[0]
		}
		if arr[i].y[0] > maxY {
			maxY = arr[i].y[0]
		}
	}
	for i := 0; i < len(arr); i++ {
		normalizedArr[i] = arr[i]
		normalizedArr[i].x[0] = arr[i].x[0] / maxX
		normalizedArr[i].y[0] = arr[i].y[0] / maxY
	}
	return normalizedArr
}

func zip(arr1 []float64, arr2 []float64) ([][]float64, int) {
	output := make([][]float64, len(arr1))
	maxLen := len(arr1)
	if len(arr2) < len(arr1) {
		maxLen = len(arr2)
	}
	for i := 0; i < maxLen; i++ {
		output[i] = make([]float64, 2)
		output[i][0] = arr1[i]
		output[i][1] = arr2[i]
	}
	return output, maxLen
}

func contains[T comparable](s []T, e T) bool {
	for _, v := range s {
		if v == e {
			return true
		}
	}
	return false
}
