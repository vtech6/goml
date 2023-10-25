package network

import (
	"math/rand"
)

const (
	TestMode = false
)

func randomFloatStd() float64 {
	if TestMode == true {
		return 1
	} else {
		return (rand.Float64() * 2) - 1
	}
}

func generateBias() float64 {
	if TestMode == true {

		return 0
	} else {
		return randomFloatStd()
	}
}
