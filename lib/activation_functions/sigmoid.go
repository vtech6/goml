package activation_functions

import "math"

func Sigmoid(input float64) float64 {
	return 1.0 / (1.0 + math.Exp(-input))
}
