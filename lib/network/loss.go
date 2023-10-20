package network

func meanSquaredError(predictions []float64, target float64) float64 {
	mse := 0.0
	for i := 0; i < len(predictions); i++ {
		mse += ((target - predictions[i]) * (target - predictions[i]))
	}
	return mse / float64(len(predictions))
}
