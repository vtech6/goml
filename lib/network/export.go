package network

import (
	"encoding/json"
	"log"
	"os"
)

type SavedData struct {
	TrainX           [][]float64
	TrainY           [][]float64
	TestX            [][]float64
	TestY            [][]float64
	PredictionsTrain []float64
	PredictionsTest  []float64
	Accuracy         float64
}

func saveData(data SavedData) SavedData {
	file, error := json.MarshalIndent(data, "", " ")
	if error != nil {
		log.Fatal(error)
	}
	error = os.WriteFile("IrisOutput.json", file, 0644)
	return data
}
