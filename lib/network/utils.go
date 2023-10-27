package network

import (
	"encoding/csv"
	"log"
	"math/rand"
	"os"
	"strconv"
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

func loadIrisData() ([][]float64, [][]float64) {
	file, error := os.Open("./lib/dataset/IRIS.csv")
	if error != nil {
		log.Fatal(error)
	}
	defer file.Close()

	csvReader := csv.NewReader(file)
	data, error := csvReader.ReadAll()
	if error != nil {
		log.Fatal(error)
	}
	return createIrisData(data)
}

func createIrisData(data [][]string) ([][]float64, [][]float64) {
	labels := []string{"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
	xValues := make([][]float64, 0)
	yValues := make([][]float64, 0)
	for i, line := range data {
		if i > 0 {
			_xValues := make([]float64, 0)
			for j, entry := range line {
				if j < 4 {
					value, error := strconv.ParseFloat(entry, 64)
					if error != nil {
						log.Fatal(error)
					}

					_xValues = append(_xValues, value)
				}
				if j == 4 {
					for labelIndex, label := range labels {
						if label == entry {
							yValues = append(yValues, []float64{float64(labelIndex)})
						}
					}
				}

			}

			xValues = append(xValues, _xValues)
		}
	}
	return xValues, yValues
}
