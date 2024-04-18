package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"text/tabwriter"

	"github.com/montanaflynn/stats"
)

var (
	flzJanBaseScalarModel = func(row []float64) float64 {
		return row[0] * 0.611
	}
	flzJanBaseModel   = makeModelFromCoefficients([]float64{-4.8366, .6295})
	flzJanBase2DModel = makeModelFromCoefficients([]float64{-11.3963, 0.7763, -0.0646})

	flzEcotoneBaseScalarModel = func(row []float64) float64 {
		return row[0] * 0.587
	}
	flzEcotoneBaseModel   = makeModelFromCoefficients([]float64{-50.0398, 0.7706})
	flzEcotoneBase2DModel = makeModelFromCoefficients([]float64{-42.5088, 0.8726, -0.0647})

	flzOctOPScalarModel = func(row []float64) float64 {
		return row[0] * 0.707
	}
	flzOctOPModel   = makeModelFromCoefficients([]float64{-22.0716, 0.7767})
	flzOctOP2DModel = makeModelFromCoefficients([]float64{-10.1930, 0.9573, -0.1272})

	flzEcotoneOPScalarModel = func(row []float64) float64 {
		return row[0] * 0.721
	}
	flzEcotoneOPModel   = makeModelFromCoefficients([]float64{-42.5856, 0.8365})
	flzEcotoneOP2DModel = makeModelFromCoefficients([]float64{-39.3474, 0.9319, -0.0501})

	cheap4JanBaseScalarModel = func(row []float64) float64 {
		return row[0] * 0.643
	}
	cheap4JanBaseModel   = makeModelFromCoefficients([]float64{88.5893, 0.2801})
	cheap4JanBase2DModel = makeModelFromCoefficients([]float64{78.4145, 0.9583, -0.3227})

	cheap4OctOPScalarModel = func(row []float64) float64 {
		return row[0] * 0.807
	}
	cheap4OctOPModel   = makeModelFromCoefficients([]float64{-2.5313, 0.8174})
	cheap4OctOP2DModel = makeModelFromCoefficients([]float64{22.5926, 1.3793, -0.3344})

	cheap4EcotoneBaseScalarModel = func(row []float64) float64 {
		return row[0] * 0.633
	}
	cheap4EcotoneBaseModel   = makeModelFromCoefficients([]float64{-37.2661, 0.7809})
	cheap4EcotoneBase2DModel = makeModelFromCoefficients([]float64{-22.8951, 1.2074, -0.2241})

	cheap4EcotoneOPScalarModel = func(row []float64) float64 {
		return row[0] * 0.744
	}
	cheap4EcotoneOPModel   = makeModelFromCoefficients([]float64{-13.0282, 0.7806})
	cheap4EcotoneOP2DModel = makeModelFromCoefficients([]float64{-7.5084, 1.4377, -0.3140})
)

func makeModelFromCoefficients(coefficients []float64) model {
	r := &regression{
		coefficients: coefficients,
	}
	return func(row []float64) float64 {
		p := r.Predict(row)
		if p < compressedSizeFloor {
			p = compressedSizeFloor
		}
		return p
	}
}

type namedModel struct {
	name  string
	model model
}

func doEvaluate(columns [][]float64) {
	w := tabwriter.NewWriter(os.Stdout, 10, 1, 1, ' ', tabwriter.AlignRight)
	fmt.Fprintf(w, "%v\t%v\t%v\t\n", "MODEL", "MAE", "RMSE")

	flzModels := []namedModel{
		namedModel{"flzJanBaseScalarModel", flzJanBaseScalarModel},
		namedModel{"flzJanBaseModel", flzJanBaseModel},
		namedModel{"flzEcotoneBaseScalarModel", flzEcotoneBaseScalarModel},
		namedModel{"flzEcotoneBaseModel", flzEcotoneBaseModel},
		namedModel{"flzOctOPScalarModel", flzOctOPScalarModel},
		namedModel{"flzOctOPModel", flzOctOPModel},
		namedModel{"flzEcotoneOPScalarModel", flzEcotoneOPScalarModel},
		namedModel{"flzEcotoneOPModel", flzEcotoneOPModel},
	}
	indices := []int{2}
	evaluateModels(w, columns, indices, flzModels)
	return

	flz2DModels := []namedModel{
		namedModel{"flzJanBase2DModel", flzJanBase2DModel},
		namedModel{"flzEcotoneBase2DModel", flzEcotoneBase2DModel},
		namedModel{"flzOctOP2DModel", flzOctOP2DModel},
		namedModel{"flzEcotoneOP2DModel", flzEcotoneOP2DModel},
	}
	indices = []int{2, 0}
	evaluateModels(w, columns, indices, flz2DModels)

	cheapModels := []namedModel{
		namedModel{"cheap4JanBaseScalarModel", cheap4JanBaseScalarModel},
		namedModel{"cheap4JanBaseModel", cheap4JanBaseModel},
		namedModel{"cheap4EcotoneBaseScalarModel", cheap4EcotoneBaseScalarModel},
		namedModel{"cheap4EcotoneBaseModel", cheap4EcotoneBaseModel},
		namedModel{"cheap4OctOPScalarModel", cheap4OctOPScalarModel},
		namedModel{"cheap4OctOPModel", cheap4OctOPModel},
		namedModel{"cheap4EcotoneOPScalarModel", cheap4EcotoneOPScalarModel},
		namedModel{"cheap4EcotoneOPModel", cheap4EcotoneOPModel},
	}
	indices = []int{1}
	evaluateModels(w, columns, indices, cheapModels)

	cheap2DModels := []namedModel{
		namedModel{"cheap4JanBase2DModel", cheap4JanBase2DModel},
		namedModel{"cheap4EcotoneBase2DModel", cheap4EcotoneBase2DModel},
		namedModel{"cheap4OctOP2DModel", cheap4OctOP2DModel},
		namedModel{"cheap4EcotoneOP2DModel", cheap4EcotoneOP2DModel},
	}
	indices = []int{1, 0}
	evaluateModels(w, columns, indices, cheap2DModels)

	w.Flush()
}

var formatString = "%v\t%.3f\t%.3f\t\n"

func evaluateModels(w *tabwriter.Writer, c [][]float64, indices []int, models []namedModel) {
	for i := range models {
		mae, rmse := evaluateModel(c, indices, models[i].model)
		fmt.Fprintf(w, formatString, models[i].name, mae, rmse)
	}
}

func evaluateModel(columns [][]float64, indices []int, model model) (mae, rmse float64) {
	absoluteErrors := make([]float64, len(columns[0]))
	squaredErrors := make([]float64, len(columns[0]))

	maxPosError := 0.0
	maxNegError := 0.0
	for i := range columns[0] {
		// final column is always used as ground truth
		truth := columns[len(columns)-1][i]
		var estimate float64
		data := make([]float64, len(indices))
		for j := range indices {
			data[j] = columns[indices[j]][i]
		}
		estimate = model(data)
		e := estimate - truth
		percentDiff := e / estimate
		if percentDiff < maxNegError {
			maxNegError = percentDiff
		}
		if percentDiff < -1.0 {
			fmt.Println(":::", percentDiff, estimate, truth, data, columns[0][i], columns[3][i])
		}
		if e > maxPosError {
			maxPosError = e
			//fmt.Println("New max pos error:", maxPosError, estimate, truth, data, columns[0][i], columns[3][i])
		}
		absoluteErrors[i] = math.Abs(e)
		squaredErrors[i] = math.Pow(e, 2)
	}

	mae, err := stats.Mean(stats.Float64Data(absoluteErrors))
	if err != nil {
		log.Fatalln(err)
	}
	mse, err := stats.Mean(stats.Float64Data(squaredErrors))
	if err != nil {
		log.Fatalln(err)
	}
	rmse = math.Sqrt(mse)
	return mae, rmse
}
