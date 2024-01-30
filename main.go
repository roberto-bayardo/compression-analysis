package main

import (
	"bytes"
	"compress/zlib"
	"context"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"reflect"
	"runtime"
	"text/tabwriter"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethclient"
	fastlz "github.com/fananchong/fastlz-go"
	"github.com/montanaflynn/stats"
	"github.com/sajari/regression"
)

type estimator func([]byte) float64

func main() {
	blockNum := big.NewInt(9885000) // starting block; will iterate backwards from here
	txsToFetch := 2000              // min # of transactions to include in our sample
	minTxSize := 0                  // minimum transaction size to include in our sample whether to

	// spanBatchMode will remove signatures from the tx rlp before compression. This simulates the
	// behavior of span batches which segregates the signatures from the more compressible parts of
	// the tx during batch compression.
	spanBatchMode := true

	// whether to build a regression model instead of simple scalar-tuned estimator
	useRegression := true
	// when using regression, whether to also use the uncompressed tx size as a feature
	uncompressedSizeFeature := true

	bootstrapTxs := 100 // min number of txs to use to bootstrap the batch compressor

	fmt.Printf("Starting block: %v, min tx sample size: %v, min tx size: %v, span batch mode: %v\n",
		blockNum, txsToFetch, minTxSize, spanBatchMode)
	if useRegression {
		if uncompressedSizeFeature {
			fmt.Println("Using regression with estimator + uncompress-tx-size as features")
		} else {
			fmt.Println("Using regression with simple estimator as the only feature")
		}
	}

	ONE := big.NewInt(1)

	//client, err := ethclient.Dial("https://mainnet.base.org")
	client, err := ethclient.Dial("https://base-mainnet-dev.cbhq.net:8545")
	if err != nil {
		log.Fatal(err)
	}

	estimators := []estimator{
		uncompressedSizeEstimator,
		cheap0Estimator,
		cheap1Estimator,
		cheap2Estimator,
		cheap3Estimator,
		cheap4Estimator,
		cheap5Estimator,
		cheap6Estimator,
		cheap7Estimator,
		cheap8Estimator,
		repeatedByte0Estimator,
		repeatedByte1Estimator,
		repeatedByte3Estimator,
		repeatedOrZeroEstimator,
		cheap2Estimator,
		cheap3Estimator,
		cheap4Estimator,
		fastLZEstimator,
		zlibBestEstimator,
		zlibBestBatchEstimator, // final estimator value is always used as the "ground truth" against which others are measured
	}
	columns := make([][]float64, len(estimators))

	bootstrapCount := 0

	for {
		block, err := client.BlockByNumber(context.Background(), blockNum)
		if err != nil {
			log.Fatal(err)
		}
		//fmt.Println("Blocknum:", blockNum, "Txs:", len(columns[0]))
		for _, tx := range block.Transactions() {
			if tx.Type() == types.DepositTxType {
				continue
			}
			b, err := tx.MarshalBinary()
			if err != nil {
				log.Fatal(err)
			}
			if len(b) < minTxSize {
				continue
			}
			if spanBatchMode {
				// for span batch mode we trim the signature, and assume there is no estimation
				// error on this component were we to just treat it as entirely uncompressible.
				b = b[:len(b)-68.]
			}
			if bootstrapCount < bootstrapTxs {
				zlibBestBatchEstimator(b)
				bootstrapCount++
				continue
			}
			for j := range estimators {
				estimate := estimators[j](b)
				columns[j] = append(columns[j], estimate)
			}
			if len(columns[0])%1000 == 0 {
				fmt.Println(len(columns[0]), "out of", txsToFetch)
			}
		}
		if len(columns[0]) > txsToFetch {
			break
		}
		blockNum.Sub(blockNum, ONE)
	}

	// compute normalizers to eliminate estimator bias reflecting what a chain operator does via
	// scalar tuning
	avgs := []float64{}
	for j := range columns {
		avg, err := stats.Mean(stats.Float64Data(columns[j]))
		if err != nil {
			log.Fatal(err)
		}
		avgs = append(avgs, avg)
	}
	fmt.Println()
	prettyPrintStats("mean", estimators, avgs)

	scalars := make([]float64, len(avgs))
	if !useRegression {
		for j := range columns {
			scalars[j] = avgs[len(avgs)-1] / avgs[j]
		}
		fmt.Println()
		prettyPrintStats("scalar", estimators, scalars)
	}

	// Create regressors for each estimator
	reg := make([]regression.Regression, len(estimators))
	if useRegression {
		for j := range estimators {
			reg[j].SetObserved("bytes after batch compression")
			reg[j].SetVar(0, getFuncName(estimators[j]))
			if uncompressedSizeFeature {
				reg[j].SetVar(1, fmt.Sprintf("uncompressed bytes"))
			}
		}
		for j := range estimators {
			for i := range columns[j] {
				truth := columns[len(scalars)-1][i]
				estimator := columns[j][i]
				data := []float64{estimator}
				if uncompressedSizeFeature {
					data = append(data, columns[0][i]) // assumes the "uncompressed estimator" is always first
				}
				reg[j].Train(regression.DataPoint(truth, data))
			}
			reg[j].Run()
			fmt.Printf("Regression %v:\n%v\n", getFuncName(estimators[j]), reg[j].Formula)
			fmt.Println("R^2:", reg[j].R2)
			//fmt.Printf("Regression %d        :\n%s\n", j, reg[j])
		}
	}

	// compute per-tx error values
	absoluteErrors := make([][]float64, len(estimators))
	squaredErrors := make([][]float64, len(estimators))

	for j := range estimators {
		ae := make([]float64, len(columns[j]))
		se := make([]float64, len(columns[j]))
		scalar := scalars[j]
		for i := range columns[j] {
			// output of the final estimator (which we assume to be the batched compression
			// algorithm actually used by the batcher) is used as the "ground truth".
			truth := columns[len(scalars)-1][i]
			var estimate float64
			if !useRegression {
				// the estimate is the scaled output of the estimator
				estimate = columns[j][i] * scalar
			} else {
				data := []float64{columns[j][i]}
				if uncompressedSizeFeature {
					data = append(data, columns[0][i]) // assumes the "uncompressed estimator" is always first
				}
				estimate, err = reg[j].Predict(data)
				if err != nil {
					panic(err)
				}
			}
			e := estimate - truth
			ae[i] = math.Abs(e)
			se[i] = math.Pow(e, 2)
		}
		absoluteErrors[j] = ae
		squaredErrors[j] = se
	}

	// compute mean error metrics
	var mass []float64
	for j := range estimators {
		mas, err := stats.Mean(stats.Float64Data(absoluteErrors[j]))
		if err != nil {
			log.Fatal(err)
		}
		mass = append(mass, mas)
	}
	fmt.Println()
	prettyPrintStats("mean-absolute-error", estimators, mass)

	var rmses []float64
	for j := range estimators {
		mse, err := stats.Mean(stats.Float64Data(squaredErrors[j]))
		if err != nil {
			log.Fatal(err)
		}
		rmses = append(rmses, math.Sqrt(mse))
	}
	fmt.Println()
	prettyPrintStats("root-mean-sq-error", estimators, rmses)

}

func prettyPrintStats(prefix string, estimators []estimator, stats []float64) {
	w := tabwriter.NewWriter(os.Stdout, 1, 1, 1, ' ', 0)

	for j := range estimators {
		fmt.Fprintf(w, "%v\t%v\t%v\n", prefix, getFuncName(estimators[j]), stats[j])
	}
	w.Flush()
}

func getFuncName(f interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()[5:] // trim off "main." prefix
}

// only count bytes that are non-zero and non-repeated
func repeatedOrZeroEstimator(tx []byte) float64 {
	lastByte := byte(0)
	count := 0
	for _, b := range tx {
		if b != lastByte && b != 0 {
			count += 1
		}
		lastByte = b
	}
	return float64(count)
}

func repeatedByteEstimator(tx []byte, repeatedByteCost, changedByteCost int) float64 {
	lastByte := byte(0)
	count := 0
	for _, b := range tx {
		if b == lastByte {
			count += repeatedByteCost
		} else {
			count += changedByteCost
		}
		lastByte = b
	}
	return float64(count) / float64(repeatedByteCost+changedByteCost)
}

func repeatedByte0Estimator(tx []byte) float64 {
	return repeatedByteEstimator(tx, 0, 16)
}

func repeatedByte1Estimator(tx []byte) float64 {
	return repeatedByteEstimator(tx, 1, 16)
}

func repeatedByte2Estimator(tx []byte) float64 {
	return repeatedByteEstimator(tx, 2, 16)
}

func repeatedByte3Estimator(tx []byte) float64 {
	return repeatedByteEstimator(tx, 3, 16)
}

func repeatedByte4Estimator(tx []byte) float64 {
	return repeatedByteEstimator(tx, 4, 16)
}

func cheapEstimator(tx []byte, zeroByteCost int, nonZeroByteCost int) float64 {
	count := 0
	for _, b := range tx {
		if b == 0 {
			count += zeroByteCost
		} else {
			count += nonZeroByteCost
		}
	}
	return float64(count) / float64(zeroByteCost+nonZeroByteCost)
}

var (
	b           bytes.Buffer
	batchWriter *zlib.Writer
)

func init() {
	var err error
	batchWriter, err = zlib.NewWriterLevel(&b, zlib.BestCompression)
	if err != nil {
		log.Fatal(err)
	}
}

// zlibBestBatchEstimator simulates a zlib compressor at max compression that works on (large) tx
// batches.  Should bootstrap it before use by calling it on several samples of representative
// data.
func zlibBestBatchEstimator(tx []byte) float64 {
	beginLen := b.Len()
	batchWriter.Write(tx)
	batchWriter.Flush()
	return float64(b.Len() - beginLen - 2) // flush writes 2 extra "sync" bytes so don't count those
}

func zlibBestEstimator(tx []byte) float64 {
	var b bytes.Buffer
	w, err := zlib.NewWriterLevel(&b, zlib.BestCompression)
	if err != nil {
		log.Fatal(err)
	}
	defer w.Close()
	w.Write(tx)
	w.Flush()                   // flush instead of close to not include the digest
	return float64(b.Len() - 2) // flush writes 2 extra "sync" bytes so don't count those
}

func fastLZEstimator(tx []byte) float64 {
	ol := int(float64(len(tx)) * 1.1)
	out := make([]byte, ol)
	sz := fastlz.Fastlz_compress(tx, len(tx), out)
	return float64(sz)
}

func uncompressedSizeEstimator(tx []byte) float64 {
	return float64(len(tx))
}

func cheap0Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 0, 16)
}

// simulate if we could a 0.5 score for zero bytes
func cheapP5Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 1, 32)
}

func cheap1Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 1, 16)
}

func cheap2Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 2, 16)
}

func cheap3Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 3, 16)
}

// cheap4Estimator is the actual L1 Data Cost estimator currently in use
func cheap4Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 4, 16)
}

func cheap5Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 5, 16)
}

func cheap6Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 6, 16)
}

func cheap7Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 7, 16)
}

func cheap8Estimator(tx []byte) float64 {
	return cheapEstimator(tx, 8, 16)
}
