package main

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"log"
	"math"
	//"math/big"
	"os"
	"reflect"
	"runtime"
	"text/tabwriter"

	"github.com/google/brotli/go/cbrotli"
	"github.com/montanaflynn/stats"
)

// type estimator takes as input a serialized transaction and outputs a score indended to be
// predictive of what the batch-compressed size of that transaction might be.
type estimator func([]byte) float64

// type model takes as input a feature vector for a serialized transaction and outputs a prediction
// for the size of that transaction after batch compression.
type model func(data []float64) float64

var (
	txsToFetch = 2000 // max # of transactions to include in our sample

	// If this is true, then the functions will be derived on the oldest half of the transactions,
	// and evaluated on the newer half.
	separateTrainTest = false

	// spanBatchMode will remove signatures from the tx rlp before compression. This simulates the
	// behavior of span batches which segregates the signatures from the more compressible parts of
	// the tx during batch compression.
	spanBatchMode = true

	// numBlobs simulates batching with a given number of blobs per transaction. More blobs allows
	// for more batch compression, thus this affects the ground truth.
	numBlobs = 6

	// numTestPartitions is the number of equal-sized partitions of test set data over which to
	// compute error metrics, each partition being from a period successively more distant from the
	// training set transactions.
	numTestPartitions = 5

	// compressedSizeFloor is the smallest size we ever expect a transaction to reach after batch
	// compression.  It is used to lower-bound model outputs.
	compressedSizeFloor = 71.
)

type regression struct {
	coefficients []float64
}

type TxIterator interface {
	Next() []byte
	Close()
}

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}

func main() {
	fmt.Printf("Tx sample size: %v, span batch mode: %v, num blobs: %v\n", txsToFetch, spanBatchMode, numBlobs)

	//clientLocation = "https://mainnet.base.org"
	//clientLocation = "https://base-mainnet-dev.cbhq.net:8545"
	//clientLocation = "/data"
	//blockNum   = big.NewInt(12000000) // starting block; will iterate backwards from here
	//txIter := NewTxIterRPC(clientLocation, blockNum)

	//txFilename := "/Users/bayardo/tmp/op_post_ecotone_txs_result.bin"
	//txFilename := "/Users/bayardo/tmp/base_post_ecotone_txs.bin"
	txFilename := "/Users/bayardo/tmp/op-txs-oct-2024.bin"
	txIter := NewTxIterFile(txFilename)
	defer txIter.Close()

	if separateTrainTest {
		fmt.Println("Training over the older half of transactions, evaluating over the newer half.")
	} else {
		fmt.Println("Evaluating over the same set of transactions used to compute the regression.")
	}

	estimators := []estimator{
		uncompressedSizeEstimator, // first estimator should always return the length of the input
		//cheap0Estimator,
		//cheap1Estimator,
		//cheap2Estimator,
		//cheap3Estimator,
		cheap4Estimator,
		//cheap5Estimator,
		//repeatedByte0Estimator,
		//repeatedByte1Estimator,
		//repeatedByte2Estimator,
		//repeatedOrZeroEstimator,
		fastLZEstimator,
		//zlibBestEstimator,
		//brotliEstimator,
		zlibBestBatchEstimator,
		brotliBatchEstimator, // final estimator value is always used as the "ground truth" against which others are measured
	}
	columns := make([][]float64, len(estimators))

	log.Println("bootstrapping")
	bootstrapCount := 0
	for b := txIter.Next(); !zlibBatchEstimatorObj.bootstrapped || !brotliBatchEstimatorObj.bootstrapped; b = txIter.Next() {
		if b == nil {
			log.Fatalln("ran out of transactions bootstrapping")
		}
		zlibBestBatchEstimator(b)
		brotliBatchEstimator(b)
		bootstrapCount++
	}
	log.Println("finished bootstrapping over", bootstrapCount, "transactions")

	for b := txIter.Next(); ; b = txIter.Next() {
		if b == nil {
			log.Println("ran out of transactions, exiting retrieval")
			break
		}
		for j := range estimators {
			estimate := estimators[j](b)
			columns[j] = append(columns[j], estimate)
		}
		if len(columns[0])%10000 == 0 {
			log.Println(len(columns[0]), "out of", txsToFetch)
		}
		if len(columns[0]) == txsToFetch {
			log.Println("Reached tx limit.")
			break
		}
	}

	start := 0
	end := len(columns[0])
	if separateTrainTest {
		// train only on the older transactions (those that come last)
		start = end / 2
	}
	trainColumns := make([][]float64, len(estimators))
	for j := range trainColumns {
		trainColumns[j] = columns[j][start:end]
	}
	start = 0
	end = len(columns[0])
	if separateTrainTest {
		// evaluate the functions only over newer transactions (those that came first)
		end = end / 2
	}
	testColumns := make([][]float64, len(estimators))
	for j := range testColumns {
		testColumns[j] = columns[j][start:end]
	}

	if !separateTrainTest {
		numTestPartitions = 0
	}
	if separateTrainTest {
		fmt.Println("\n========= TRAINING SET SUMMARY STATS =============")
		avgs := computeMeans(trainColumns)
		fmt.Println()
		prettyPrintStats("mean", estimators, avgs)
	}

	fmt.Println("\n======== FULL TEST SET SUMMARY STATS ============")
	avgs := computeMeans(testColumns)
	fmt.Println()
	prettyPrintStats("mean", estimators, avgs)

	for i := 0; i < numTestPartitions; i++ {
		partitionSize := len(testColumns[0]) / numTestPartitions
		start = i * partitionSize
		end = start + partitionSize - 1
		fmt.Printf("\n======= TEST SET SUMMARY STATS (partition #%d) ========\n", i)
		partitionColumns := make([][]float64, len(estimators))
		for j := range partitionColumns {
			partitionColumns[j] = testColumns[j][start:end]
		}
		avgs := computeMeans(partitionColumns)
		fmt.Println()
		prettyPrintStats("mean", estimators, avgs)
	}

	fmt.Println("\nScalar models:")
	avgs = computeMeans(trainColumns)
	scalarModels := make([]model, len(estimators))
	scalars := make([]float64, len(avgs))
	// compute normalizers to eliminate estimator bias reflecting what a chain operator does via
	// scalar tuning, and use the normalized estimator as our "prediction"
	for j := range estimators {
		scalar := avgs[len(avgs)-1] / avgs[j]
		scalars[j] = scalar
		scalarModels[j] = func(data []float64) float64 {
			return data[0] * scalar
		}
	}
	fmt.Println()
	prettyPrintStats("scalar", estimators, scalars)

	singleFeatureRegressionModels := doRegression(estimators, trainColumns, false)
	twoFeatureRegressionModels := doRegression(estimators, trainColumns, true)

	if separateTrainTest {
		// print out the training set performance stats separately from the test set
		scalarMae, scalarRmse := evaluate(trainColumns, scalarModels)
		regMae, regRmse := evaluate(trainColumns, singleFeatureRegressionModels)
		twoMae, twoRmse := evaluate(trainColumns, twoFeatureRegressionModels)
		fmt.Println("\n========= TRAINING SET STATS: SCALAR MODEL, 1D REGRESSION, 2D REGRESSION ==========\n")
		prettyPrintStats("mean-absolute-error", estimators, scalarMae, regMae, twoMae)
		fmt.Println()
		prettyPrintStats("root-mean-sq-error ", estimators, scalarRmse, regRmse, twoRmse)
	}

	scalarMae, scalarRmse := evaluate(testColumns, scalarModels)
	regMae, regRmse := evaluate(testColumns, singleFeatureRegressionModels)
	twoMae, twoRmse := evaluate(testColumns, twoFeatureRegressionModels)
	fmt.Println("\n====== FULL TEST SET STATS: SCALAR MODEL, 1D REGRESSION, 2D REGRESSION ======\n")
	prettyPrintStats("mean-absolute-error", estimators, scalarMae, regMae, twoMae)
	fmt.Println()
	prettyPrintStats("root-mean-sq-error ", estimators, scalarRmse, regRmse, twoRmse)

	for i := 0; i < numTestPartitions; i++ {
		partitionSize := len(testColumns[0]) / numTestPartitions
		start = i * partitionSize
		end = start + partitionSize - 1
		partitionColumns := make([][]float64, len(estimators))
		for j := range partitionColumns {
			partitionColumns[j] = testColumns[j][start:end]
		}
		scalarMae, scalarRmse := evaluate(partitionColumns, scalarModels)
		regMae, regRmse := evaluate(partitionColumns, singleFeatureRegressionModels)
		twoMae, twoRmse := evaluate(partitionColumns, twoFeatureRegressionModels)
		fmt.Printf("\n====== TEST SET STATS (partition %d): SCALAR MODEL, 1D REGRESSION, 2D REGRESSION ======\n\n", i)
		prettyPrintStats("mean-absolute-error", estimators, scalarMae, regMae, twoMae)
		fmt.Println()
		prettyPrintStats("root-mean-sq-error ", estimators, scalarRmse, regRmse, twoRmse)
	}
}

func evaluateModel(columns [][]float64, model model) (mae, rmse float64) {
	absoluteErrors := make([]float64, len(columns[0]))
	squaredErrors := make([]float64, len(columns[0]))

	for i := range columns[0] {
		// final column is always used as ground truth
		truth := columns[len(columns)-1][i]
		var estimate float64
		data := make([]float64, len(columns)-1)
		for j := 0; j < len(columns)-1; j++ {
			data[j] = columns[j][i]
		}
		estimate = model(data)
		e := estimate - truth
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

func evaluate(columns [][]float64, models []model) (mae []float64, rmse []float64) {
	// compute per-tx error values
	absoluteErrors := make([][]float64, len(columns))
	squaredErrors := make([][]float64, len(columns))

	for j := range columns {
		ae := make([]float64, len(columns[j]))
		se := make([]float64, len(columns[j]))
		for i := range columns[j] {
			// final column (which we assume to be the batched compression algorithm actually used
			// by the batcher) is used as the "ground truth".
			truth := columns[len(columns)-1][i]
			var estimate float64
			data := []float64{columns[j][i], columns[0][i]}
			estimate = models[j](data)
			e := estimate - truth
			ae[i] = math.Abs(e)
			se[i] = math.Pow(e, 2)
		}
		absoluteErrors[j] = ae
		squaredErrors[j] = se
	}

	// compute mean error metrics
	mae = []float64{}
	for j := range columns {
		mas, err := stats.Mean(stats.Float64Data(absoluteErrors[j]))
		if err != nil {
			log.Fatalln(err)
		}
		mae = append(mae, mas)
	}
	rmse = []float64{}
	for j := range columns {
		mse, err := stats.Mean(stats.Float64Data(squaredErrors[j]))
		if err != nil {
			log.Fatalln(err)
		}
		rmse = append(rmse, math.Sqrt(mse))
	}
	return mae, rmse
}

func (r *regression) Learn(rows [][]float64, y []float64) error {
	// performs batch gradient descent with momentum
	fmt.Println()
	log.Println("Learning....")
	alpha := .00001 // alpha higher than this tends to result in divergence for this data
	momentum := .99 // very high momentum seems to work best for this data

again:
	r.coefficients = make([]float64, len(rows[0])+1)
	b, _ := r.gradient(rows, y)
	lastMse := 0.

	for i := 0; i < 1000000; i++ {
		for j := range b {
			r.coefficients[j] = r.coefficients[j] - (b[j] * alpha)
		}
		g, mse := r.gradient(rows, y)
		if math.IsNaN(mse) {
			alpha /= 2
			log.Println("Model diverging, cutting alpha:", alpha)
			goto again
		}
		// check for convergence
		if math.Abs(mse-lastMse) < .000001 {
			log.Println("Converged at iteration:", i)
			break
		}
		lastMse = mse
		for j := range b {
			b[j] = momentum*b[j] + g[j]
		}
	}

	return nil
}

// returns the mean gradient of the model for the given dataset
func (r *regression) gradient(rows [][]float64, y []float64) ([]float64, float64) {
	gradient := make([]float64, len(rows[0])+1)
	var mse float64
	for i := range rows {
		row := rows[i]
		p := r.Predict(row)
		e := p - y[i]
		mse += e * e
		gradient[0] += e
		for j := range row {
			gradient[j+1] += e * row[j]
		}
	}
	for j := range gradient {
		gradient[j] /= float64(len(rows))
	}
	return gradient, mse / float64(len(rows))
}

func (r *regression) Predict(row []float64) float64 {
	sum := r.coefficients[0]
	for i := range row {
		sum += r.coefficients[i+1] * row[i]
	}
	return sum
}

func (r regression) String() string {
	str := fmt.Sprintf("%.4f", r.coefficients[0])
	for i := 1; i < len(r.coefficients); i++ {
		str += fmt.Sprintf(" + %.4f*x_%d", r.coefficients[i], i-1)
	}
	return str
}

func doRegression(estimators []estimator, columns [][]float64, uncompressedSizeFeature bool) []model {
	// create a linear regression model for each simple estimator
	models := make([]model, len(estimators))
	truth := columns[len(columns)-1]
	for j := range estimators {
		var featureRows [][]float64
		for i := range columns[j] {
			var data []float64
			estimator := columns[j][i]
			if uncompressedSizeFeature {
				data = []float64{estimator, columns[0][i]} // assumes the "uncompressed estimator" is always first
			} else {
				data = []float64{estimator}
			}
			featureRows = append(featureRows, data)
		}
		reg := regression{}
		if j == len(estimators)-1 {
			// we already know coefficients for an estimator whose output always equals the ground
			// truth, though if you change the learning algorithm it makes sense to verify it's
			// deriving the expected coefficients even in this case.
			reg.coefficients = []float64{0.0, 1.0, 0.0}
		} else {
			err := reg.Learn(featureRows, truth)
			if err != nil {
				log.Fatalln(err)
			}
		}
		fmt.Printf("\nRegression %v: %v\n", getFuncName(estimators[j]), reg)
		models[j] = func(row []float64) float64 {
			if !uncompressedSizeFeature {
				row = row[:1]
			}
			r := reg.Predict(row)
			if r < compressedSizeFloor {
				r = compressedSizeFloor
			}
			return r
		}
	}
	return models
}

func computeMeans(columns [][]float64) []float64 {
	avgs := []float64{}
	for j := range columns {
		avg, err := stats.Mean(stats.Float64Data(columns[j]))
		if err != nil {
			log.Fatalln(err)
		}
		avgs = append(avgs, avg)
	}
	return avgs
}

func prettyPrintStats(prefix string, estimators []estimator, stats ...[]float64) {
	w := tabwriter.NewWriter(os.Stdout, 10, 1, 1, ' ', tabwriter.AlignRight)

	formatString := "%v\t%v\t"
	for i := 0; i < len(stats); i++ {
		formatString += "%.3f\t"
	}
	formatString += "\n"
	for j := range estimators {
		row := make([]any, len(stats)+2)
		row[0] = prefix
		row[1] = getFuncName(estimators[j])
		for i := range stats {
			row[i+2] = stats[i][j]
		}
		fmt.Fprintf(w, formatString, row...)
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

var b bytes.Buffer
var w, _ = zlib.NewWriterLevel(&b, zlib.BestCompression)

// zlibBestEstimator runs a freshly reset zlib compressor at BestCompression level and returns the
// length of the result.  This function is not thread safe.
func zlibBestEstimator(tx []byte) float64 {
	b.Reset()
	w.Reset(&b)
	w.Write(tx)
	w.Flush()                   // flush instead of close to not include the digest
	return float64(b.Len() - 2) // flush writes 2 extra "sync" bytes so don't count those
}

var b2 bytes.Buffer

// brotliEstimator runs a freshly reset brotli compressor at level 10.
// This function is not thread safe.
func brotliEstimator(tx []byte) float64 {
	b2.Reset()
	w2 := cbrotli.NewWriter(&b2, cbrotli.WriterOptions{Quality: 10})
	w2.Write(tx)
	w2.Flush()                   // flush instead of close to not include the digest
	return float64(b2.Len() - 2) // flush writes 2 extra "sync" bytes so don't count those
}

func fastLZEstimator(tx []byte) float64 {
	return float64(flzCompressLen(tx))
}

// uncompressedSizeEstimator just returns the length of the input
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

var (
	// buffer for flzCompressLen -- need to fix this if we start using it concurrently
	ht = make([]uint32, 8192)
)

func flzCompressLen(ib []byte) uint32 {
	for i := 0; i < len(ht); i++ {
		ht[i] = 0
	}
	ht = make([]uint32, 8192)
	n := uint32(0)
	u24 := func(i uint32) uint32 {
		return uint32(ib[i]) | (uint32(ib[i+1]) << 8) | (uint32(ib[i+2]) << 16)
	}
	cmp := func(p uint32, q uint32, e uint32) uint32 {
		l := uint32(0)
		for e -= q; l < e; l++ {
			if ib[p+l] != ib[q+l] {
				e = 0
			}
		}
		return l
	}
	literals := func(r uint32) {
		n += 0x21 * (r / 0x20)
		r %= 0x20
		if r != 0 {
			n += r + 1
		}
	}
	match := func(l uint32) {
		l--
		n += 3 * (l / 262)
		if l%262 >= 6 {
			n += 3
		} else {
			n += 2
		}
	}
	hash := func(v uint32) uint32 {
		return ((2654435769 * v) >> 19) & 0x1fff
	}
	setNextHash := func(ip uint32) uint32 {
		ht[hash(u24(ip))] = ip
		return ip + 1
	}
	a := uint32(0)
	ipLimit := uint32(len(ib)) - 13
	if len(ib) < 13 {
		ipLimit = 0
	}
	for ip := a + 2; ip < ipLimit; {
		r := uint32(0)
		d := uint32(0)
		for {
			s := u24(ip)
			h := hash(s)
			r = ht[h]
			ht[h] = ip
			d = ip - r
			if ip >= ipLimit {
				break
			}
			ip++
			if d <= 0x1fff && s == u24(r) {
				break
			}
		}
		if ip >= ipLimit {
			break
		}
		ip--
		if ip > a {
			literals(ip - a)
		}
		l := cmp(r+3, ip+3, ipLimit+9)
		match(l)
		ip = setNextHash(setNextHash(ip + l))
		a = ip
	}
	literals(uint32(len(ib)) - a)
	return n
}
