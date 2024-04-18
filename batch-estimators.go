package main

import (
	"bytes"
	"compress/zlib"
	"io"
	"log"

	"github.com/google/brotli/go/cbrotli"
)

type batchWriter interface {
	io.Writer
	Flush() error
	Reset(w io.Writer)
}

// zlibBestBatchEstimator simulates a zlib compressor at max compression that works on (large) tx
// batches. This function is not thread safe.
func zlibBestBatchEstimator(tx []byte) float64 {
	return zlibBatchEstimatorObj.write(tx)
}

// brotliBatchEstimator simulates a brotli compressor at compression lvl 10 that works on (large)
// tx batches. This function is not thread safe.
func brotliBatchEstimator(tx []byte) float64 {
	return brotliBatchEstimatorObj.write(tx)
}

var zlibBatchEstimatorObj = newZlibBatchEstimatorObj()

type batchEstimator struct {
	b [2]*bytes.Buffer
	w [2]batchWriter
	// bootstrapped is set to true once this estimator has processed enough transactions to be
	// considered reliable in its output
	bootstrapped bool
}

func newZlibBatchEstimatorObj() *batchEstimator {
	b := &batchEstimator{}
	for i := range b.w {
		b.b[i] = new(bytes.Buffer)
		w, err := zlib.NewWriterLevel(b.b[i], zlib.BestCompression)
		if err != nil {
			log.Fatalln(err)
		}
		b.w[i] = w
	}
	return b
}

var brotliBatchEstimatorObj = newBrotliBatchEstimatorObj()

type brotliBatchWriter struct {
	w *cbrotli.Writer
}

func (b *brotliBatchWriter) Write(by []byte) (n int, err error) {
	return b.w.Write(by)
}

func (b *brotliBatchWriter) Flush() error {
	return b.w.Flush()
}

func (b *brotliBatchWriter) Reset(w io.Writer) {
	if b.w != nil {
		b.w.Close()
	}
	b.w = cbrotli.NewWriter(w, cbrotli.WriterOptions{Quality: 10})
}

func newBrotliBatchEstimatorObj() *batchEstimator {
	b := &batchEstimator{}
	for i := range b.w {
		b.b[i] = new(bytes.Buffer)
		w := cbrotli.NewWriter(b.b[i], cbrotli.WriterOptions{Quality: 10})
		b.w[i] = &brotliBatchWriter{w}
	}
	return b
}

func (w *batchEstimator) write(p []byte) float64 {
	if spanBatchMode {
		// span batch mode segregates the tx signatures, which we simulate by stripping them out
		// before compression and treating them as 65 uncompressible bytes.
		p = p[:len(p)-65]
	}
	// targeting:
	//	b[0] == 0-64kb
	//	b[1] == 64kb-128kb
	before := w.b[1].Len()
	_, err := w.w[1].Write(p)
	if err != nil {
		log.Fatalln(err)
	}
	err = w.w[1].Flush()
	if err != nil {
		log.Fatalln(err)
	}
	after := w.b[1].Len()
	// if b[1] > 64kb, write to b[0]
	if w.b[1].Len() > numBlobs*64*1024 {
		_, err = w.w[0].Write(p)
		if err != nil {
			log.Fatalln(err)
		}
		err = w.w[0].Flush()
		if err != nil {
			log.Fatalln(err)
		}
		w.bootstrapped = true
	}
	// if b[1] > 128kb, rotate and clear shadow buffer b[0]
	if w.b[1].Len() > numBlobs*128*1024 {
		tb := w.b[1]
		tw := w.w[1]
		w.b[1] = w.b[0]
		w.w[1] = w.w[0]
		w.b[0] = tb
		w.w[0] = tw
		w.b[0].Reset()
		w.w[0].Reset(w.b[0])
	}
	r := float64(after - before - 2) // flush writes 2 extra "sync" bytes so don't count those
	if spanBatchMode {
		return r + 65.
	}
	return r
}
