package main

import (
	"bufio"
	"fmt"
	"log"
	"math/big"
	"math/rand"
	"os"
)

// dumpTxs dumps transactions from a geth database or instance to a file, one hexencoded tx per line.
func dumpTxs() {
	//clientLocation := "/nvme/op-geth/snapdata-path/geth.ipc"
	clientLocation := "/nvme/op-geth/snapdata-path/"
	startBlock := big.NewInt(13000000) // head of base as of Apr 10 2024
	stopBlock := big.NewInt(11792527)  // first ecotone block

	txIter := NewTxIterRPC(clientLocation, startBlock)

	f, err := os.Create("./dump")
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	count := 0
	for b := txIter.Next(); ; b = txIter.Next() {
		if b == nil {
			log.Println("ran out of transactions, exiting loop")
			break
		}
		if txIter.BlockNumber().Cmp(stopBlock) <= 0 {
			fmt.Println("reached stopping block, exiting. txs:", count)
			return
		}
		fmt.Fprintf(w, "%x\n", b)
		count++
		if count%100000 == 0 {
			fmt.Println("tx count:", count, "on block:", txIter.BlockNumber())
		}
	}
}

// dumpTxsAsSamples dumps transactions from geth to individual files that can be passed to brotli
// or zstd dictionary generators. p is the sampling probability.
func dumpTxsAsSamples(num int, p float64) {
	//clientLocation := "/nvme/op-geth/snapdata-path/geth.ipc"
	//clientLocation := "/nvme/op-geth/snapdata-path/"
	clientLocation := "http://optimism-mainnet.rosetta-api-v2.us-east-1.development.cbhq.net:8545"
	//startBlock := big.NewInt(13043496)
	//stopBlock := big.NewInt(11792527) // first base ecotone block
	stopBlock := big.NewInt(117939221) // shortly after ecotone optimism hardfork
	startBlock := big.NewInt(118939254)

	txIter := NewTxIterRPC(clientLocation, startBlock)

	count := 0
	r := rand.New(rand.NewSource(99))

	for b := txIter.Next(); ; b = txIter.Next() {
		if b == nil {
			log.Println("ran out of transactions, exiting loop")
			break
		}
		if txIter.BlockNumber().Cmp(stopBlock) <= 0 {
			fmt.Println("reached stopping block, exiting. txs:", count)
			return
		}
		count++
		if count%100000 == 0 {
			fmt.Println("tx count:", count, "on block:", txIter.BlockNumber())
		}

		if r.Float64() > p {
			continue
		}

		f, err := os.Create(fmt.Sprintf("./sample-%d", num))
		if err != nil {
			log.Fatalln(err)
		}
		num--
		w := bufio.NewWriter(f)
		if _, err := w.Write(b); err != nil {
			log.Fatalln(err)
		}
		w.Flush()
		f.Close()
		if num == 0 {
			fmt.Println("generated requested # of samples, exiting.")
			return
		}
	}
}
