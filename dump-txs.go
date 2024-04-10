package main

import (
	"bufio"
	"fmt"
	"log"
	"math/big"
	"os"
)

func dumpTxs() {
	//clientLocation := "/nvme/op-geth/snapdata-path/geth.ipc"
	clientLocation := "/nvme/op-geth/snapdata-path/"
	blockNum := big.NewInt(10000000) // starting block; will iterate backwards from here
	txIter := NewTxIterRPC(clientLocation, blockNum)

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
		fmt.Fprintf(w, "%x\n", b)
		count++
		if count%100000 == 0 {
			fmt.Println("tx count:", count, "on block:", txIter.BlockNumber())
		}
	}
}
