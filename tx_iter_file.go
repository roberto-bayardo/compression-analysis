package main

import (
	"bufio"
	"encoding/hex"
	"fmt"
	"log"
	"os"
)

type TxIterFile struct {
	file    *os.File
	scanner *bufio.Scanner
}

func NewTxIterFile(filename string) *TxIterFile {
	fmt.Printf("Transaction file name: %v\n", filename)
	it := &TxIterFile{}
	var err error
	it.file, err = os.Open(filename)
	if err != nil {
		log.Fatalln("couldn't open file", err)
	}
	it.scanner = bufio.NewScanner(it.file)
	buf := make([]byte, 0, 64*1024)
	it.scanner.Buffer(buf, 1024*1024)
	return it
}

func (it *TxIterFile) Next() []byte {
	if !it.scanner.Scan() {
		if it.scanner.Err() == nil {
			return nil
		}
		log.Fatalln("Couldn't scan next line in file:", it.scanner.Err())
	}
	line := it.scanner.Text()
	b, err := hex.DecodeString(line)
	if err != nil {
		log.Fatalln("Couldn't decode line from file:", err)
	}
	return b
}

func (it *TxIterFile) Close() {
	it.file.Close()
}
