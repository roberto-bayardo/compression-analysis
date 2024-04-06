package main

import (
	"bufio"
	"encoding/hex"
	"log"
	"os"
)

type TxIterFile struct {
	file    *os.File
	scanner *bufio.Scanner
}

func NewTxIterFile(filename string) *TxIterFile {
	it := &TxIterFile{}
	var err error
	it.file, err = os.Open(filename)
	if err != nil {
		log.Fatal("couldn't open file", err)
	}
	it.scanner = bufio.NewScanner(it.file)
	buf := make([]byte, 0, 64*1024)
	it.scanner.Buffer(buf, 1024*1024)
	return it
}

func (it *TxIterFile) Next() []byte {
	if !it.scanner.Scan() {
		log.Fatal("Couldn't scan next line in file", it.scanner.Err())
	}
	line := it.scanner.Text()
	b, err := hex.DecodeString(line)
	if err != nil {
		log.Fatal("Couldn't decode line from file", err)
	}
	return b
}

func (it *TxIterFile) Close() {
	it.file.Close()
}
