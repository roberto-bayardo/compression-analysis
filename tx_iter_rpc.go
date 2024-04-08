package main

import (
	"context"
	"fmt"
	"log"
	"math/big"
	"strings"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethclient"
)

type TxIterRPC struct {
	client          Client
	parentBlockHash *common.Hash
	block           *types.Block
	nextTx          int
}

func NewTxIterRPC(rpcEndpoint string, startBlock *big.Int) *TxIterRPC {
	fmt.Println("Starting block: %v, RPC endpoint: %v\n", startBlock, rpcEndpoint)
	it := &TxIterRPC{}
	var err error
	if strings.HasPrefix(rpcEndpoint, "http://") || strings.HasPrefix(rpcEndpoint, "https://") {
		it.client, err = ethclient.Dial(rpcEndpoint)
	} else {
		it.client, err = NewLocalClient(rpcEndpoint)
	}
	if err != nil {
		log.Fatalln(err)
	}
	it.block, err = it.client.BlockByNumber(context.Background(), startBlock)
	if err != nil {
		log.Fatalln(err)
	}
	return it
}

func (it *TxIterRPC) Next() []byte {
	if it.block == nil {
		return nil
	}
	txs := it.block.Transactions()
	var err error
	if it.nextTx == len(txs) {
		it.block, err = it.client.BlockByHash(context.Background(), it.block.ParentHash())
		if err != nil {
			log.Fatalln(err)
		}
		it.nextTx = 0
		return it.Next()
	}

	tx := txs[it.nextTx]
	it.nextTx++
	if tx.Type() == types.DepositTxType {
		return it.Next()
	}
	b, err := tx.MarshalBinary()
	if err != nil {
		log.Fatalln(err)
	}
	return b
}

func (it *TxIterRPC) Close() {
	it.client.Close()
}
