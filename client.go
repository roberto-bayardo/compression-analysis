package main

import (
	"bytes"
	"context"
	"math/big"

	"github.com/ethereum/go-ethereum/cmd/utils"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/node"
)

type Client interface {
	BlockByHash(ctx context.Context, hash common.Hash) (*types.Block, error)
	BlockByNumber(ctx context.Context, number *big.Int) (*types.Block, error)
	Close()
}

type LocalClient struct {
	n  *node.Node
	db ethdb.Database
}

func NewLocalClient(dataDir string) (Client, error) {
	nodeCfg := node.DefaultConfig
	nodeCfg.Name = "geth"
	nodeCfg.DataDir = dataDir
	n, err := node.New(&nodeCfg)
	if err != nil {
		return nil, err
	}
	handles := utils.MakeDatabaseHandles(0)
	db, err := n.OpenDatabaseWithFreezer("chaindata", 512, handles, "", "", true)
	if err != nil {
		return nil, err
	}
	return &LocalClient{
		n:  n,
		db: db,
	}, nil
}

func (c *LocalClient) Close() {
	_ = c.db.Close()
	_ = c.n.Close()
}

func (c *LocalClient) BlockByHash(ctx context.Context, hash common.Hash) (*types.Block, error) {
	number := rawdb.ReadHeaderNumber(c.db, hash)
	if number == nil {
		return nil, nil
	}
	return rawdb.ReadBlock(c.db, hash, *number), nil
}

func (c *LocalClient) BlockByNumber(ctx context.Context, number *big.Int) (*types.Block, error) {
	if number.Int64() < 0 {
		return c.BlockByHash(ctx, rawdb.ReadHeadBlockHash(c.db))
	}
	hash := rawdb.ReadCanonicalHash(c.db, number.Uint64())
	if bytes.Equal(hash.Bytes(), common.Hash{}.Bytes()) {
		return nil, nil
	}
	return rawdb.ReadBlock(c.db, hash, number.Uint64()), nil
}
