package llama2

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"syscall"
	"unsafe"
)

type MmapCheckpoint struct {
	f    *os.File
	data []byte

	dim       int
	hiddenDim int
	nLayers   int
	nHeads    int
	nKvHeads  int
	vocabSize int
	seqLen    int

	tokenEmbeddingTable []float32
	rmsAttWeight        []float32
	wq                  []float32
	wk                  []float32
	wv                  []float32
	wo                  []float32
	rmsFfnWeight        []float32
	w1                  []float32
	w2                  []float32
	w3                  []float32
	rmsFinalWeight      []float32
	freqCisReal         []float32
	freqCisImag         []float32
	wcls                []float32

	sharedWeights bool
}

func NewMmapCheckpoint(pathname string) (*MmapCheckpoint, error) {
	f, err := os.Open(pathname)
	if err != nil {
		return nil, fmt.Errorf("loading checkpoint file: %w", err)
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()

	var config struct {
		Dim       int32
		HiddenDim int32
		NLayers   int32
		NHeads    int32
		NKvHeads  int32
		VocabSize int32
		SeqLen    int32
	}
	if err := binary.Read(f, binary.LittleEndian, &config); err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	checkpoint := &MmapCheckpoint{
		dim:       int(config.Dim),
		hiddenDim: int(config.HiddenDim),
		nLayers:   int(config.NLayers),
		nHeads:    int(config.NHeads),
		nKvHeads:  int(config.NKvHeads),
		vocabSize: int(config.VocabSize),
		seqLen:    int(config.SeqLen),
	}
	checkpoint.sharedWeights = checkpoint.vocabSize > 0
	if !checkpoint.sharedWeights {
		checkpoint.vocabSize = -checkpoint.vocabSize
	}

	offset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("finding weight offset: %w", err)
	}
	size, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		return nil, fmt.Errorf("finding file size: %w", err)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	floats := unsafe.Slice((*float32)(unsafe.Pointer(&data[offset])), len(data)-int(offset)/4)
	off := 0
	assign := func(size int) []float32 {
		field := floats[off : off+int(size)]
		off += int(size)
		return field
	}

	checkpoint.tokenEmbeddingTable = assign(checkpoint.vocabSize * checkpoint.dim)
	checkpoint.rmsAttWeight = assign(checkpoint.nLayers * checkpoint.dim)
	checkpoint.wq = assign(checkpoint.nLayers * checkpoint.dim * checkpoint.dim)
	checkpoint.wk = assign(checkpoint.nLayers * checkpoint.dim * checkpoint.dim)
	checkpoint.wv = assign(checkpoint.nLayers * checkpoint.dim * checkpoint.dim)
	checkpoint.wo = assign(checkpoint.nLayers * checkpoint.dim * checkpoint.dim)
	checkpoint.rmsFfnWeight = assign(checkpoint.nLayers * checkpoint.dim)
	checkpoint.w1 = assign(checkpoint.nLayers * checkpoint.hiddenDim * checkpoint.dim)
	checkpoint.w2 = assign(checkpoint.nLayers * checkpoint.dim * checkpoint.hiddenDim)
	checkpoint.w3 = assign(checkpoint.nLayers * checkpoint.hiddenDim * checkpoint.dim)
	checkpoint.rmsFinalWeight = assign(checkpoint.dim)
	checkpoint.freqCisReal = assign(checkpoint.seqLen * checkpoint.dim / checkpoint.nHeads / 2)
	checkpoint.freqCisImag = assign(checkpoint.seqLen * checkpoint.dim / checkpoint.nHeads / 2)

	if !checkpoint.sharedWeights {
		checkpoint.wcls = assign(checkpoint.vocabSize * checkpoint.dim)
	} else {
		checkpoint.wcls = checkpoint.tokenEmbeddingTable
	}

	checkpoint.f = f
	checkpoint.data = data

	return checkpoint, err
}

func (c *MmapCheckpoint) Error() error {
	return nil
}

func (c *MmapCheckpoint) Close() error {
	defer c.f.Close()

	if err := syscall.Munmap(c.data); err != nil {
		return err
	}
	return c.f.Close()
}

func (c *MmapCheckpoint) Dim() int {
	return c.dim
}

func (c *MmapCheckpoint) HiddenDim() int {
	return c.hiddenDim
}

func (c *MmapCheckpoint) NLayers() int {
	return c.nLayers
}

func (c *MmapCheckpoint) NHeads() int {
	return c.nHeads
}
func (c *MmapCheckpoint) NKvHeads() int {
	return c.nKvHeads
}
func (c *MmapCheckpoint) VocabSize() int {
	return c.vocabSize
}

func (c *MmapCheckpoint) SeqLen() int {
	return c.seqLen
}

func (c *MmapCheckpoint) TokenEmbeddingTable(token int) []float32 {
	return c.tokenEmbeddingTable[token*c.dim : (token+1)*c.dim]
}

func (c *MmapCheckpoint) RmsAttWeight(layer int) []float32 {
	return c.rmsAttWeight[layer*c.dim : (layer+1)*c.dim]
}

func (c *MmapCheckpoint) RmsFfnWeight(layer int) []float32 {
	return c.rmsFfnWeight[layer*c.dim : (layer+1)*c.dim]
}

func (c *MmapCheckpoint) Wq(layer int) []float32 {
	return c.wq[layer*c.dim*c.dim : (layer+1)*c.dim*c.dim]
}

func (c *MmapCheckpoint) Wk(layer int) []float32 {
	return c.wk[layer*c.dim*c.dim : (layer+1)*c.dim*c.dim]
}

func (c *MmapCheckpoint) Wv(layer int) []float32 {
	return c.wv[layer*c.dim*c.dim : (layer+1)*c.dim*c.dim]
}

func (c *MmapCheckpoint) Wo(layer int) []float32 {
	return c.wo[layer*c.dim*c.dim : (layer+1)*c.dim*c.dim]
}

func (c *MmapCheckpoint) W1(layer int) []float32 {
	return c.w1[layer*c.dim*c.hiddenDim : (layer+1)*c.dim*c.hiddenDim]
}

func (c *MmapCheckpoint) W2(layer int) []float32 {
	return c.w2[layer*c.dim*c.hiddenDim : (layer+1)*c.dim*c.hiddenDim]
}

func (c *MmapCheckpoint) W3(layer int) []float32 {
	return c.w3[layer*c.dim*c.hiddenDim : (layer+1)*c.dim*c.hiddenDim]
}

func (c *MmapCheckpoint) RmsFinalWeight() []float32 {
	return c.rmsFinalWeight
}

func (c *MmapCheckpoint) FreqCisReal(pos int) []float32 {
	return c.freqCisReal[pos*c.dim/c.nHeads/2 : (pos+1)*c.dim/c.nHeads/2]
}

func (c *MmapCheckpoint) FreqCisImag(pos int) []float32 {
	return c.freqCisImag[pos*c.dim/c.nHeads/2 : (pos+1)*c.dim/c.nHeads/2]
}

func (c *MmapCheckpoint) Wcls() []float32 {
	return c.wcls
}
