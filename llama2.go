package llama2

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sync"
	"syscall"
	"unsafe"
)

type Checkpoint struct {
	f    *os.File
	data []byte

	Config             *Config
	TransformerWeights *TransformerWeights
}

type Config struct {
	Dim       int32
	HiddenDim int32
	NLayers   int32
	NHeads    int32
	NKvHeads  int32
	VocabSize int32
	SeqLen    int32
}

type TransformerWeights struct {
	TokenEmbeddingTable []float32
	RmsAttWeight        []float32
	RmsFfnWeight        []float32
	Wq                  []float32
	Wk                  []float32
	Wv                  []float32
	Wo                  []float32
	W1                  []float32
	W2                  []float32
	W3                  []float32
	RmsFinalWeight      []float32
	FreqCisReal         []float32
	FreqCisImag         []float32
	Wcls                []float32
}

type RunState struct {
	X          []float32
	Xb         []float32
	Xb2        []float32
	Hb         []float32
	Hb2        []float32
	Q          []float32
	K          []float32
	V          []float32
	Att        []float32
	Logits     []float32
	KeyCache   []float32
	ValueCache []float32
}

func LoadCheckpoint(pathname string) (c *Checkpoint, err error) {
	f, err := os.Open(pathname)
	if err != nil {
		return nil, fmt.Errorf("loading checkpoint file: %w", err)
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()

	config := &Config{}
	if err := binary.Read(f, binary.LittleEndian, config); err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	sharedWeights := config.VocabSize > 0
	if !sharedWeights {
		config.VocabSize = -config.VocabSize
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
	assign := func(size int32) []float32 {
		field := floats[off : off+int(size)]
		off += int(size)
		return field
	}

	weights := &TransformerWeights{
		TokenEmbeddingTable: assign(config.VocabSize * config.Dim),
		RmsAttWeight:        assign(config.NLayers * config.Dim),
		Wq:                  assign(config.NLayers * config.Dim * config.Dim),
		Wk:                  assign(config.NLayers * config.Dim * config.Dim),
		Wv:                  assign(config.NLayers * config.Dim * config.Dim),
		Wo:                  assign(config.NLayers * config.Dim * config.Dim),
		RmsFfnWeight:        assign(config.NLayers * config.Dim),
		W1:                  assign(config.NLayers * config.HiddenDim * config.Dim),
		W2:                  assign(config.NLayers * config.Dim * config.HiddenDim),
		W3:                  assign(config.NLayers * config.HiddenDim * config.Dim),
		RmsFinalWeight:      assign(config.Dim),
		FreqCisReal:         assign(config.SeqLen * config.Dim / config.NHeads / 2),
		FreqCisImag:         assign(config.SeqLen * config.Dim / config.NHeads / 2),
	}

	if sharedWeights {
		weights.Wcls = weights.TokenEmbeddingTable
	} else {
		weights.Wcls = assign(config.VocabSize * config.Dim)
	}

	return &Checkpoint{
		f:                  f,
		data:               data,
		Config:             config,
		TransformerWeights: weights,
	}, nil
}

func (c *Checkpoint) Close() {
	syscall.Munmap(c.data)
	c.f.Close()
}

func LoadTokenizer(pathname string, size int) ([]string, error) {
	vocab := make([]string, size)

	f, err := os.Open(pathname)
	if err != nil {
		return nil, fmt.Errorf("loading tokenizer file: %w", err)
	}
	defer f.Close()

	r := bufio.NewReader(f)

	for i := 0; i < size; i++ {
		var len int32
		if err := binary.Read(r, binary.LittleEndian, &len); err != nil {
			return nil, fmt.Errorf("reading length: %w", err)
		}

		data := make([]byte, len)
		if _, err := io.ReadFull(r, data); err != nil {
			return nil, fmt.Errorf("reading data: %w", err)
		}
		vocab[i] = string(data)
	}

	return vocab, nil
}

func NewRunState(config *Config) *RunState {
	return &RunState{
		X:          make([]float32, config.Dim),
		Xb:         make([]float32, config.Dim),
		Xb2:        make([]float32, config.Dim),
		Hb:         make([]float32, config.HiddenDim),
		Hb2:        make([]float32, config.HiddenDim),
		Q:          make([]float32, config.Dim),
		K:          make([]float32, config.Dim),
		V:          make([]float32, config.Dim),
		Att:        make([]float32, config.NHeads*config.SeqLen),
		Logits:     make([]float32, config.VocabSize),
		KeyCache:   make([]float32, config.NLayers*config.SeqLen*config.Dim),
		ValueCache: make([]float32, config.NLayers*config.SeqLen*config.Dim),
	}
}

func accum(a, b []float32) {
	_ = a[len(a)-1]
	_ = b[len(a)-1]
	for i := range a {
		a[i] += b[i]
	}
}

func rmsnorm(o, x, weight []float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	ss /= float32(len(x))
	ss += 1e-5
	ss = 1.0 / float32(math.Sqrt(float64(ss)))

	_ = weight[len(o)-1]
	_ = x[len(o)-1]
	for j := range o {
		o[j] = weight[j] * ss * x[j]
	}
}

func Softmax(x []float32) {
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - maxVal)))
		sum += x[i]
	}

	for i := range x {
		x[i] /= sum
	}
}

func matmul(xout, x, w []float32, n, d int) {
	_ = xout[d-1]

	for i := 0; i < d; i++ {
		var val float32
		in := i * n

		for j := 0; j < n-4; j += 4 {
			w := w[in+j : in+j+4]
			x := x[j : j+4]

			val += w[0] * x[0]
			val += w[1] * x[1]
			val += w[2] * x[2]
			val += w[3] * x[3]
		}
		xout[i] = val
	}
}

func Transformer(token, pos int, p *Config, s *RunState, w *TransformerWeights) {
	var wg sync.WaitGroup

	// a few convenience variables
	x := s.X
	dim := int(p.Dim)
	hiddenDim := int(p.HiddenDim)
	headSize := dim / int(p.NHeads)

	// copy the token embedding into x
	copy(x, w.TokenEmbeddingTable[token*dim:])

	// pluck out the "pos" row of freq_cis_real and freq_cis_imag
	freqCisRealRow := w.FreqCisReal[pos*headSize/2:]
	freqCisImagRow := w.FreqCisImag[pos*headSize/2:]

	// forward all the layers
	for l := 0; l < int(p.NLayers); l++ {
		// attention rmsnorm
		rmsnorm(s.Xb, x, w.RmsAttWeight[l*dim:])

		// qkv matmuls for this position
		wg.Add(3)
		go func() { matmul(s.Q, s.Xb, w.Wq[l*dim*dim:], dim, dim); wg.Done() }()
		go func() { matmul(s.K, s.Xb, w.Wk[l*dim*dim:], dim, dim); wg.Done() }()
		go func() { matmul(s.V, s.Xb, w.Wv[l*dim*dim:], dim, dim); wg.Done() }()
		wg.Wait()

		// apply RoPE rotation to the q and k vectors for each head
		for h := 0; h < int(p.NHeads); h++ {
			// get the q and k vectors for this head
			q := s.Q[h*headSize:]
			k := s.K[h*headSize:]
			// rotate q and k by the freq_cis_real and freq_cis_imag
			for i := 0; i < headSize; i += 2 {
				q0 := q[i]
				q1 := q[i+1]
				k0 := k[i]
				k1 := k[i+1]
				fcr := freqCisRealRow[i/2]
				fci := freqCisImagRow[i/2]
				q[i] = q0*fcr - q1*fci
				q[i+1] = q0*fci + q1*fcr
				k[i] = k0*fcr - k1*fci
				k[i+1] = k0*fci + k1*fcr
			}
		}

		// save key,value at this time step (pos) to our kv cache
		loff := l * int(p.SeqLen) * dim // kv cache layer offset for convenience
		copy(s.KeyCache[loff+pos*dim:], s.K[:dim])
		copy(s.ValueCache[loff+pos*dim:], s.V[:dim])

		// multihead attention. iterate over all heads
		wg.Add(int(p.NHeads))
		for h := 0; h < int(p.NHeads); h++ {
			h := h
			go func() {
				// get the query vector for this head
				q := s.Q[h*headSize:]
				// attention scores for this head
				att := s.Att[h*int(p.SeqLen):]
				// iterate over all timesteps, including the current one
				for t := 0; t <= pos; t++ {
					// get the key vector for this head and at this timestep
					k := s.KeyCache[loff+t*dim+h*headSize:]
					// calculate the attention score as the dot product of q and k
					var score float32
					for i := 0; i < headSize; i++ {
						score += q[i] * k[i]
					}
					score /= float32(math.Sqrt(float64(headSize)))
					// save the score to the attention buffer
					att[t] = score
				}

				// softmax the scores to get attention weights, from 0..pos inclusively
				Softmax(att[:pos+1])

				// weighted sum of the values, store back into xb
				for i := 0; i < headSize; i++ {
					val := 0.0
					for t := 0; t <= pos; t++ {
						val += float64(att[t] * s.ValueCache[loff+t*dim+h*headSize+i]) // note bad locality
					}
					s.Xb[h*headSize+i] = float32(val)
				}
				wg.Done()
			}()
		}
		wg.Wait()

		// final matmul to get the output of the attention
		matmul(s.Xb2, s.Xb, w.Wo[l*dim*dim:], dim, dim)

		// residual connection back into x
		accum(x, s.Xb2)

		// ffn rmsnorm
		rmsnorm(s.Xb, x, w.RmsFfnWeight[l*dim:])

		wg.Add(2)
		go func() { matmul(s.Hb, s.Xb, w.W1[l*dim*hiddenDim:], dim, hiddenDim); wg.Done() }()
		go func() { matmul(s.Hb2, s.Xb, w.W3[l*dim*hiddenDim:], dim, hiddenDim); wg.Done() }()
		wg.Wait()

		// F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
		for i := 0; i < hiddenDim; i++ {
			s.Hb[i] = s.Hb[i] * (1.0 / (1.0 + float32(math.Exp(-float64(s.Hb[i])))))
		}

		// elementwise multiply with w3(x)
		for i := 0; i < hiddenDim; i++ {
			s.Hb[i] = s.Hb[i] * s.Hb2[i]
		}

		// final matmul to get the output of the ffn
		matmul(s.Xb, s.Hb, w.W2[l*dim*hiddenDim:], hiddenDim, dim)

		// residual connection
		accum(x, s.Xb)
	}

	// final rmsnorm
	rmsnorm(x, x, w.RmsFinalWeight)

	// classifier into logits
	matmul(s.Logits, x, w.Wcls, int(p.Dim), int(p.VocabSize))
}

func Sample(rnd rand.Source, probabilities []float32) int {
	r := float32(rnd.Int63()) / (1 << 63)
	var cdf float32
	for i, p := range probabilities {
		cdf += p
		if r < cdf {
			return i
		}
	}
	return len(probabilities) - 1
}

func Argmax(v []float32) int {
	maxI := 0
	maxP := v[0]
	for i, p := range v[1:] {
		if p > maxP {
			maxI = i + 1
			maxP = p
		}
	}
	return maxI
}
