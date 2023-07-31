package llama2

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sync"
)

type Config struct {
	Dim       int
	HiddenDim int
	NLayers   int
	NHeads    int
	NKvHeads  int
	VocabSize int
	SeqLen    int
}

type Checkpoint interface {
	Error() error
	Close() error

	Dim() int
	HiddenDim() int
	NLayers() int
	NHeads() int
	NKvHeads() int
	VocabSize() int
	SeqLen() int

	TokenEmbeddingTable(token int) []float32
	RmsAttWeight(layer int) []float32
	RmsFfnWeight(layer int) []float32
	Wq(layer int) []float32
	Wk(layer int) []float32
	Wv(layer int) []float32
	Wo(layer int) []float32
	W1(layer int) []float32
	W2(layer int) []float32
	W3(layer int) []float32
	RmsFinalWeight() []float32
	FreqCisReal(pos int) []float32
	FreqCisImag(pos int) []float32
	Wcls() []float32
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

type Vocab struct {
	strings []string
	scores  []float32

	maxTokenLength int32
}

func (v Vocab) id(s string) int {
	for idx, v := range v.strings {
		if v == s {
			return idx
		}
	}

	return -1
}

func (v Vocab) String(id int) string {
	return v.strings[id]
}

func (v Vocab) BPEEncode(text string) ([]int, error) {
	tokens := make([]int, 0, v.maxTokenLength)

	// first encode every individual byte in the input string
	for _, b := range []byte(text) {
		id := v.id(string(b))
		if id == -1 {
			return nil, fmt.Errorf("not good")
		}

		tokens = append(tokens, id)
	}

	// merge the best consecutive pair each iteration, according the scores in vocab_scores
	for {
		bestScore := float32(-1e10)
		bestId := -1
		bestIdx := -1

		for i := 0; i < len(tokens)-1; i++ {
			// check if we can merge the pair (tokens[i], tokens[i+1])
			id := v.id(v.String(tokens[i]) + v.String(tokens[i+1]))
			if id == -1 {
				continue
			}

			if v.scores[id] > bestScore {
				bestScore = v.scores[id]
				bestId = id
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break // we couldn't find any more pairs to merge, so we're done
		}

		// merge the consecutive pair (best_idx, best_idx+1) into new token best_id
		tokens[bestIdx] = bestId

		// delete token at position best_idx+1, shift the entire sequence back 1
		for i := bestIdx + 1; i < len(tokens)-1; i++ {
			tokens[i] = tokens[i+1]
		}
		tokens = tokens[:len(tokens)-1]
	}

	return tokens, nil
}

func LoadTokenizer(pathname string, size int) (Vocab, error) {
	vocab := Vocab{
		strings: make([]string, size),
		scores:  make([]float32, size),
	}

	f, err := os.Open(pathname)
	if err != nil {
		return vocab, fmt.Errorf("loading tokenizer file: %w", err)
	}
	defer f.Close()

	r := bufio.NewReader(f)

	if err := binary.Read(r, binary.LittleEndian, &vocab.maxTokenLength); err != nil {
		return vocab, fmt.Errorf("reading max token length: %w", err)
	}

	for i := 0; i < size; i++ {
		if err := binary.Read(r, binary.LittleEndian, &vocab.scores[i]); err != nil {
			return vocab, fmt.Errorf("reading vocab scores: %w", err)
		}

		var len int32
		if err := binary.Read(r, binary.LittleEndian, &len); err != nil {
			return vocab, fmt.Errorf("reading length: %w", err)
		}

		data := make([]byte, len)
		if _, err := io.ReadFull(r, data); err != nil {
			return vocab, fmt.Errorf("reading data: %w", err)
		}
		vocab.strings[i] = string(data)
	}

	return vocab, nil
}

func NewRunState(c Checkpoint) *RunState {
	return &RunState{
		X:          make([]float32, c.Dim()),
		Xb:         make([]float32, c.Dim()),
		Xb2:        make([]float32, c.Dim()),
		Hb:         make([]float32, c.HiddenDim()),
		Hb2:        make([]float32, c.HiddenDim()),
		Q:          make([]float32, c.Dim()),
		K:          make([]float32, c.Dim()),
		V:          make([]float32, c.Dim()),
		Att:        make([]float32, c.NHeads()*c.SeqLen()),
		Logits:     make([]float32, c.VocabSize()),
		KeyCache:   make([]float32, c.NLayers()*c.SeqLen()*c.Dim()),
		ValueCache: make([]float32, c.NLayers()*c.SeqLen()*c.Dim()),
	}
}

func Transformer(token, pos int, c Checkpoint, s *RunState) {
	var wg sync.WaitGroup

	// copy the token embedding into x
	copy(s.X, c.TokenEmbeddingTable(token))

	// pluck out the "pos" row of freq_cis_real and freq_cis_imag
	freqCisRealRow := c.FreqCisReal(pos)
	freqCisImagRow := c.FreqCisImag(pos)

	// forward all the layers
	headSize := c.Dim() / c.NHeads()

	if len(freqCisRealRow) < headSize/2 {
		return
	}
	if len(freqCisImagRow) < headSize/2 {
		return
	}

	for l := 0; l < c.NLayers(); l++ {
		// attention rmsnorm
		rmsnorm(s.Xb, s.X, c.RmsAttWeight(l))

		// qkv matmuls for this position
		wg.Add(3)
		go func() { matmul(s.Q, s.Xb, c.Wq(l)); wg.Done() }()
		go func() { matmul(s.K, s.Xb, c.Wk(l)); wg.Done() }()
		go func() { matmul(s.V, s.Xb, c.Wv(l)); wg.Done() }()
		wg.Wait()

		// apply RoPE rotation to the q and k vectors for each head
		for h := 0; h < int(c.NHeads()); h++ {
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
		loff := l * c.SeqLen() * c.Dim() // kv cache layer offset for convenience
		copy(s.KeyCache[loff+pos*c.Dim():], s.K[:c.Dim()])
		copy(s.ValueCache[loff+pos*c.Dim():], s.V[:c.Dim()])

		// multihead attention. iterate over all heads
		wg.Add(c.NHeads())
		for h := 0; h < c.NHeads(); h++ {
			h := h
			go func() {
				hhs := h * headSize
				// get the query vector for this head
				q := s.Q[hhs:]
				// attention scores for this head
				att := s.Att[h*c.SeqLen():]
				// iterate over all timesteps, including the current one
				for t := 0; t <= pos; t++ {
					// get the key vector for this head and at this timestep
					k := s.KeyCache[loff+t*c.Dim()+hhs:]
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
				xb := s.Xb[hhs : hhs+headSize]
				for i := range xb {
					xb[i] = 0.0
				}
				for t := 0; t <= pos; t++ {
					v := s.ValueCache[loff+t*c.Dim()+hhs : loff+t*c.Dim()+hhs+headSize]
					a := att[t]
					for i := range v {
						xb[i] += a * v[i]
					}
				}
				wg.Done()
			}()
		}
		wg.Wait()

		// final matmul to get the output of the attention
		matmul(s.Xb2, s.Xb, c.Wo(l))

		// residual connection back into x
		accum(s.X, s.Xb2)

		// ffn rmsnorm
		rmsnorm(s.Xb, s.X, c.RmsFfnWeight(l))

		wg.Add(2)
		go func() { matmul(s.Hb, s.Xb, c.W1(l)); wg.Done() }()
		go func() { matmul(s.Hb2, s.Xb, c.W3(l)); wg.Done() }()
		wg.Wait()

		// F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
		for i := 0; i < c.HiddenDim(); i++ {
			s.Hb[i] = s.Hb[i] * (1.0 / (1.0 + float32(math.Exp(-float64(s.Hb[i])))))
		}

		// elementwise multiply with w3(x)
		for i := 0; i < c.HiddenDim(); i++ {
			s.Hb[i] = s.Hb[i] * s.Hb2[i]
		}

		// final matmul to get the output of the ffn
		matmul(s.Xb, s.Hb, c.W2(l))

		// residual connection
		accum(s.X, s.Xb)
	}

	// final rmsnorm
	rmsnorm(s.X, s.X, c.RmsFinalWeight())

	// classifier into logits
	matmul(s.Logits, s.X, c.Wcls())
}

func Sample(seed uint64, probabilities []float32) int {
	r := randomF32(seed)
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

func Softmax(x []float32) {
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sum float32
	var i int
	for ; i < len(x)-4; i += 4 {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		x[i+1] = float32(math.Exp(float64(x[i+1] - maxVal)))
		x[i+2] = float32(math.Exp(float64(x[i+2] - maxVal)))
		x[i+3] = float32(math.Exp(float64(x[i+3] - maxVal)))

		sum += x[i]
		sum += x[i+1]
		sum += x[i+2]
		sum += x[i+3]
	}

	for ; i < len(x); i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}

	for i := range x {
		x[i] /= sum
	}
}

func accum(a, b []float32) {
	if len(a) != len(b) {
		return
	}

	var i int
	for ; i < len(a)-4; i += 4 {
		a[i] += b[i]
		a[i+1] += b[i+1]
		a[i+2] += b[i+2]
		a[i+3] += b[i+3]
	}

	for ; i < len(a); i++ {
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

	if len(x) != len(o) || len(weight) != len(o) {
		return
	}
	for j := range o {
		o[j] = weight[j] * ss * x[j]
	}
}

func randomU32(seed uint64) uint32 {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	seed ^= seed >> 12
	seed ^= seed << 25
	seed ^= seed >> 27
	return uint32((seed * 0x2545F4914F6CDD1D) >> 32)
}

func randomF32(seed uint64) float32 { // random float32 in [0,1)
	return float32(randomU32(seed)>>8) / 16777216.0
}
