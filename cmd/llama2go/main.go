package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/saracen/llama2.go"
)

func main() {
	// parse flags
	var (
		temperatureArg = flag.Float64("temperature", 0.9, "temperature for sampling")
		stepsArg       = flag.Int64("steps", 256, "max number of steps to run for, 0: use seq_len")
	)
	{
		flag.Usage = func() {
			flag.PrintDefaults()
		}
		flag.Parse()
		if flag.Arg(0) == "" {
			flag.Usage()
		}
	}

	// read model.bin
	checkpoint, err := llama2.LoadCheckpoint(flag.Arg(0))
	if err != nil {
		fmt.Println("loading checkpoint:", err)
		os.Exit(1)
	}
	defer checkpoint.Close()

	// read tokenizer.bin
	vocab, err := llama2.LoadTokenizer("tokenizer.bin", int(checkpoint.Config.VocabSize))
	if err != nil {
		fmt.Println("Failed to load tokenizer.bin:", err)
		os.Exit(1)
	}

	var (
		next        = 0
		token       = 1
		temperature = float32(*temperatureArg)
		steps       = int(*stepsArg)
		start       = time.Now()
		state       = llama2.NewRunState(checkpoint.Config)
		seed        = uint64(time.Now().Unix())
	)
	if steps <= 0 || steps > int(checkpoint.Config.SeqLen) {
		steps = int(checkpoint.Config.SeqLen)
	}

	fmt.Println("<s>") // explicit print the initial BOS token (=1), stylistically symmetric
	for pos := 0; pos < steps; pos++ {
		// forward the transformer to get logits for the next token
		llama2.Transformer(token, pos, checkpoint.Config, state, checkpoint.TransformerWeights)

		// sample the next token
		if temperature == 0 {
			// greedy argmax sampling
			next = llama2.Argmax(state.Logits)
		} else {
			// apply the temperature to the logits
			for q := 0; q < int(checkpoint.Config.VocabSize); q++ {
				state.Logits[q] /= temperature
			}
			// apply softmax to the logits to get the probabilities for next token
			llama2.Softmax(state.Logits)
			// we now want to sample from this distribution to get the next token
			next = llama2.Sample(seed, state.Logits)
		}
		fmt.Print(vocab[next])

		// advance forward
		token = next
	}
	fmt.Println()

	// report achieved tok/s
	fmt.Printf("achieved tok/s: %f\n", float64(checkpoint.Config.SeqLen)/time.Since(start).Seconds())
}
