package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"time"

	"github.com/saracen/llama2.go"
)

func main() {
	// parse flags
	var (
		temperatureArg = flag.Float64("temperature", 0.9, "temperature for sampling")
		stepsArg       = flag.Int64("steps", 256, "max number of steps to run for, 0: use seq_len")
		promptArg      = flag.String("prompt", "", "prompt")

		cpuprofileArg = flag.String("cpuprofile", "", "write cpu profile to file")
	)
	{
		flag.Usage = func() {
			fmt.Fprintf(flag.CommandLine.Output(), "llama2go: <checkpoint>\n")
			flag.PrintDefaults()
		}
		flag.Parse()
		if flag.Arg(0) == "" {
			flag.Usage()
		}
	}

	if *cpuprofileArg != "" {
		f, err := os.Create(*cpuprofileArg)
		if err != nil {
			exit("Creating CPU Profile:", err)
		}
		defer f.Close() // error handling omitted for example
		if err := pprof.StartCPUProfile(f); err != nil {
			exit("Starting CPU Profile:", err)
		}
		defer pprof.StopCPUProfile()
	}

	// read model.bin
	checkpoint, err := llama2.LoadCheckpoint(flag.Arg(0))
	if err != nil {
		exit("Loading checkpoint:", err)
	}
	defer checkpoint.Close()

	// read tokenizer.bin
	vocab, err := llama2.LoadTokenizer("tokenizer.bin", int(checkpoint.Config.VocabSize))
	if err != nil {
		exit("Failed to load tokenizer.bin:", err)
	}

	var (
		next         = 0
		token        = 1
		temperature  = float32(*temperatureArg)
		steps        = int(*stepsArg)
		state        = llama2.NewRunState(checkpoint.Config)
		seed         = uint64(time.Now().Unix())
		start        time.Time
		promptTokens []int
	)
	if steps <= 0 || steps > int(checkpoint.Config.SeqLen) {
		steps = int(checkpoint.Config.SeqLen)
	}
	if *promptArg != "" {
		promptTokens, err = vocab.BPEEncode(*promptArg)
		if err != nil {
			exit("Encoding prompt:", err)
		}
	}

	fmt.Println("<s>") // explicit print the initial BOS token (=1), stylistically symmetric
	for pos := 0; pos < steps; pos++ {
		// forward the transformer to get logits for the next token
		llama2.Transformer(token, pos, checkpoint.Config, state, checkpoint.TransformerWeights)

		if len(promptTokens) > 0 {
			next = promptTokens[0]
			promptTokens = promptTokens[1:]
		} else {
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
		}

		tokenString := vocab.String(next)
		if token == 1 && tokenString[0] == ' ' {
			fmt.Print(tokenString[1:])
		} else {
			fmt.Print(tokenString)
		}

		// advance forward
		token = next
		if pos == 0 {
			start = time.Now()
		}
	}
	fmt.Println()

	// report achieved tok/s
	fmt.Printf("achieved tok/s: %f\n", float64(steps-1)/time.Since(start).Seconds())
}

func exit(msg string, err error) {
	fmt.Println(msg, err)
	os.Exit(1)
}
