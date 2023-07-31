# llama2.go

<p align="center">
  <img src="web/cute-llama-to-go.png" width="300" height="300" alt="Cute Llama">
</p>

This is a Go port of [llama2.c](https://github.com/karpathy/llama2.c).

## Setup

1. Download a [model](https://github.com/karpathy/llama2.c#models):
1. Download [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin)
1. `go install github.com/saracen/llama2.go/cmd/llama2go@latest`
1. Do things:

   ```shell
   ./llama2go --help
   llama2go: <checkpoint>
     -cpuprofile string
            write cpu profile to file
     -prompt string
            prompt
     -steps int
            max number of steps to run for, 0: use seq_len (default 256)
     -temperature float
            temperature for sampling (default 0.9)

   ./llama2go -prompt "Cute llamas are" -steps 38 --temperature 0 stories110M.bin
   <s>
   Cute llamas are two friends who love to play together. They have a special game that they play every day. They pretend to be superheroes and save the world.
   achieved tok/s: 43.268528
   ```

## Performance

Tokens per second:

| system                 | model       | llama2.c   | llama2.go (no cgo) | llama2.go (cgo)
| ---------------------- | ----------- | -----------| ------------------ | ---------------
| M1 Max, 10-Core, 32 GB | stories15M  | 676.392573 | 246.885611         | 473.840849
| M1 Max, 10-Core, 32 GB | stories42M  | 267.295597 | 98.165245          | 151.396638
| M1 Max, 10-Core, 32 GB | stories110M | 100.671141 | 42.592345          | 69.804907
