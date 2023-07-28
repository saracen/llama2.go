# llama2.go

This is a Go port of [llama2.c](https://github.com/karpathy/llama2.c).

## Performance

| system                                    | model           | llama2.c         | llama2.go
| ----------------------------------------- | --------------- | ---------------- |
| MacBook Pro, Apple M1 Max, 10-Core, 32 GB | stories15M.bin  | 676.392573 tok/s | 73.702954 tok/s
| MacBook Pro, Apple M1 Max, 10-Core, 32 GB | stories42M.bin  | 267.295597 tok/s | 27.107370 tok/s
| MacBook Pro, Apple M1 Max, 10-Core, 32 GB | stories110M.bin | 100.671141 tok/s | 10.857624 tok/s
