# llama2.go

This is a Go port of [llama2.c](https://github.com/karpathy/llama2.c).

## Performance

| system                 | model           | llama2.c         | llama2.go
| ---------------------- | --------------- | ---------------- | ----------------
| M1 Max, 10-Core, 32 GB | stories15M.bin  | 676.392573 tok/s | 230.144629 tok/s
| M1 Max, 10-Core, 32 GB | stories42M.bin  | 267.295597 tok/s | 94.539509  tok/s
| M1 Max, 10-Core, 32 GB | stories110M.bin | 100.671141 tok/s | 42.359789  tok/s
