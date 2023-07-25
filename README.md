# llama2.go

This is a Go port of [llama2.c](https://github.com/karpathy/llama2.c).

## Performance

| model           | llama2.c         | llama2.go
| --------------- | ---------------- | ---------
| stories15M.bin  | 498.05447  tok/s | 70.359456 tok/s
| stories42M.bin  | 191.473448 tok/s | 106.393798 tok/s
| stories110M.bin | 95.522388  tok/s | 43.138391 tok/s
