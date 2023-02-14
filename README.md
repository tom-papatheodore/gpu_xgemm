## GPU XGEMM

Performs either SGEMM or DGEMM operation using hipBLAS and measures FLOP/s over an average of 10 iterations.

### Build

On Crusher or Frontier:
```text
$ source setup_environment.sh
$ make
```

### Usage

```text
$ ./gpu_xgemm --help
----------------------------------------------------------------
Usage: ./gpu_xgemm [OPTIONS]

--matrix_size=<value>, -m:       Size of matrices
                                 (default is 1024)

 --precision=<value>,   -p:       <value> can be single or double
                                 to select sgemm or dgemm
                                 (default is double)

 --help,                -h:       Show help
----------------------------------------------------------------
```
