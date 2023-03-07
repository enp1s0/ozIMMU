# ozIMMA - DGEMM on Integer Tensor Core with Ozaki scheme

This library hijacks the function calls for cuBLAS DGEMM functions and execute ozIMMA instead of them

## Build
```bash
git clone https://github.com/enp1s0/ozIMMA --recursive
cd ozIMMA
mkdir build
cd build
cmake ..
make -j4
```

## Usage

1. Set an environmental variable to hijack the function calls
```bash
export LD_PRELOAD=/path/to/ozIMMA/build/libozimma.so
```

2. Set an environmental variable to choose the compute mode
```bash
export OZIMMA_COMPUTE_MODE=fp64_int8_9
```
The supported compute modes are [here](#supported-compute-mode).

3. Execute the application

### Supported compute mode
| Mode       | Tensor Core type | Num splits |                   |
|:-----------|:-----------------|:-----------|:------------------|
|dgemm       | --               | --         | Disable hijacking |
|fp64_int8_6 | Int8 TC          | 6          |                   |
|fp64_int8_7 | Int8 TC          | 7          |                   |
|fp64_int8_8 | Int8 TC          | 8          |                   |
|fp64_int8_9 | Int8 TC          | 9          |                   |
|fp64_int8_10| Int8 TC          | 10         |                   |
|fp64_int8_11| Int8 TC          | 11         |                   |
|fp64_int8_12| Int8 TC          | 12         |                   |
|fp64_int8_13| Int8 TC          | 13         |                   |


### Optional environmental variables
```bash
# Show info log
export OZIMMA_INFO=1

# Show error and warning log
export OZIMMA_ERROR=1

# Show CULiP ( https://github.com/enp1s0/CULiP ) log
export OZIMMA_ENABLE_CULIP_PROFILING=1
```

## License
MIT
