# ozIMMA : DGEMM on IMMA Tensor Cores

A test library for DGEMM equivalent GEMM on Integer Tensor Cores

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
1. Set `LD_PRELOAD`.
```bash
export LD_PRELOAD=/path/to/build/libozimma.so:$LD_PRELOAD
```

2. Set compute mode
```bash
export OZIMMA_COMPUTE_MODE=fp64_int8_9
```
Compute modes:
```
fp64_int8_6 fp64_int8_7 fp64_int8_8 fp64_int8_9 fp64_int8_10 fp64_int8_11 fp64_int8_12 fp64_int8_13 dgemm
```

### Environmental variable
```bash
# Enable / disable standard log output
export OZIMMA_INFO=[1 or 0]

# Enable / disable error log output
export OZIMMA_ERROR_LOG=[1 or 0]

# Enable / disable CULiP log output
export OZIMMA_ENABLE_CULIP_PROFILING=[0 or 1]
```

## License
MIT
