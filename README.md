# ozIMMU - DGEMM on Int8 Tensor Core

This library intercepts function calls for cuBLAS DGEMM functions and executes ozIMMU instead

## Build
```bash
git clone https://github.com/enp1s0/ozIMMU --recursive
cd ozIMMU
mkdir build
cd build
cmake ..
make -j4
```

## Usage

1. Set an environmental variable to hijack the function calls
```bash
export LD_PRELOAD=/path/to/ozIMMU/build/libozimmu.so
```

2. Set an environmental variable to choose the compute mode
```bash
export OZIMMU_COMPUTE_MODE=fp64_int8_9
```
The supported compute modes are [here](#supported-compute-mode).

3. Execute the application

### Supported compute mode
| Mode          | Tensor Core type | Num splits |                         |
|:--------------|:-----------------|:-----------|:------------------------|
|dgemm          | --               | --         | Disable hijacking       |
|fp64_int8_3    | Int8 TC          | 3          |                         |
|fp64_int8_4    | Int8 TC          | 4          |                         |
|fp64_int8_5    | Int8 TC          | 5          |                         |
|fp64_int8_6    | Int8 TC          | 6          |                         |
|fp64_int8_7    | Int8 TC          | 7          |                         |
|fp64_int8_8    | Int8 TC          | 8          |                         |
|fp64_int8_9    | Int8 TC          | 9          |                         |
|fp64_int8_10   | Int8 TC          | 10         |                         |
|fp64_int8_11   | Int8 TC          | 11         |                         |
|fp64_int8_12   | Int8 TC          | 12         |                         |
|fp64_int8_13   | Int8 TC          | 13         |                         |
|fp64_int8_14   | Int8 TC          | 14         |                         |
|fp64_int8_15   | Int8 TC          | 15         |                         |
|fp64_int8_16   | Int8 TC          | 16         |                         |
|fp64_int8_17   | Int8 TC          | 17         |                         |
|fp64_int8_18   | Int8 TC          | 18         |                         |
|fp64_int8_auto | Int8 TC          | AUTO       | fp64_int8_3..18 / dgemm |


### Optional environmental variables
```bash
# Show info log
export OZIMMU_INFO=1

# Show error and warning log
export OZIMMU_ERROR=1

# Show CULiP ( https://github.com/enp1s0/CULiP ) log
export OZIMMU_ENABLE_CULIP_PROFILING=1

# Choose malloc mode
export OZIMMU_MALLOC_ASYNC=1

# Set AUTO mode mantissa loss threshold
export OZIMMU_AUTO_AVG_MANTISSA_LOSS_THRESHOLD=1.5
```

## Citation
```bibtex
@misc{ootomo2023dgemm,
      title={DGEMM on Integer Matrix Multiplication Unit},
      author={Hiroyuki Ootomo and Katsuhisa Ozaki and Rio Yokota},
      year={2023},
      eprint={2306.11975},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

## License
MIT
