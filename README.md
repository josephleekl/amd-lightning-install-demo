## Kokkos-MPI installation docs

Follow instructions in here https://docs.pennylane.ai/projects/lightning/en/latest/lightning_kokkos/installation_hpc.html to install lightning-kokkos with MPI.

Note:

- You can ignore the following lines:

````
   export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CXX_FLAGS='--gcc-install-dir=/opt/cray/pe/gcc/11.2.0/snos/lib/gcc/x86_64-suse-linux/11.2.0/'"
    export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS:FILEPATH=/opt/rocm-6.2.4/lib/llvm/bin/clang-scan-deps"
    ```
````

- The extra MPI flags are probably not necessary on your cluster


## Execution

`mpirun -n <num_gpus> python qft.py <num_qubits>`
