import pennylane as qp
import numpy as np
import sys
from timeit import default_timer as timer

device_name = "lightning.kokkos"
mpi = True
repeats = 5

def circuit(n_qubits):
    """Mock performing a quantum Fourier transform.

    Args:
        n_qubits (int): number of wires.
    """
    dev = qp.device(device_name, wires=n_qubits, mpi=mpi)

    @qp.qnode(dev, diff_method=None)
    def qft_circuit():
        qp.QFT(wires=range(n_qubits))
        return qp.expval(qp.PauliZ(0))
    return qft_circuit


if len(sys.argv) < 2:
    print("Usage: python test.py <n_qubits>")
    sys.exit(1)

n_qubits = int(sys.argv[1])

# warmup
circuit(n_qubits)()


start_time = timer()
for _ in range(repeats):
    circuit(n_qubits)()
end_time = timer()
print(f"Time taken for {n_qubits} qubits: {(end_time - start_time)/repeats:.4f} seconds")
