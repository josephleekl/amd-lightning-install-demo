import pennylane as qml
import numpy as np
import time
import sys

device_name = "lightning.amdgpu"
repeats = 3

if len(sys.argv) > 1:
    try:
        n_qubits = int(sys.argv[1])
    except ValueError:
        print("Error: The number of qubits must be an integer.")
        sys.exit(1)
else:
    # Default to 30 if no argument is provided
    n_qubits = 30

print(f"Running with {n_qubits} qubits")

def circuit(n_qubits):
    """Mock performing a quantum Fourier transform.

    Args:
        n_qubits (int): number of wires.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def qft_circuit():
        qml.QFT(wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))
    return qft_circuit

# warmup 
circuit(n_qubits)()

start_time = time.time()
for _ in range(repeats):
    circuit(n_qubits)()
end_time = time.time()

print(f"Time taken for {n_qubits} qubits: {(end_time - start_time)/repeats:.4f} seconds")
