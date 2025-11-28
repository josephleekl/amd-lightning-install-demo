import pennylane as qml
from pennylane import numpy as pnp
from timeit import default_timer as timer

# Try and scale up these numbers!
wires = 20
layers = 2

# Set a seed
pnp.random.seed(42)

# Set number of runs for timing averaging
num_runs = 3

# Use `lightning.qubit` for CPU 
# Use `lightning.amdgpu` for AMD GPU
dev = qml.device('lightning.amdgpu', wires=wires)

# Create QNode of device and circuit
@qml.qnode(dev)
def circuit(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(wires))
    return qml.math.hstack([qml.expval(qml.PauliZ(i)) for i in range(wires)])

# Set trainable parameters for calculating circuit Jacobian
shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=wires)
weights = pnp.random.random(size=shape)

# Run, calculate the quantum circuit Jacobian and average the timing results
timing = []
for t in range(num_runs):
    start = timer()
    jac = qml.jacobian(circuit)(weights)
    end = timer()
    timing.append(end - start)

print('Circuit measurements: \n',circuit(weights),'\n')
#print('Raw circuit: \n',qml.draw(circuit)(weights),'\n')
#print('Expanded circuit: \n',qml.draw(circuit,level='device')(weights),'\n')
print('Mean timing: ',qml.numpy.mean(timing),'\n')
