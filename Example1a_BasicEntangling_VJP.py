import pennylane as qml
import jax.numpy as jnp
from timeit import default_timer as timer
# For pennylane==0.43 , this requires jax==0.6.2
import jax

# Try and scale up these numbers!
wires = 20
layers = 2

# Set a seed
key = jax.random.PRNGKey(0)

# Set number of runs for timing averaging
num_runs = 3

# Use `lightning.qubit` for CPU 
# Use `lightning.amdgpu` for AMD GPU
dev = qml.device('lightning.amdgpu', wires=wires)

# Create QNode of device and circuit
@qml.qnode(dev)
def circuit(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(wires))
    return qml.expval(qml.PauliZ(wires=0))

# Set trainable parameters 
shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=wires)
weights = jax.random.uniform(key, shape)

# Generate a random cotangent vector (matching output shape of circuit)
key, subkey = jax.random.split(key)
cotangent_vector = jax.random.uniform(subkey, shape=()) 

# Define VJP wrapper
def compute_vjp(params, vec):
    primals, vjp_fn = jax.vjp(circuit, params)
    return vjp_fn(vec)[0]

# JIT the VJP function
jit_vjp = jax.jit(compute_vjp)

# Warm-up run
_ = jit_vjp(weights, cotangent_vector).block_until_ready()

timing = []
vjp_res = None

for t in range(num_runs):
    start = timer()
    # Pass weights and the random cotangent vector
    vjp_res = jit_vjp(weights, cotangent_vector).block_until_ready()
    end = timer()
    timing.append(end - start)


print('Circuit measurements: \n', circuit(weights), '\n')
print('VJP Result (Gradient scaled by random vector): \n', vjp_res[0], '\n')
print('Mean timing: ', qml.numpy.mean(timing), '\n')
