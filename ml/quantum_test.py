
import pennylane as qml
from pennylane import numpy as np

# Use 4 simulated qubits
n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode classical features into qubits
    qml.AngleEmbedding(
        inputs,
        wires=range(n_qubits),
        rotation="Y"
    )

    # Trainable quantum layers
    qml.BasicEntanglerLayers(
        weights,
        wires=range(n_qubits)
    )

    # Return measurement results
    return [
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]


# Example input: 4 features
inputs = np.array([0.1, 0.2, 0.3, 0.4])

# Random initial quantum parameters
weights = np.random.random((n_layers, n_qubits))

result = quantum_circuit(inputs, weights)

print("Quantum circuit output:")
print(result)

# Test with a different input
inputs_2 = np.array([0.8, 0.6, 0.4, 0.2])

result_2 = quantum_circuit(inputs_2, weights)

print("\nOutput for second input:")
print(result_2)