from qiskit.visualization import visualize_transition
from qiskit import QuantumCircuit
from math import pi
import numpy as np

circuit = QuantumCircuit(1)

circuit.ry(np.pi / 2, 0)

visualize_transition(circuit=circuit, trace=True)
