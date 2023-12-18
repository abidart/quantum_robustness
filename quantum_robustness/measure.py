import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
import matplotlib.pyplot as plt
import math


def create_adversarial_circuit(adversarial_power, qubits, corruption):
    circuit = QuantumCircuit(qubits, qubits)
    corrupted_qubits = random.sample(range(qubits), corruption)
    for corrupted_qubit in corrupted_qubits:
        y_angle = adversarial_power
        circuit.ry(y_angle, corrupted_qubit)
        z_angle = np.random.random_sample() * 2 * math.pi
        circuit.rz(z_angle, corrupted_qubit)
    circuit.barrier(label="adversary")
    return circuit


def adversarial_measurement(adversarial_power, qubits, corruption, draw=False):
    circuit = create_adversarial_circuit(
        adversarial_power=adversarial_power, qubits=qubits, corruption=corruption
    )

    circuit.measure(
        [qubit for qubit in range(qubits)], [qubit for qubit in range(qubits)]
    )

    if draw:
        circuit.draw(output="mpl", style="clifford")
        plt.show()

    return circuit


def execute_adversarial_circuit(
    shots, qubits, adversarial_power, corruption, correction
):
    _circuit = adversarial_measurement(
        adversarial_power=adversarial_power,
        qubits=qubits,
        corruption=corruption,
        correction=correction,
    )
    simulator = Aer.get_backend("qasm_simulator")
    result = execute(
        _circuit, backend=simulator, shots=shots, optimization_level=0
    ).result()
    return result


def execute_multiple_adversarial_circuits(
    qubits, adversarial_power, runs, shots, corruption, correction
):
    successes = 0
    str_key = "".join(["0"] * qubits)
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(runs):
            futures.append(
                executor.submit(
                    execute_adversarial_circuit,
                    shots=shots,
                    qubits=qubits,
                    adversarial_power=adversarial_power,
                    corruption=corruption,
                    correction=correction,
                )
            )
        for future in futures:
            result = future.result()
            counts = result.get_counts()
            if str_key in counts:
                successes += counts[str_key]

    return successes / shots / runs


def growing_power_experiment_with_correction(
    power_start,
    power_stop,
    qubits,
    corruption_start,
    corruption_stop,
    corruption_step,
    runs=5000,
    shots=1,
    step=0.314,
):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=96)
    powers = list(np.arange(power_start, power_stop, step))
    corruptions = list(np.arange(corruption_start, corruption_stop, corruption_step))
    for corruption in corruptions:
        half_success_rate = []
        for power in powers:
            distribution = execute_multiple_adversarial_circuits(
                adversarial_power=power,
                qubits=qubits,
                shots=shots,
                runs=runs,
                corruption=corruption,
                correction=0,
            )
            half_success_rate.append(distribution)
        ax.scatter(
            powers,
            half_success_rate,
            marker="x",
            label=f"{corruption}/{qubits}-corrupted",
        )
        cos_squared = [(math.cos(p / 2) ** 2) ** corruption for p in powers]
        ax.plot(
            powers,
            cos_squared,
            color="gray",
            linewidth=1.0,
            linestyle="--",
        )
    ax.set(
        xlim=(powers[0], powers[-1]),
        xticks=np.arange(0, powers[-1] + 0.314, step=0.628),
        ylim=(0, 1.1),
    )

    plt.xlabel("Power θ (radians)")
    plt.ylabel("Success Rate")
    plt.title(f"Measurement success rate vs Adversarial Angle θ")
    ax.legend(
        loc="upper right",
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    plt.show()


if __name__ == "__main__":
    growing_power_experiment_with_correction(
        0,
        3.15,
        step=0.314,
        corruption_start=1,
        corruption_stop=8,
        corruption_step=2,
        qubits=8,
        runs=1000,
        shots=1,
    )
