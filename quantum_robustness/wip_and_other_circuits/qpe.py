import random
import time
import timeit
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.primitives import Sampler
from qiskit_algorithms import AmplificationProblem
from qiskit.circuit.library import MCMT, ZGate, GroverOperator
from qiskit_aer import StatevectorSimulator, AerSimulator
from qiskit.visualization import *
import matplotlib.pyplot as plt
from random import randrange, choice
import math

from qiskit.circuit.library.phase_oracle import PhaseOracle


def create_adversarial_circuit(adversarial_power, qubits, corruption):
    circuit = QuantumCircuit(qubits, qubits - 1)
    corrupted_qubits = random.sample(range(qubits), corruption)
    for corrupted_qubit in corrupted_qubits:
        # y_angle = randrange(0, 100) / 100 * adversarial_power
        # y_angle = np.random.random_sample() * adversarial_power
        # adversarial_y_angle = math.pi * adversarial_power
        z_angle = np.random.random_sample() * 2 * math.pi
        circuit.ry(adversarial_power, corrupted_qubit)
        circuit.rz(z_angle, corrupted_qubit)
    circuit.barrier(label="adversary")
    return circuit


def adversarial_grover(secret_key, adversarial_power, qubits, corruption, draw=False):
    circuit = create_adversarial_circuit(
        adversarial_power=adversarial_power, qubits=qubits, corruption=corruption
    )

    for qubit in range(qubits - 1):
        circuit.h(qubit)

    circuit.barrier()
    circuit.x(qubits - 1)
    circuit.h(qubits - 1)

    circuit.barrier()

    for pos, is_set in enumerate(secret_key):
        if is_set:
            circuit.cx(pos, qubits - 1)

    circuit.barrier(label="secret key")
    for qubit in range(qubits - 1):
        circuit.h(qubit)

    circuit.barrier()

    circuit.measure(
        [qubit for qubit in range(qubits - 1)], [qubit for qubit in range(qubits - 1)]
    )

    if draw:
        circuit.draw(output="mpl", style="clifford")
        plt.show()

    return circuit


def execute_adversarial_circuit(
    shots, qubits, adversarial_power, secret_key, corruption
):
    _circuit = adversarial_grover(
        secret_key=secret_key,
        adversarial_power=adversarial_power,
        qubits=qubits,
        corruption=corruption,
    )
    simulator = Aer.get_backend("qasm_simulator")
    result = execute(
        _circuit, backend=simulator, shots=shots, optimization_level=0
    ).result()
    counts = result.get_counts()
    return list(counts.keys())[0]


def execute_multiple_adversarial_circuits(
    qubits, adversarial_power, runs, shots, secret_key, corruption
):
    answers = dict()
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(runs):
            futures.append(
                executor.submit(
                    execute_adversarial_circuit,
                    shots=shots,
                    qubits=qubits,
                    adversarial_power=adversarial_power,
                    secret_key=secret_key,
                    corruption=corruption,
                )
            )
            # answer = execute_adversarial_circuit(
            #     circuit_to_run=circuit_to_run, shots=shots
            # )
            # answers[answer] = answers.get(answer, 0) + 1
        for future in futures:
            answers[future.result()] = answers.get(future.result(), 0) + 1

    sorted_total_count = {
        k: v / runs for k, v in sorted(answers.items(), key=lambda item: -item[1])
    }

    return sorted_total_count


def generate_key(qubits, ones):
    one_position = random.sample(range(qubits - 1), ones)
    values = [False] * (qubits - 1)
    for pos in one_position:
        values[pos] = True

    return tuple(values)


def multiple_key_experiments(qubit_start, qubit_stop, power=0.1, shots=5000, runs=1):
    half_success_rate = []
    zero_success_rate = []
    one_success_rate = []
    qubit_count = []
    for qubits in range(qubit_start, qubit_stop + 1, 2):
        ones = qubits // 2
        corruption = qubits // 2
        secret_key = generate_key(qubits, ones)
        str_key = "".join(list(reversed(list(map(str, map(int, secret_key))))))
        print(str_key)
        start = time.time()
        distribution = execute_multiple_adversarial_circuits(
            adversarial_power=power,
            qubits=qubits,
            shots=shots,
            runs=runs,
            secret_key=secret_key,
            corruption=corruption,
        )
        end = time.time()
        print(end - start)
        print(qubits, runs, shots, power, distribution)
        half_success_rate.append(distribution[str_key])
        qubit_count.append(qubits)

        zero_key = [False] * (qubits - 1)
        zero_str_key = "".join(list(reversed(list(map(str, map(int, zero_key))))))
        one_key = [True] * (qubits - 1)
        one_str_key = "".join(list(reversed(list(map(str, map(int, one_key))))))

        zero_distribution = execute_multiple_adversarial_circuits(
            adversarial_power=power,
            qubits=qubits,
            shots=shots,
            runs=runs,
            secret_key=zero_key,
            corruption=corruption,
        )

        one_distribution = execute_multiple_adversarial_circuits(
            adversarial_power=power,
            qubits=qubits,
            shots=shots,
            runs=runs,
            secret_key=one_key,
            corruption=corruption,
        )

        zero_success_rate.append(zero_distribution[zero_str_key])
        one_success_rate.append(one_distribution[one_str_key])

        # plot_distribution(data=distribution)
        # plt.show()

    fig, ax = plt.subplots()

    ax.scatter(qubit_count, half_success_rate, marker="x")
    ax.scatter(qubit_count, one_success_rate, marker="^")
    ax.scatter(qubit_count, zero_success_rate, marker="o")
    # ax.plot(qubit_count, success_rate, linewidth=2.0)

    ax.set(
        xlim=(qubit_count[0], qubit_count[-1]),
        xticks=np.arange(0, qubit_count[-1] + 3, step=2),
        ylim=(0.75, 1),
    )

    plt.show()


def growing_power_experiment(power_start, power_stop, qubits, runs=5000, shots=1):
    half_success_rate = []
    power_tracker = []
    for power in range(power_start, power_stop + 1, 5):
        ones = qubits // 2
        corruption = qubits // 2
        secret_key = generate_key(qubits, ones)
        str_key = "".join(list(reversed(list(map(str, map(int, secret_key))))))
        print(str_key)
        start = time.time()
        distribution = execute_multiple_adversarial_circuits(
            adversarial_power=power / 100,
            qubits=qubits,
            shots=shots,
            runs=runs,
            secret_key=secret_key,
            corruption=corruption,
        )
        end = time.time()
        print(end - start)
        print(qubits, runs, shots, power, distribution)
        half_success_rate.append(distribution[str_key])
        power_tracker.append(power)

        # plot_distribution(data=distribution)
        # plt.show()

    fig, ax = plt.subplots()

    ax.scatter(power_tracker, half_success_rate, marker="x")
    # ax.plot(qubit_count, success_rate, linewidth=2.0)

    ax.set(
        xlim=(power_tracker[0], power_tracker[-1]),
        xticks=np.arange(0, power_tracker[-1] + 3, step=5),
        ylim=(0, 1),
    )

    plt.show()


def execute_circuit_vqe(circuit, backend, shots, ol, power, qubits, corruption):
    qc = create_adversarial_circuit(
        adversarial_power=power,
        qubits=qubits,
        corruption=corruption,
    )

    qc.compose(circuit, inplace=True)

    _result = execute(qc, backend=backend, shots=shots, optimization_level=ol).result()
    return _result


if __name__ == "__main__":
    # for level in range(0, 8):
    #     power = level / 10
    #     distribution = run_adversarial_bv_circuits(
    #         adversarial_power=power,
    #         qubits=QUBITS,
    #         bits=BITS,
    #         shots=SHOTS,
    #         runs=RUNS
    #     )
    #     print(power, distribution)
    #     plot_distribution(data=distribution)

    # SHOTS = 1
    # RUNS = 5000
    # POWER = 0.1
    # QUBITS = 16

    # secret_key = tuple([True] * (QUBITS - 1))
    # adversarial_bv(
    #     secret_key=secret_key,
    #     adversarial_power=POWER,
    #     qubits=QUBITS,
    #     corruption=CORRUPTION,
    #     draw=True,
    # )

    # growing_power_experiment(5, 50, qubits=16)
    qubits = 14
    qpe = QuantumCircuit(qubits, qubits - 1)
    qpe.x(qubits - 1)
    qpe.draw()

    for qubit in range(qubits - 1):
        qpe.h(qubit)
    # qpe.draw()

    repetitions = 1
    for counting_qubit in range(qubits - 1):
        for i in range(repetitions):
            qpe.cp(math.pi / 4, counting_qubit, qubits - 1)
            # This is CU
        repetitions *= 2
    # qpe.draw()

    def qft_dagger(qc, n):
        """n-qubit QFTdagger the first n qubits in circ"""
        # Don't forget the Swaps!
        for qubit in range(n // 2):
            qc.swap(qubit, n - qubit - 1)
        for j in range(n):
            for m in range(j):
                qc.cp(-math.pi / float(2 ** (j - m)), m, j)
            qc.h(j)

    qpe.barrier()
    # Apply inverse QFT
    qft_dagger(qpe, qubits - 1)
    # Measure
    qpe.barrier()
    for n in range(qubits - 1):
        qpe.measure(n, n)
    # qpe.draw(output="mpl", style="clifford")

    # plt.show()

    simulator = Aer.get_backend("qasm_simulator")
    powers = list(np.arange(0, 3.142, 0.314))
    fig, ax = plt.subplots()
    ax.set(
        xlim=(powers[0], powers[-1]),
        xticks=np.arange(0, powers[-1] + 0.314, step=0.628),
        ylim=(0, 1.1),
    )
    list_key = ["0"] * (qubits - 1)
    list_key[2] = "1"
    str_key = "".join(list_key)

    shots = 1
    runs = 20

    for corruption in range(1, 8, 2):
        success_rates = []
        for power in powers:
            total_counts = dict()
            successes = 0
            ###
            with ProcessPoolExecutor() as executor:
                futures = []
                for _ in range(runs):
                    futures.append(
                        executor.submit(
                            execute_circuit_vqe,
                            circuit=qpe,
                            backend=simulator,
                            shots=shots,
                            ol=0,
                            power=power,
                            qubits=qubits,
                            corruption=corruption,
                        )
                    )
                    # answer = execute_adversarial_circuit(
                    #     circuit_to_run=circuit_to_run, shots=shots
                    # )
                    # answers[answer] = answers.get(answer, 0) + 1
                for future in futures:
                    result = future.result()
                    counts = result.get_counts()
                    if str_key in counts:
                        successes += counts[str_key]
            ###
            # for shot in range(shots):
            #     if shot % 10 == 0:
            #         print(shot)
            #     result = execute(
            #         qc, backend=simulator, shots=1, optimization_level=0
            #     ).result()
            #     counts = result.get_counts()
            #     if str_key in counts:
            #         successes += counts[str_key]

            # sorted_counts = {
            #     k: v / shots
            #     for k, v in sorted(counts.items(), key=lambda item: -item[1])
            # }
            print(power, successes)
            # print(power, sorted_counts)
            success_rates.append(successes / shots / runs)
            # print(power, "Success")
            # except Exception:
            #     success_rates.append(0)
            # print("Failure at power", power)

        # print(success_rates)
        # print(powers)

        ax.plot(powers, success_rates, label=f"{corruption}/14")
        # ax.plot(qubit_count, success_rate, linewidth=2.0)

    # cos = [math.cos(p / 200 * math.pi) for p in powers]
    # cos_squared = [(math.cos(p / 2) ** 2) ** 1 for p in powers]
    # ax.scatter(powers, cos)
    # ax.scatter(powers, cos_squared)
    plt.xlabel("θ (radians)")
    plt.ylabel("Success Rate")
    plt.title("Quantum Phase Estimation vs Adversarial Angle θ")
    plt.legend(loc="upper right", ncol=1)
    plt.show()
    print("test")
