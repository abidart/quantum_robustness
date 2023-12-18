import random
import time
import timeit
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit_aer import StatevectorSimulator, AerSimulator
from qiskit.visualization import *
import matplotlib.pyplot as plt
from random import randrange, choice
import math


def create_adversarial_circuit(adversarial_power, qubits, corruption):
    circuit = QuantumCircuit(qubits, qubits - 1)
    # corrupted_qubits = random.sample(range(qubits), corruption)
    # for corrupted_qubit in corrupted_qubits:
    #     # y_angle = randrange(0, 100) / 100 * adversarial_power
    #     # adversarial_y_angle = math.pi * adversarial_power
    #     y_angle = adversarial_power
    #     # y_angle = np.random.random_sample() * adversarial_power
    #     circuit.ry(y_angle, corrupted_qubit)
    #     z_angle = np.random.random_sample() * 2 * math.pi
    #     circuit.rz(z_angle, corrupted_qubit)
    circuit.h(qubits - 1)
    circuit.barrier(label="adversary")
    return circuit


def adversarial_bv(
    secret_key, adversarial_power, qubits, corruption, correction=0, draw=False
):
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

    # robust_qubits = random.sample(
    #     range(qubits - 1), corruption
    # )  # TODO decide range qubits or qubits -1
    for qubit in range(corruption):
        # z_angle = np.random.random_sample() * 2 * math.pi
        # circuit.rz(z_angle, qubit)
        # y_angle = np.random.random_sample() * -adversarial_power
        circuit.ry(correction * adversarial_power, qubit)

    circuit.measure(
        [qubit for qubit in range(qubits - 1)], [qubit for qubit in range(qubits - 1)]
    )

    if draw:
        circuit.draw(output="mpl", style="clifford")
        plt.show()

    return circuit


def execute_adversarial_circuit(
    shots, qubits, adversarial_power, secret_key, corruption, correction
):
    _circuit = adversarial_bv(
        secret_key=secret_key,
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
    qubits, adversarial_power, runs, shots, secret_key, corruption, correction
):
    # answers = dict()
    successes = 0
    str_key = "".join(list(reversed(list(map(str, map(int, secret_key))))))
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
                    correction=correction,
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
            # answers[future.result()] = answers.get(future.result(), 0) + 1

    # sorted_total_count = {
    #     k: v / runs for k, v in sorted(answers.items(), key=lambda item: -item[1])
    # }

    return successes / shots / runs


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


def growing_power_experiment(
    power_start, power_stop, qubits, corruption, runs=5000, shots=1, step=0.314
):
    half_success_rate = []
    powers = list(np.arange(power_start, power_stop, step))
    for power in powers:
        ones = qubits // 2
        corruption = corruption
        secret_key = generate_key(qubits, ones)
        str_key = "".join(list(reversed(list(map(str, map(int, secret_key))))))
        print(str_key)
        # start = time.time()
        distribution = execute_multiple_adversarial_circuits(
            adversarial_power=power,
            qubits=qubits,
            shots=shots,
            runs=runs,
            secret_key=secret_key,
            corruption=corruption,
            correction=0,
        )
        half_success_rate.append(distribution)
        # end = time.time()
        # print(end - start)
        print(qubits, runs, shots, power, distribution)
        # if str_key in distribution:
        #     half_success_rate.append(distribution[str_key] / shots / runs)
        # else:
        #     half_success_rate.append(0)

        # plot_distribution(data=distribution)
        # plt.show()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=96)

    ax.scatter(powers, half_success_rate, marker="x", label="Bernstein-Vazirani")
    # ax.plot(qubit_count, success_rate, linewidth=2.0)
    cos_squared = [(math.cos(p / 2) ** 2) ** corruption for p in powers]
    ax.plot(powers, cos_squared, label="cos(θ/2)^2", color="orange")
    ax.set(
        xlim=(powers[0], powers[-1]),
        xticks=np.arange(0, powers[-1] + 0.314, step=0.628),
        ylim=(0, 1.1),
    )
    area = 0
    for i in range(len(powers) - 1):
        x1, y1 = powers[i], half_success_rate[i]
        x2, y2 = powers[i + 1], half_success_rate[i + 1]
        area += (y1 + y2) * (x2 - x1) / 2

    print("Area under the curve:", area)

    # plt.legend(["BV success rate", "cos(x/2)^2"], loc="lower right")
    plt.xlabel("θ (radians)")
    plt.ylabel("Success Rate")
    plt.title(f"Bernstein-Vazirani vs Adversarial Angle θ, area: {area}")
    plt.legend(loc="upper right", ncol=1)
    plt.show()


def growing_correction_experiment(
    correction_start,
    correction_stop,
    step,
    adversarial_power,
    qubits,
    corruption,
    runs=5000,
    shots=1,
):
    half_success_rate = []
    corrections = list(np.arange(correction_start, correction_stop, step))
    for correction in corrections:
        ones = qubits // 2
        secret_key = generate_key(qubits, ones)
        str_key = "".join(list(reversed(list(map(str, map(int, secret_key))))))
        print(str_key)
        # start = time.time()
        distribution = execute_multiple_adversarial_circuits(
            adversarial_power=adversarial_power,
            qubits=qubits,
            shots=shots,
            runs=runs,
            secret_key=secret_key,
            corruption=corruption,
            correction=correction,
        )
        half_success_rate.append(distribution)
        # end = time.time()
        # print(end - start)
        print(
            f"success rate={distribution}, {correction=}, {qubits=}, {adversarial_power=}, {runs=}, {shots=}"
        )
        # if str_key in distribution:
        #     half_success_rate.append(distribution[str_key] / shots / runs)
        # else:
        #     half_success_rate.append(0)

        # plot_distribution(data=distribution)
        # plt.show()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=96)

    ax.scatter(
        corrections, half_success_rate, marker="x", label=f"power={adversarial_power}"
    )
    # ax.plot(qubit_count, success_rate, linewidth=2.0)
    # cos_squared = [(math.cos(p / 2) ** 2) ** corruption for p in corrections]
    # ax.plot(powers, cos_squared, label="cos(θ/2)^2", color="orange")
    ax.set(
        xlim=(corrections[0], corrections[-1]),
        xticks=np.arange(0, corrections[-1] + step, step=2 * step),
        # ylim=(0, 1.1),
    )
    # area = 0
    # for i in range(len(powers) - 1):
    #     x1, y1 = powers[i], half_success_rate[i]
    #     x2, y2 = powers[i + 1], half_success_rate[i + 1]
    #     area += (y1 + y2) * (x2 - x1) / 2

    # print("Area under the curve:", area)

    # plt.legend(["BV success rate", "cos(x/2)^2"], loc="lower right")
    plt.xlabel("θ (radians)")
    plt.ylabel("Success Rate")
    plt.title(f"Bernstein-Vazirani vs correction")
    plt.legend(loc="upper right", ncol=1)
    plt.show()
    print(
        f"zero-correction {half_success_rate[0]}, max correction {max(half_success_rate)} at {corrections[half_success_rate.index(max(half_success_rate))]}"
    )


def growing_power_experiment_with_correction(
    power_start, power_stop, qubits, corruption, runs=5000, shots=1, step=0.314
):
    half_success_rate = []
    # half_success_rate_corrected = []
    powers = list(np.arange(power_start, power_stop, step))
    for power in powers:
        ones = qubits // 2
        secret_key = generate_key(qubits, qubits // 2)
        str_key = "".join(list(reversed(list(map(str, map(int, secret_key))))))
        print(str_key)
        # start = time.time()
        distribution = execute_multiple_adversarial_circuits(
            adversarial_power=power / corruption,
            qubits=qubits,
            shots=shots,
            runs=runs,
            secret_key=secret_key,
            corruption=corruption,
            correction=0,
        )
        half_success_rate.append(distribution)
        # if power >= 2.5:
        #     distribution_corrected = execute_multiple_adversarial_circuits(
        #         adversarial_power=power / qubits,
        #         qubits=qubits,
        #         shots=shots,
        #         runs=runs,
        #         secret_key=secret_key,
        #         corruption=corruption,
        #         correction=1,
        #     )
        #     half_success_rate_corrected.append(distribution_corrected)
        # else:
        #     half_success_rate_corrected.append(distribution)
        # end = time.time()
        # print(end - start)
        print(qubits, runs, shots, power, distribution)
        # if str_key in distribution:
        #     half_success_rate.append(distribution[str_key] / shots / runs)
        # else:
        #     half_success_rate.append(0)

        # plot_distribution(data=distribution)
        # plt.show()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=96)

    ax.scatter(powers, half_success_rate, marker="x", label=f"uncorrected")
    # ax.scatter(
    #     powers,
    #     half_success_rate_corrected,
    #     marker="+",
    #     label=f"{corruption}-corrected",
    # )
    # ax.plot(qubit_count, success_rate, linewidth=2.0)
    cos_squared = [(math.cos(p / 2) ** 2) ** 1 for p in powers]
    ax.plot(powers, cos_squared, label=f"cos(θ/2)^{2*corruption}", color="orange")
    ax.set(
        xlim=(powers[0], powers[-1]),
        xticks=np.arange(0, powers[-1] + 0.314, step=0.628),
        ylim=(0, 1.1),
    )
    # area = 0
    # area_corrected = 0
    # for i in range(len(powers) - 1):
    #     x1, y1 = powers[i], half_success_rate[i]
    #     x2, y2 = powers[i + 1], half_success_rate[i + 1]
    #     area += (y1 + y2) * (x2 - x1) / 2
    #     x1c, y1c = powers[i], half_success_rate_corrected[i]
    #     x2c, y2c = powers[i + 1], half_success_rate_corrected[i + 1]
    #     area_corrected += (y1c + y2c) * (x2c - x1c) / 2
    #
    # print("Area under the curve:", area)
    # print("Area under the corrected curve:", area_corrected)
    # print("Difference:", area_corrected - area)

    # plt.legend(["BV success rate", "cos(x/2)^2"], loc="lower right")
    plt.xlabel("θ (radians)")
    plt.ylabel("Success Rate")
    plt.title(
        f"({corruption}/{qubits})-corrupted Bernstein-Vazirani vs Adversarial Angle θ"
    )
    # plt.legend(loc="upper right", ncol=1)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.show()


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
    adversarial_bv(
        secret_key=(True, False),
        adversarial_power=math.pi * 0.1,
        qubits=3,
        corruption=1,
        draw=True,
    )
    # growing_power_experiment(
    #     0, 3.15, step=0.157, corruption=1, qubits=6, runs=200, shots=1
    # )
    # growing_correction_experiment(
    #     correction_start=0,
    #     correction_stop=1.5,
    #     step=0.1,
    #     adversarial_power=2.5,
    #     qubits=8,
    #     corruption=1,
    #     runs=5000,
    #     shots=100,
    # )
    growing_power_experiment_with_correction(
        0, 3.15, step=0.25, corruption=1, qubits=3, runs=1, shots=1
    )
