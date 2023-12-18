import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import MCMT, XGate
import matplotlib.pyplot as plt
import math
from scipy import stats


def adversarial_init(adversarial_power, qubits, corruption) -> QuantumCircuit:
    circuit = QuantumCircuit(qubits)
    corrupted_qubits = random.sample(range(qubits), corruption)
    for corrupted_qubit in corrupted_qubits:
        circuit.ry(adversarial_power, corrupted_qubit)
        z_angle = np.random.random_sample() * 2 * math.pi
        circuit.rz(z_angle, corrupted_qubit)
    circuit.barrier(label="adversary")
    return circuit


def make_grover_circuit(marked_state, iterations) -> QuantumCircuit:
    marked_state_length = len(marked_state)
    qubits = marked_state_length + 1
    circuit = QuantumCircuit(qubits, marked_state_length)
    num_iterations = (
        math.floor(math.pi / 4 * math.sqrt(2**marked_state_length))
        if iterations
        else 1
    )

    circuit.x(qubits - 1)
    circuit.barrier()

    for qubit in range(qubits):
        circuit.h(qubit)

    circuit.barrier()

    for _ in range(num_iterations):
        for pos, val in enumerate(reversed(marked_state)):
            if val == "0":
                circuit.x(pos)

        circuit.compose(MCMT(XGate(), qubits - 1, 1), inplace=True)

        for pos, val in enumerate(reversed(marked_state)):
            if val == "0":
                circuit.x(pos)

        circuit.barrier(label=f"oracle ({marked_state})")
        for qubit in range(qubits - 1):
            circuit.h(qubit)
            circuit.x(qubit)
        circuit.barrier()
        circuit.h(qubits - 2)
        circuit.compose(MCMT(XGate(), qubits - 2, 1), inplace=True)
        circuit.h(qubits - 2)
        circuit.barrier()
        for qubit in range(qubits - 1):
            circuit.x(qubit)
            circuit.h(qubit)

        circuit.barrier()

    circuit.measure(
        [qubit for qubit in range(qubits - 1)], [qubit for qubit in range(qubits - 1)]
    )

    return circuit


def plot_grover(marked_state):
    circuit = make_grover_circuit(marked_state=marked_state)
    circuit.draw(output="mpl", style="clifford")
    plt.show()


def execute_adversarial_circuit(
    qubits, adversarial_power, corruption, iterations
) -> str:
    circuit = adversarial_init(
        adversarial_power=adversarial_power, qubits=qubits, corruption=corruption
    )
    marked_state = generate_marked_state(qubits)
    circuit.compose(
        make_grover_circuit(
            marked_state=marked_state,
            iterations=iterations,
        ),
        inplace=True,
    )
    simulator = Aer.get_backend("qasm_simulator")
    result = execute(circuit, backend=simulator, shots=1, optimization_level=0).result()
    counts = result.get_counts()
    return counts.get(marked_state, 0)


def execute_multiple_adversarial_circuits(
    qubits, adversarial_power, corruption, runs, iterations
):
    successes = 0
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(runs):
            futures.append(
                executor.submit(
                    execute_adversarial_circuit,
                    qubits=qubits,
                    adversarial_power=adversarial_power,
                    corruption=corruption,
                    iterations=iterations,
                )
            )
        for future in futures:
            successes += future.result()

    return successes / runs


def generate_marked_state(qubits) -> str:
    return "".join(random.choice(["0", "1"]) for _ in range(qubits - 1))


def update_stats(old_mean, old_variance, old_count, new_value):
    new_count = old_count + 1
    new_mean = old_mean + (new_value - old_mean) / new_count
    new_variance = old_variance + (new_value - old_mean) * (new_value - new_mean)
    return new_mean, new_variance, new_count


def growing_qubits_experiment(
    qubits_start,
    qubits_stop,
    step,
    adversarial_power,
    corruption,
    runs,
    iterations,
    trials,
):
    qubit_sizes = list(np.arange(qubits_start, qubits_stop, step))
    corrupted_success_rate = []
    uncorrupted_success_rate = []
    confidence_intervals = []
    robustness = []

    for qubits in qubit_sizes:
        uncorrupted_distribution = execute_multiple_adversarial_circuits(
            qubits=qubits,
            adversarial_power=adversarial_power,
            corruption=0,
            runs=runs,
            iterations=iterations,
        )
        uncorrupted_success_rate.append(uncorrupted_distribution)

        mean_corrupted = 0
        variance_corrupted = 0
        count_corrupted = 0
        for _ in range(trials):
            corrupted_distribution = execute_multiple_adversarial_circuits(
                qubits=qubits,
                adversarial_power=adversarial_power,
                corruption=corruption,
                runs=runs,
                iterations=iterations,
            )
            # Calculate the average and 95% confidence interval
            mean_corrupted, variance_corrupted, count_corrupted = update_stats(
                mean_corrupted,
                variance_corrupted,
                count_corrupted,
                corrupted_distribution,
            )
        std_dev_corrupted = np.sqrt(variance_corrupted / count_corrupted)
        std_err_corrupted = std_dev_corrupted / np.sqrt(count_corrupted)
        corrupted_success_rate.append(mean_corrupted)
        robustness.append(mean_corrupted / uncorrupted_distribution)
        ci_95 = stats.t.ppf(1 - 0.025, count_corrupted - 1) * std_err_corrupted
        confidence_intervals.append(ci_95 / uncorrupted_distribution)

        print(
            f"{mean_corrupted=}, {uncorrupted_distribution=}, {qubits=}, {adversarial_power=}, {runs=}"
        )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=96)

    ax.errorbar(
        qubit_sizes,
        robustness,
        yerr=confidence_intervals,
        fmt="o",
        capsize=10,
        label=f"95% CI (theta = {adversarial_power:.2f}, {runs=}, {trials=})",
    )
    ax.axhline(0.5, linestyle="--", color="gray", label="baseline")
    ax.set(
        xlim=(qubit_sizes[0], qubit_sizes[-1]),
        xticks=np.arange(0, qubit_sizes[-1] + 2, step=step),
        ylim=(0.35, 0.7),
    )

    plt.xlabel("n (qubits)")
    plt.ylabel("p tilde")
    plt.title(f"Grover's Algorithm Robustness")
    ax.legend(
        loc="upper left",
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    plt.show()


def growing_powers_experiment(
    power_start,
    power_stop,
    step,
    qubits,
    corruption_start,
    corruption_stop,
    corruption_step,
    runs,
    iterations,
):
    powers = list(np.arange(power_start, power_stop, step))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=96)
    corruptions = list(np.arange(corruption_start, corruption_stop, corruption_step))
    corruption_to_robustness = dict()
    for corruption in corruptions:
        corruption_to_robustness[corruption] = []
    for power in powers:
        uncorrupted_success_rate = []
        uncorrupted_distribution = execute_multiple_adversarial_circuits(
            qubits=qubits,
            adversarial_power=power,
            corruption=0,
            runs=runs,
            iterations=iterations,
        )
        uncorrupted_success_rate.append(uncorrupted_distribution)

        for corruption in corruptions:
            corrupted_success_rate = []

            corrupted_distribution = execute_multiple_adversarial_circuits(
                qubits=qubits,
                adversarial_power=power,
                corruption=corruption,
                runs=runs,
                iterations=iterations,
            )
            corrupted_success_rate.append(corrupted_distribution)
            if uncorrupted_distribution == 0.0:
                if corrupted_distribution == 0.0:
                    corruption_to_robustness[corruption].append(0)
                else:
                    corruption_to_robustness[corruption].append(1)
            else:
                corruption_to_robustness[corruption].append(
                    corrupted_distribution / uncorrupted_distribution
                )

            print(
                f"{power=}, {corrupted_distribution=}, {uncorrupted_distribution=}, {qubits=}, {corruption=}, {runs=}"
            )

    colors = ["blue", "orange", "green", "yellow", "red"]
    for key in corruption_to_robustness:
        cos_squared = [(math.cos(p / 2) ** 2) ** key for p in powers]
        ax.plot(
            powers,
            cos_squared,
            label=f"cos(θ/2)^{2*key}",
            color=colors[(key - corruption_start) // corruption_step],
            linewidth=1.0,
            linestyle="--",
        )
        ax.scatter(
            powers,
            corruption_to_robustness[key],
            marker="x",
            label=f"{key}/{qubits}-corrupted",
            linewidth=1.0,
            color=colors[(key - corruption_start) // corruption_step],
        )
    ax.set(
        xlim=(powers[0], powers[-1]),
        xticks=np.arange(powers[0], powers[-1] + step, step=step),
        ylim=(0, 1.05),
    )

    plt.xlabel("Power θ (radians)")
    plt.ylabel("Robustness")
    plt.title(f"Robustness for ss-GA vs power ({qubits} qubits)")
    ax.legend(
        loc="upper right",
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    plt.show()


def constrained_vs_unconstrained_experiment(
    qubits_start,
    qubits_stop,
    step,
    adversarial_power,
    runs,
):
    qubit_sizes = list(np.arange(qubits_start, qubits_stop, step))
    constrained_success_rate = []
    unconstrained_success_rate = []

    for qubits in qubit_sizes:
        unconstrained_distribution = execute_multiple_adversarial_circuits(
            qubits=qubits,
            adversarial_power=adversarial_power,
            corruption=0,
            runs=runs,
            iterations=True,
        )
        unconstrained_success_rate.append(unconstrained_distribution)

        constrained_distribution = execute_multiple_adversarial_circuits(
            qubits=qubits,
            adversarial_power=adversarial_power,
            corruption=0,
            runs=runs,
            iterations=False,
        )
        constrained_success_rate.append(constrained_distribution)

        print(
            f"{constrained_distribution=}, {unconstrained_distribution=}, {qubits=}, {adversarial_power=}, {runs=}"
        )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=96)

    ax.scatter(qubit_sizes, constrained_success_rate, marker="x", label="ss-GA")
    ax.scatter(qubit_sizes, unconstrained_success_rate, marker="o", label="GA")
    ax.set(
        xlim=(qubit_sizes[0], qubit_sizes[-1]),
        xticks=np.arange(qubit_sizes[0], qubit_sizes[-1] + 2, step=step),
        ylim=(0, 1.1),
    )
    plt.xlabel("n (qubits)")
    plt.ylabel("p")
    plt.title(f"ss-GA vs GA success rate")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.show()


if __name__ == "__main__":
    growing_qubits_experiment(
        qubits_start=10,
        qubits_stop=14,
        step=1,
        adversarial_power=np.pi / 2,
        corruption=1,
        runs=5000,
        iterations=True,
        trials=10,
    )
    # constrained_vs_unconstrained_experiment(
    #     qubits_start=3,
    #     qubits_stop=13,
    #     step=1,
    #     adversarial_power=0,
    #     runs=5000,
    # )
    # growing_powers_experiment(
    #     power_start=0,
    #     power_stop=3.15,
    #     step=0.314,
    #     qubits=9,
    #     corruption_start=1,
    #     corruption_stop=7,
    #     corruption_step=2,
    #     runs=5000,
    #     iterations=False,
    # )
