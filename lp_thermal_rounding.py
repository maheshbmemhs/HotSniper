import numpy as np
from scipy.optimize import linprog


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_min = np.min(x)
    x_max = np.max(x)
    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def validate_problem(
    throughput: np.ndarray,
    power: np.ndarray,
    core_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    throughput = np.asarray(throughput, dtype=float)
    power = np.asarray(power, dtype=float)
    core_count = np.asarray(core_count, dtype=int)

    if throughput.ndim != 2:
        raise ValueError("throughput must be a 2D matrix.")
    if power.shape != throughput.shape:
        raise ValueError("power must have the same shape as throughput.")
    if core_count.ndim != 1 or core_count.shape[0] != throughput.shape[1]:
        raise ValueError("core_count must have one entry per state.")
    if np.any(core_count < 0):
        raise ValueError("core_count must be non-negative.")
    if np.sum(core_count) < throughput.shape[0]:
        raise ValueError("Total core capacity must cover all threads in this prototype.")

    return throughput, power, core_count


def solve_lp_relaxation(
    throughput: np.ndarray,
    power: np.ndarray,
    throughput_requirement: float,
    core_count: np.ndarray,
) -> np.ndarray:
    throughput, power, core_count = validate_problem(throughput, power, core_count)

    n_threads, n_states = throughput.shape
    n_vars = n_threads * n_states

    c = power.flatten()
    a_ub = []
    b_ub = []

    a_ub.append(-throughput.flatten())
    b_ub.append(-float(throughput_requirement))

    for i in range(n_threads):
        row = np.zeros(n_vars)
        for j in range(n_states):
            row[i * n_states + j] = 1.0
        a_ub.append(row)
        b_ub.append(1.0)

    for j in range(n_states):
        row = np.zeros(n_vars)
        for i in range(n_threads):
            row[i * n_states + j] = 1.0
        a_ub.append(row)
        b_ub.append(float(core_count[j]))

    result = linprog(
        c=c,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        bounds=[(0.0, 1.0) for _ in range(n_vars)],
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"LP relaxation failed: {result.message}")

    return result.x.reshape((n_threads, n_states))


def thermal_aware_capacity_rounding(
    a_lp: np.ndarray,
    power: np.ndarray,
    temperature: np.ndarray,
    core_count: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.5,
) -> np.ndarray:
    a_lp = np.asarray(a_lp, dtype=float)
    power = np.asarray(power, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    core_count = np.asarray(core_count, dtype=int)

    if power.shape != a_lp.shape:
        raise ValueError("power must have the same shape as a_lp.")
    if temperature.ndim != 1 or temperature.shape[0] != a_lp.shape[1]:
        raise ValueError("temperature must have one entry per state.")
    if core_count.ndim != 1 or core_count.shape[0] != a_lp.shape[1]:
        raise ValueError("core_count must have one entry per state.")
    if np.sum(core_count) < a_lp.shape[0]:
        raise ValueError("Total core capacity must cover all threads in this prototype.")

    n_threads, n_states = a_lp.shape
    norm_power = normalize(power)
    norm_temp = normalize(temperature)
    score = alpha * a_lp - beta * norm_power - gamma * norm_temp.reshape(1, n_states)

    candidates = [
        (score[i, j], i, j)
        for i in range(n_threads)
        for j in range(n_states)
    ]
    candidates.sort(reverse=True, key=lambda item: item[0])

    a_round = np.zeros_like(a_lp, dtype=int)
    assigned = np.zeros(n_threads, dtype=bool)
    remaining_capacity = core_count.copy()

    for _, i, j in candidates:
        if not assigned[i] and remaining_capacity[j] > 0:
            a_round[i, j] = 1
            assigned[i] = True
            remaining_capacity[j] -= 1

    for i in range(n_threads):
        if assigned[i]:
            continue
        feasible_states = np.where(remaining_capacity > 0)[0]
        if feasible_states.size == 0:
            raise RuntimeError("No remaining capacity to assign all threads.")
        best_j = max(feasible_states, key=lambda j: score[i, j])
        a_round[i, best_j] = 1
        assigned[i] = True
        remaining_capacity[best_j] -= 1

    return a_round


def assignment_throughput(assignment: np.ndarray, throughput: np.ndarray) -> float:
    return float(np.sum(np.asarray(assignment) * np.asarray(throughput)))


def assignment_power(assignment: np.ndarray, power: np.ndarray) -> float:
    return float(np.sum(np.asarray(assignment) * np.asarray(power)))


def validate_assignment(assignment: np.ndarray, core_count: np.ndarray) -> None:
    assignment = np.asarray(assignment)
    core_count = np.asarray(core_count, dtype=int)

    if not np.all((assignment == 0) | (assignment == 1)):
        raise RuntimeError("Assignment must be binary.")
    if np.any(np.sum(assignment, axis=1) != 1):
        raise RuntimeError("Every thread must have exactly one assignment.")
    if np.any(np.sum(assignment, axis=0) > core_count):
        raise RuntimeError("Assignment exceeds core capacity.")


def repair_assignment(
    a_round: np.ndarray,
    throughput: np.ndarray,
    power: np.ndarray,
    temperature: np.ndarray,
    throughput_requirement: float,
    core_count: np.ndarray,
    lambda_temp: float = 0.5,
    epsilon: float = 1e-9,
) -> np.ndarray:
    throughput, power, core_count = validate_problem(throughput, power, core_count)
    a = np.array(a_round, dtype=int, copy=True)
    temperature = np.asarray(temperature, dtype=float)

    if a.shape != throughput.shape:
        raise ValueError("a_round must have the same shape as throughput.")
    if temperature.ndim != 1 or temperature.shape[0] != a.shape[1]:
        raise ValueError("temperature must have one entry per state.")

    validate_assignment(a, core_count)

    n_threads, n_states = a.shape

    def current_state(thread_idx: int) -> int:
        states = np.where(a[thread_idx] == 1)[0]
        if states.size != 1:
            raise RuntimeError(f"Thread {thread_idx} does not have exactly one assignment.")
        return int(states[0])

    while assignment_throughput(a, throughput) < throughput_requirement:
        best_score = -np.inf
        best_operation = None
        used_capacity = np.sum(a, axis=0)

        for i in range(n_threads):
            old_j = current_state(i)
            for new_j in range(n_states):
                if new_j == old_j or used_capacity[new_j] >= core_count[new_j]:
                    continue
                delta_t = throughput[i, new_j] - throughput[i, old_j]
                if delta_t <= 0:
                    continue
                delta_p = power[i, new_j] - power[i, old_j]
                delta_temp = max(0.0, temperature[new_j] - temperature[old_j])
                score = delta_t / (max(delta_p, 0.0) + lambda_temp * delta_temp + epsilon)
                if score > best_score:
                    best_score = score
                    best_operation = ("promotion", i, old_j, new_j)

        for i in range(n_threads):
            old_i = current_state(i)
            for k in range(i + 1, n_threads):
                old_k = current_state(k)
                if old_i == old_k:
                    continue
                delta_t = (
                    throughput[i, old_k]
                    + throughput[k, old_i]
                    - throughput[i, old_i]
                    - throughput[k, old_k]
                )
                if delta_t <= 0:
                    continue
                delta_p = (
                    power[i, old_k]
                    + power[k, old_i]
                    - power[i, old_i]
                    - power[k, old_k]
                )
                delta_temp = max(
                    0.0,
                    abs(float(temperature[old_k]) - float(temperature[old_i])),
                )
                score = delta_t / (max(delta_p, 0.0) + lambda_temp * delta_temp + epsilon)
                if score > best_score:
                    best_score = score
                    best_operation = ("swap", i, k, old_i, old_k)

        if best_operation is None:
            break

        if best_operation[0] == "promotion":
            _, i, old_j, new_j = best_operation
            a[i, old_j] = 0
            a[i, new_j] = 1
        elif best_operation[0] == "swap":
            _, i, k, old_i, old_k = best_operation
            a[i, old_i] = 0
            a[i, old_k] = 1
            a[k, old_k] = 0
            a[k, old_i] = 1

    validate_assignment(a, core_count)

    if assignment_throughput(a, throughput) < throughput_requirement:
        raise RuntimeError("Repair failed: throughput requirement cannot be satisfied.")

    return a


def run_demo() -> None:
    throughput = np.array(
        [
            [10, 18, 30],
            [12, 17, 24],
            [8, 20, 31],
            [15, 19, 22],
            [9, 16, 28],
        ],
        dtype=float,
    )

    power = np.array(
        [
            [2, 4, 9],
            [2, 5, 8],
            [1, 4, 10],
            [3, 5, 7],
            [2, 4, 9],
        ],
        dtype=float,
    )

    temperature = np.array([55, 65, 78], dtype=float)
    core_count = np.array([2, 2, 1], dtype=int)
    throughput_requirement = 85.0

    a_lp = solve_lp_relaxation(
        throughput,
        power,
        throughput_requirement,
        core_count,
    )
    a_round = thermal_aware_capacity_rounding(
        a_lp,
        power,
        temperature,
        core_count,
    )
    a_final = repair_assignment(
        a_round,
        throughput,
        power,
        temperature,
        throughput_requirement,
        core_count,
    )

    np.set_printoptions(precision=3, suppress=True)

    print("Fractional LP assignment A_lp:")
    print(a_lp)
    print()

    print("Rounded assignment A_round:")
    print(a_round)
    print("Throughput:", assignment_throughput(a_round, throughput))
    print("Power:", assignment_power(a_round, power))
    print()

    print("Final assignment A_final:")
    print(a_final)
    print("Throughput:", assignment_throughput(a_final, throughput))
    print("Power:", assignment_power(a_final, power))
    print()

    print("Requirement:", throughput_requirement)


if __name__ == "__main__":
    run_demo()
