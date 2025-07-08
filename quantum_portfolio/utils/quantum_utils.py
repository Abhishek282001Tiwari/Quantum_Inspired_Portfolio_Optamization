"""
Quantum computing utility functions.

This module provides utilities for quantum computing operations
including Pauli operators, circuit construction, and quantum-classical conversion.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import QFT, GroverOperator
from qiskit_optimization import QuadraticProgram
import itertools


def create_pauli_operator(
    qubo: Union[Dict, QuadraticProgram],
    offset: float = 0.0
) -> SparsePauliOp:
    """
    Create Pauli operator from QUBO problem.
    
    Args:
        qubo: QUBO problem dictionary or QuadraticProgram
        offset: Constant offset
        
    Returns:
        SparsePauliOp: Pauli operator representation
    """
    if isinstance(qubo, QuadraticProgram):
        # Extract QUBO coefficients from QuadraticProgram
        linear_coeffs = qubo.objective.linear.to_dict()
        quadratic_coeffs = qubo.objective.quadratic.to_dict()
        num_vars = qubo.get_num_vars()
    else:
        # Assume qubo is a dictionary with 'linear' and 'quadratic' keys
        linear_coeffs = qubo.get('linear', {})
        quadratic_coeffs = qubo.get('quadratic', {})
        num_vars = max(max(linear_coeffs.keys(), default=0), 
                      max(max(k) for k in quadratic_coeffs.keys() if k) + 1)
    
    pauli_list = []
    
    # Add constant offset
    if offset != 0:
        pauli_list.append(("I" * num_vars, offset))
    
    # Add linear terms
    for var, coeff in linear_coeffs.items():
        if coeff != 0:
            pauli_string = "I" * num_vars
            pauli_string = pauli_string[:var] + "Z" + pauli_string[var+1:]
            pauli_list.append((pauli_string, coeff))
    
    # Add quadratic terms
    for (var1, var2), coeff in quadratic_coeffs.items():
        if coeff != 0:
            pauli_string = "I" * num_vars
            pauli_string = pauli_string[:var1] + "Z" + pauli_string[var1+1:]
            pauli_string = pauli_string[:var2] + "Z" + pauli_string[var2+1:]
            pauli_list.append((pauli_string, coeff))
    
    return SparsePauliOp.from_list(pauli_list)


def ising_to_qubo(h: np.ndarray, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Ising model to QUBO formulation.
    
    Ising: min sum_i h_i * s_i + sum_ij J_ij * s_i * s_j
    where s_i ∈ {-1, +1}
    
    QUBO: min sum_i q_i * x_i + sum_ij Q_ij * x_i * x_j
    where x_i ∈ {0, 1}
    
    Transformation: s_i = 2*x_i - 1
    
    Args:
        h: Ising local fields
        J: Ising coupling matrix
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (q, Q) QUBO parameters
    """
    n = len(h)
    q = np.zeros(n)
    Q = np.zeros((n, n))
    
    # Transform linear terms
    for i in range(n):
        q[i] = 2 * h[i] - 4 * np.sum(J[i, :])
    
    # Transform quadratic terms
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = 4 * J[i, j]
    
    return q, Q


def qubo_to_ising(q: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert QUBO to Ising model formulation.
    
    Args:
        q: QUBO linear coefficients
        Q: QUBO quadratic matrix
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (h, J) Ising parameters
    """
    n = len(q)
    h = np.zeros(n)
    J = np.zeros((n, n))
    
    # Transform linear terms
    for i in range(n):
        h[i] = (q[i] + np.sum(Q[i, :])) / 2
    
    # Transform quadratic terms
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = Q[i, j] / 4
    
    return h, J


def encode_constraints(
    constraints: List[Dict],
    num_vars: int,
    penalty_strength: float = 10.0
) -> SparsePauliOp:
    """
    Encode constraints as Pauli operators.
    
    Args:
        constraints: List of constraint dictionaries
        num_vars: Number of variables
        penalty_strength: Penalty strength for constraint violations
        
    Returns:
        SparsePauliOp: Constraint penalty operator
    """
    constraint_ops = []
    
    for constraint in constraints:
        constraint_type = constraint.get('type', 'equality')
        coefficients = constraint.get('coefficients', [])
        rhs = constraint.get('rhs', 0)
        
        if constraint_type == 'equality':
            # Equality constraint: (sum_i c_i * x_i - rhs)^2
            op = create_equality_constraint_operator(coefficients, rhs, num_vars)
        elif constraint_type == 'inequality':
            # Inequality constraint: max(0, sum_i c_i * x_i - rhs)^2
            op = create_inequality_constraint_operator(coefficients, rhs, num_vars)
        else:
            continue
        
        constraint_ops.append(penalty_strength * op)
    
    if not constraint_ops:
        return SparsePauliOp.from_list([("I" * num_vars, 0.0)])
    
    # Sum all constraint operators
    total_constraint_op = constraint_ops[0]
    for op in constraint_ops[1:]:
        total_constraint_op += op
    
    return total_constraint_op


def create_equality_constraint_operator(
    coefficients: List[float],
    rhs: float,
    num_vars: int
) -> SparsePauliOp:
    """Create Pauli operator for equality constraint."""
    pauli_list = []
    
    # Constant term: rhs^2
    pauli_list.append(("I" * num_vars, rhs**2))
    
    # Linear terms: -2 * rhs * sum_i c_i * x_i
    for i, coeff in enumerate(coefficients):
        if coeff != 0:
            pauli_string = "I" * num_vars
            pauli_string = pauli_string[:i] + "Z" + pauli_string[i+1:]
            pauli_list.append((pauli_string, -rhs * coeff))
    
    # Quadratic terms: sum_i sum_j c_i * c_j * x_i * x_j
    for i, coeff_i in enumerate(coefficients):
        for j, coeff_j in enumerate(coefficients):
            if coeff_i != 0 and coeff_j != 0:
                if i == j:
                    # Diagonal term: c_i^2 * x_i (since x_i^2 = x_i for binary)
                    pauli_string = "I" * num_vars
                    pauli_string = pauli_string[:i] + "Z" + pauli_string[i+1:]
                    pauli_list.append((pauli_string, coeff_i * coeff_j))
                else:
                    # Off-diagonal term: c_i * c_j * x_i * x_j
                    pauli_string = "I" * num_vars
                    pauli_string = pauli_string[:i] + "Z" + pauli_string[i+1:]
                    pauli_string = pauli_string[:j] + "Z" + pauli_string[j+1:]
                    pauli_list.append((pauli_string, coeff_i * coeff_j))
    
    return SparsePauliOp.from_list(pauli_list)


def create_inequality_constraint_operator(
    coefficients: List[float],
    rhs: float,
    num_vars: int
) -> SparsePauliOp:
    """Create Pauli operator for inequality constraint."""
    # For now, treat as equality constraint
    # In practice, you would need slack variables
    return create_equality_constraint_operator(coefficients, rhs, num_vars)


def create_quantum_fourier_transform(num_qubits: int) -> QuantumCircuit:
    """
    Create Quantum Fourier Transform circuit.
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        QuantumCircuit: QFT circuit
    """
    qft_circuit = QFT(num_qubits)
    return qft_circuit


def create_grover_operator(
    oracle: QuantumCircuit,
    state_preparation: Optional[QuantumCircuit] = None
) -> GroverOperator:
    """
    Create Grover operator for quantum search.
    
    Args:
        oracle: Oracle circuit that marks target states
        state_preparation: State preparation circuit
        
    Returns:
        GroverOperator: Grover operator
    """
    if state_preparation is None:
        num_qubits = oracle.num_qubits
        state_preparation = QuantumCircuit(num_qubits)
        state_preparation.h(range(num_qubits))
    
    return GroverOperator(oracle, state_preparation)


def decompose_pauli_operator(operator: SparsePauliOp) -> List[Tuple[str, complex]]:
    """
    Decompose Pauli operator into individual Pauli strings.
    
    Args:
        operator: Pauli operator to decompose
        
    Returns:
        List[Tuple[str, complex]]: List of (Pauli string, coefficient) pairs
    """
    decomposition = []
    
    for pauli, coeff in zip(operator.paulis, operator.coeffs):
        pauli_str = str(pauli)
        decomposition.append((pauli_str, coeff))
    
    return decomposition


def pauli_expectation_value(
    pauli_string: str,
    state_vector: np.ndarray
) -> complex:
    """
    Calculate expectation value of Pauli operator.
    
    Args:
        pauli_string: Pauli string (e.g., "XIZY")
        state_vector: Quantum state vector
        
    Returns:
        complex: Expectation value
    """
    num_qubits = len(pauli_string)
    
    if len(state_vector) != 2**num_qubits:
        raise ValueError("State vector dimension must match number of qubits")
    
    # Convert Pauli string to matrix
    pauli_matrix = np.eye(1, dtype=complex)
    
    pauli_matrices = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    for pauli_char in pauli_string:
        pauli_matrix = np.kron(pauli_matrix, pauli_matrices[pauli_char])
    
    # Calculate expectation value
    expectation = np.conj(state_vector).T @ pauli_matrix @ state_vector
    
    return expectation


def quantum_state_tomography(
    measurement_results: Dict[str, int],
    num_qubits: int
) -> np.ndarray:
    """
    Perform quantum state tomography from measurement results.
    
    Args:
        measurement_results: Dictionary of measurement outcomes and counts
        num_qubits: Number of qubits
        
    Returns:
        np.ndarray: Reconstructed density matrix
    """
    dim = 2**num_qubits
    total_shots = sum(measurement_results.values())
    
    # Initialize density matrix
    rho = np.zeros((dim, dim), dtype=complex)
    
    # Reconstruct density matrix from measurement probabilities
    for outcome, count in measurement_results.items():
        probability = count / total_shots
        
        # Convert bit string to state index
        state_index = int(outcome, 2)
        
        # Add contribution to density matrix
        rho[state_index, state_index] += probability
    
    return rho


def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate fidelity between two quantum states.
    
    Args:
        state1: First quantum state (state vector or density matrix)
        state2: Second quantum state (state vector or density matrix)
        
    Returns:
        float: Fidelity between states
    """
    # Check if states are vectors or matrices
    if state1.ndim == 1 and state2.ndim == 1:
        # Both are state vectors
        return abs(np.dot(np.conj(state1), state2))**2
    elif state1.ndim == 2 and state2.ndim == 2:
        # Both are density matrices
        sqrt_rho1 = np.sqrt(state1)
        product = sqrt_rho1 @ state2 @ sqrt_rho1
        return np.trace(np.sqrt(product))**2
    else:
        raise ValueError("States must be both vectors or both matrices")


def quantum_volume(
    num_qubits: int,
    depth: int,
    success_threshold: float = 2/3
) -> Dict[str, float]:
    """
    Calculate quantum volume metrics.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        success_threshold: Success probability threshold
        
    Returns:
        Dict: Quantum volume metrics
    """
    # Simplified quantum volume calculation
    # In practice, this would involve running actual quantum volume circuits
    
    quantum_volume = min(num_qubits, depth)**2
    
    return {
        'quantum_volume': quantum_volume,
        'num_qubits': num_qubits,
        'depth': depth,
        'success_threshold': success_threshold,
        'log2_qv': np.log2(quantum_volume)
    }


def create_parameterized_circuit(
    num_qubits: int,
    num_layers: int,
    entanglement: str = 'linear'
) -> QuantumCircuit:
    """
    Create parameterized quantum circuit.
    
    Args:
        num_qubits: Number of qubits
        num_layers: Number of layers
        entanglement: Entanglement pattern
        
    Returns:
        QuantumCircuit: Parameterized circuit
    """
    from qiskit.circuit import Parameter
    
    qc = QuantumCircuit(num_qubits)
    
    # Parameters for rotation gates
    params = []
    
    for layer in range(num_layers):
        # Rotation gates
        for qubit in range(num_qubits):
            theta = Parameter(f'θ_{layer}_{qubit}')
            params.append(theta)
            qc.ry(theta, qubit)
        
        # Entangling gates
        if entanglement == 'linear':
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
        elif entanglement == 'circular':
            for qubit in range(num_qubits):
                qc.cx(qubit, (qubit + 1) % num_qubits)
        elif entanglement == 'full':
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qc.cx(i, j)
    
    return qc