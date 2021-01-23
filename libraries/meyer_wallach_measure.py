import numpy as np
import qutip

def compute_Q_ptrace(ket, N):
    """Computes Meyer-Wallach measure using alternative interpretation, i.e. as
    an average over the entanglements of each qubit with the rest of the system
    (see https://arxiv.org/pdf/quant-ph/0305094.pdf).
   
    Args:
    =====
    ket : numpy.ndarray or list
        Vector of amplitudes in 2**N dimensions
    N : int
        Number of qubits
​
    Returns:
    ========
    Q : float
        Q value for input ket
    """
    ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()
    print('KET=  ', ket)
    entanglement_sum = 0
    for k in range(N):
        print('value of n', k, 'PTrace: ',ket.ptrace([k])**2 )
        rho_k_sq = ket.ptrace([k])**2
        entanglement_sum += rho_k_sq.tr()  
   
    Q = 2*(1 - (1/N)*entanglement_sum)
    return Q

if __name__ == "__main__":

    # Test #1: bell state (Q should be 1)
    n_qubits = 2
    test_state = np.zeros(2**n_qubits)
    test_state[0] = 1
    test_state[-1] = 1
    test_state /= np.linalg.norm(test_state)
    print('test state:',test_state.shape)

    print('Test #1 (Q=1):')
    Q_value = compute_Q_ptrace(ket=test_state, N=n_qubits)
    print('Q = {}\n'.format(Q_value))

    # Test #2: product state (Q should be 0)
    n_qubits = 4
    test_state = np.zeros(2**n_qubits)
    test_state[0] = 1
    test_state[1] = 1
    test_state /= np.linalg.norm(test_state)

    print('Test #2 (Q=0):')
    Q_value = compute_Q_ptrace(ket=test_state, N=n_qubits)
    print('Q = {}\n'.format(Q_value))