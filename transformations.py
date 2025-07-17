import random

from qiskit.circuit.library import EvolvedOperatorAnsatz

from qiskit.compiler import transpile
from transpiler import modified_transpile

from utils import split_circuit_by_barrier

# Giving this the wrapper structure for consistency, though it isn't really necessary
def shuffle_ops():

    def _shuffle_ops(ansatz):
        ansatz = ansatz

        operators = ansatz.operators
        operators = random.sample(operators, len(operators))
        
        ansatz.operators = operators

        return ansatz
    
    return _shuffle_ops

def random_prune(num_pruned_ops):

    def _random_prune(ansatz):
        ansatz = ansatz

        operators = ansatz.operators
        operators = random.sample(operators, len(operators)-num_pruned_ops)

        ansatz.operators = operators

        return ansatz
    
    return _random_prune


def cnot_shuffle_in_op(backend):

    def _cnot_shuffle_in_op(ansatz):
        ansatz = ansatz

        operators = ansatz.operators
        new_ops = []
        for op in operators:
            # op_circ = transpile(EvolvedOperatorAnsatz(op), backend=backend, initial_layout=list(range(4)), optimization_level=0)
            op_circ = modified_transpile(EvolvedOperatorAnsatz(op), backend=backend, initial_layout=list(range(ansatz.num_qubits)))
            # print(op_circ)

            op_list = op_circ.data
            ids = []
            cx_ops = []
            for idx, op in enumerate(op_list):
                if op.operation.name == "cx":
                    ids.append(idx)
                    cx_ops.append(op)
            
            perm = random.sample(cx_ops, len(cx_ops))

            reordered_op = []
            cur_cnot_idx = 0
            for idx in range(len(op_list)):
                if idx in ids:
                    curop = perm[cur_cnot_idx]
                    cur_cnot_idx += 1
                else:
                    curop = op_list[idx]
                reordered_op.append(curop)
            
            op_circ.data = reordered_op

            new_ops.append(op_circ)

        ansatz2 = EvolvedOperatorAnsatz(new_ops, initial_state=ansatz.initial_state)

        return ansatz2
    
    return _cnot_shuffle_in_op

# Potential method to finish for remapping as a transformation instead of via the pass manager
def remap(input_layout, output_layout):

    def _remap(ansatz):
        ansatz = ansatz
        operators = ansatz.operators

        new_operators = []
        for op in operators:
            new_operators.append(op)
        
        ansatz._qubits
        
        ansatz.operators = new_operators

        return ansatz

    return _remap

def transpile_transform(**kwargs):
    
    def _transpile_transform(ansatz):
        return modified_transpile(ansatz, **kwargs)

    return _transpile_transform

def transpile_transform_barrier(**kwargs):
    
    def _transpile_transform(ansatz):

        subcircuits = split_circuit_by_barrier(ansatz)
        ansatz = transpile(subcircuits[0], backend=kwargs["backend"], optimization_level=0)
        ansatz.barrier()
        ansatz = ansatz.compose(modified_transpile(subcircuits[1], **kwargs))

        return ansatz

    return _transpile_transform

def qiskit_transpile_transform(**kwargs):
    
    def _transpile_transform(ansatz):
        return transpile(ansatz, **kwargs)

    return _transpile_transform

# Doesn't work :(
# Has higher error than normal qiskit transpile, I'm guessing due to routing mismatch
def qiskit_transpile_transform_barrier(**kwargs):
    def _transpile_transform(ansatz):

        subcircuits = split_circuit_by_barrier(ansatz)
        ansatz = transpile(subcircuits[0], **kwargs)
        ansatz.barrier()
        ansatz = ansatz.compose(transpile(subcircuits[1], **kwargs))

        return ansatz

    return _transpile_transform