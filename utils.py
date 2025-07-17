from collections import OrderedDict
from collections.abc import Sequence

import torch

# import torchquantum as tq
# from torchquantum.torchquantum.plugin import tq2qiskit

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.layout import TranspileLayout
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import Qubit
from qiskit.primitives.utils import init_observable
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes.routing import LayoutTransformation
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2, FakeSherbrooke, FakeKyiv

seed = 1000

#  -- Qiskit --

api_token = "8993ed4d23bd63e182ce559de04863f7aaa62376da8d293aa1a4f33a9c7366d719bd13f5948742e288728ef18a84638ec61d6cd38515ba2ae0cb1879650876bf"
api_token_2 = "58165d49a2f2bfd142d8377fbcbf459a49f2b95c2512371d35c2863387033ac69a93e6c585cdd010ecfe96dac6c981d28e0ca95383017cf1ed397f42fdd41826"
api_token_3 = "a5be4247068b1e6ae2096e9757c8c6cbce921676db4b5642a876c59f58c9b09f8652bd7162985a5c2bae1e03dd4f8c62e7dd38739c27d1a42083f82817afa673"
api_token_4 = "b5556cb4d937f8b249d73f72532c1949a2a3ddbe69cdf2639fbad8e66c64d5fd0ac691622797c0315dba9ca0c3f8c814a7a6482824c5f1263670172614ad7d35"
api_token_5 = "555e81c91abc1b272840b63301564b059e77f099283410edcdbb558ff6e1a8783c5f19af08f1edd73aa10baaea9a2f119881a375e62638123ca59c194d66a780"
api_token_6 = "6447ae2e2063725229d7ed63d85206f969d8e35cef3bf8a5bc9e8c1bd0c58146b64c9fc557a34986833f5bd036f96c48a9750210a7f1245d057161af24d673a3"

api_tokens = [api_token, api_token_2, api_token_3, api_token_4, api_token_5, api_token_6]

def qiskit_backend(backend_name, computer):
    if computer == "fake":
        if backend_name == "ibm_guadalupe":
            return FakeGuadalupeV2()
        if backend_name == "ibm_kyiv":
            return FakeKyiv()
        elif backend_name == "ibm_sherbrooke" or "ibm_kyoto":
            #Kyoto and Sherbrooke have the same shape
            return FakeSherbrooke()
        return None
    else:
        service = qiskit_service(computer)
        backend = service.backend(backend_name)

        return backend

def qiskit_service(computer):
    service = QiskitRuntimeService(channel="ibm_quantum", token = api_tokens[int(computer)-1])
    
    return service

def molecule_driver(benchmark):
    r = 1.5

    driver = None
    if benchmark == "h2":
        driver = PySCFDriver(f"H 0.0 0.0 0.0; H 0.0 0.0 {r}")
    elif benchmark == "h4":
        driver = PySCFDriver(f"H 0.0 0.0 0.0; H 0.0 0.0 {r}; H 0.0 0.0 {2*r}; H 0.0 0.0 {3*r}")
    elif benchmark == "h6":
        driver = PySCFDriver(f"H 0.0 0.0 0.0; H 0.0 0.0 {r}; H 0.0 0.0 {2*r}; H 0.0 0.0 {3*r}; H 0.0 0.0 {4*r}; H 0.0 0.0 {5*r}") # 400 GB of memory
    elif benchmark == "lih":
        driver = PySCFDriver(f"Li 0.0 0.0 0.0; H 0.0 0.0 {r}") # 196 GB of memory [OrderedDict([('cx', 20939), ('rz', 13120), ('sx', 4992), ('x', 4)])]
    elif benchmark == "beh2":
        driver = PySCFDriver(f"Be 0.0 0.0 0.0; H 0.0 0.0 {r}; H 0.0 0.0 {2*r}") # 3 TB??? of memory
    
    return driver

# From PR posted in Qiskit Slack
def transpile_operator(
    operator: BaseOperator | Pauli | str,
    layout: TranspileLayout,
    original_qubits: Sequence[Qubit],
) -> SparsePauliOp:
    """Utility function for transpilation of operator.
    If skip_transpilation is True, users need to transpile the operator corresponding to the layout
    of the transpiled circuit. This function helps the transpilation of operator.
    Args:
       operator: Operator to be transpiled.
       layout: The layout of the transpiled circuit.
       original_qubits: Qubits that original circuit has.
    Returns:
        The operator for the given layout.
    """
    operator = init_observable(operator)
    virtual_bit_map = layout.initial_layout.get_virtual_bits()
    identity = SparsePauliOp("I" * len(virtual_bit_map))
    perm_pattern = [virtual_bit_map[v] for v in original_qubits]
    if layout.final_layout is not None:
        final_mapping = dict(enumerate(layout.final_layout.get_virtual_bits().values()))
        perm_pattern = [final_mapping[i] for i in perm_pattern]
    return identity.compose(operator, qargs=perm_pattern)

# From StackOverflow
def remove_idle_qwires(circ):
    dag = circuit_to_dag(circ)

    idle_wires = list(dag.idle_wires())

    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)

    dag.qregs = OrderedDict()

    return dag_to_circuit(dag)

def split_circuit_by_barrier(circuit):
    inst_lists = []
    cur_list = []
    for gate in circuit:
        if gate.operation.name == "barrier":
            inst_lists.append(cur_list)
            cur_list = []
        else:
            cur_list.append(gate)
    inst_lists.append(cur_list)
    
    out_circs = []
    for inst_list in inst_lists:
        qc = QuantumCircuit.from_instructions(inst_list, qubits=circuit._qubits, clbits=circuit._clbits)
        out_circs.append(qc)
    
    return out_circs

def change_mapping(qc, from_layout, to_layout, coupling_map):
    
    transpile_layout = None
    if (qc.layout):
        transpile_layout = qc.layout
        transpile_layout.final_layout = to_layout

    coupling_map_obj = CouplingMap(coupling_map)

    pm = PassManager()
    pm.append([LayoutTransformation(from_layout=from_layout, to_layout=to_layout, coupling_map=coupling_map_obj)])

    new_qc = pm.run(qc)
    
    return new_qc, transpile_layout

#  -- TorchQuantum

# torch_device_name = "cuda" if torch.cuda.is_available() else "cpu"
torch_device_name = "cpu"
torch_device = torch.device(torch_device_name)

def example_accuracy(preds, labels):
    _, indices = preds.topk(1, dim=1)
    masks = indices.eq(labels.view(-1, 1).expand_as(indices))
    corrects = masks.sum().item()

    size = labels.shape[0]
    accuracy = corrects / size

    return accuracy

# -- UNCOMMENT THESE AFTER TORCHQUANTUM INSTALL --

# def qiskit2tq_Operator(circ: QuantumCircuit):
#     if getattr(circ, "_layout", None) is not None:
#         try:
#             p2v_orig = circ._layout.final_layout.get_physical_bits().copy()
#         except:
#             try:
#                 p2v_orig = circ._layout.get_physical_bits().copy()
#             except:
#                 p2v_orig = circ._layout.initial_layout.get_physical_bits().copy() #MODIFIED: Using initial_layout in case final_layout isn't set
#         p2v = {}
#         for p, v in p2v_orig.items():
#             if v.register.name == "q":
#                 p2v[p] = v.index
#             else:
#                 p2v[p] = f"{v.register.name}.{v.index}"
#     else:
#         p2v = {}
#         for p in range(circ.num_qubits):
#             p2v[p] = p

#     ops = []
#     for gate in circ.data:
#         op_name = gate[0].name
#         wires = list(map(lambda x: x.index, gate[1]))
#         wires = [p2v[wire] for wire in wires]
#         # sometimes the gate.params is ParameterExpression class
#         init_params = (
#             list(map(float, gate[0].params)) if len(gate[0].params) > 0 else None
#         )

#         if op_name in [
#             "h",
#             "x",
#             "y",
#             "z",
#             "s",
#             "t",
#             "sx",
#             "cx",
#             "cz",
#             "cy",
#             "swap",
#             "cswap",
#             "ccx",
#         ]:
#             ops.append(tq.op_name_dict[op_name](wires=wires))
#         elif op_name in [
#             "rx",
#             "ry",
#             "rz",
#             "rxx",
#             "xx",
#             "ryy",
#             "yy",
#             "rzz",
#             "zz",
#             "rzx",
#             "zx",
#             "p",
#             "cp",
#             "crx",
#             "cry",
#             "crz",
#             "u1",
#             "cu1",
#             "u2",
#             "u3",
#             "cu3",
#             "u",
#             "cu",
#         ]:
#             ops.append(
#                 tq.op_name_dict[op_name](
#                     has_params=True,
#                     trainable=True,
#                     init_params=init_params,
#                     wires=wires,
#                 )
#             )
#         elif op_name in ["barrier", "measure"]:
#             continue
#         else:
#             raise NotImplementedError(
#                 f"{op_name} conversion to tq is currently not supported."
#             )
#     return ops


# def qiskit2tq(circ: QuantumCircuit):
#     ops = qiskit2tq_Operator(circ)
#     return tq.QuantumModuleFromOps(ops)