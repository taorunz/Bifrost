import logging
from copy import deepcopy
import time

import rustworkx

from qiskit.circuit import SwitchCaseOp, ControlFlowOp, Clbit, ClassicalRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.controlflow import condition_resources, node_resources
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.dagcircuit import DAGCircuit
# from qiskit.tools.parallel import CPU_COUNT
from qiskit.utils.parallel import CPU_COUNT

import random

import rustworkx

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.library import CXGate, DCXGate, iSwapGate, CZGate
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils

from qiskit.visualization import dag_drawer

import logging
from copy import copy, deepcopy

from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit import DAGOpNode
# from qiskit.tools.parallel import CPU_COUNT

from collections import defaultdict
import numpy as np

from qiskit.circuit.quantumregister import Qubit

logger = logging.getLogger(__name__)

from qiskit.transpiler.passes.routing.sabre_swap import (
    _build_sabre_dag,
    Heuristic,
    NeighborTable,
    SabreDAG
)
'''
from qiskit._accelerate.sabre_swap import (
    build_swap_map,
    Heuristic,
    NeighborTable,
    SabreDAG,
)'
'''
from qiskit._accelerate.nlayout import NLayout

#Adapted from BasicSwap
class RandomBridgeSwap(TransformationPass):
    """Map (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates.

    The basic mapper is a minimum effort to insert swap gates to map the DAG onto
    a coupling map. When a cx is not in the coupling map possibilities, it inserts
    one or more swaps in front to make it compatible.
    """

    #MODIFICATION: Add bridging_factor, briding_factor = -1 allows for random factor on each run
    def __init__(self, coupling_map, bridging_factor, bridge_gate, fake_run=False):
        """BasicSwap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): Directed graph represented a coupling map.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        
        self.bridge_gate = bridge_gate
        
        if bridging_factor == -1:
            self.bridging_factor = random.random()
        else:
            self.bridging_factor = bridging_factor
        
        self.fake_run = fake_run
    
    def run(self, dag):
        """Run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None.
        """
        if self.fake_run:
            return self._fake_run(dag)

        new_dag = dag.copy_empty_like()

        if self.coupling_map is None:
            raise TranspilerError("BasicSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Basic swap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")
        disjoint_utils.require_layout_isolated_to_component(
            dag, self.coupling_map if self.target is None else self.target
        )
        canonical_register = dag.qregs["q"]
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        bridge_count = 0
        for layer in dag.serial_layers():
            subdag = layer["graph"]

            for gate in subdag.two_qubit_ops():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(canonical_register)

                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    kept_swaps = []
                    for swap in range(len(path) - 2):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap + 1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        #MODIFICATION: Apply CX instead of swap when bridging applied
                        rng = random.random()

                        # create the swap operation
                        if rng > self.bridging_factor:
                            swap_layer.apply_operation_back(
                                SwapGate(), qargs=[qubit_1, qubit_2], cargs=[], check=False
                            )
                        else:
                            if self.bridge_gate == "cx":
                                swap_layer.apply_operation_back(
                                    CXGate(), qargs=[qubit_1, qubit_2], cargs=[]
                                )
                            elif self.bridge_gate == "dcx":
                                swap_layer.apply_operation_back(
                                    DCXGate(), qargs=[qubit_1, qubit_2], cargs=[]
                                )
                            elif self.bridge_gate == "iswap":
                                swap_layer.apply_operation_back(
                                    iSwapGate(), qargs=[qubit_1, qubit_2], cargs=[]
                                )
                            elif self.bridge_gate == "cz":
                                swap_layer.apply_operation_back(
                                    CZGate(), qargs=[qubit_1, qubit_2], cargs=[]
                                )
                            bridge_count += 1
                        kept_swaps.append(swap)              

                    # layer insertion
                    order = current_layout.reorder_bits(new_dag.qubits)
                    new_dag.compose(swap_layer, qubits=order)

                    for swap in kept_swaps:
                        current_layout.swap(path[swap], path[swap + 1])

            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)

        self.property_set["final_layout"] = current_layout

        # print("Bridges:", bridge_count)
        # print()
        # print()
        # print()
        # print()
        return new_dag


    def _fake_run(self, dag):
        """Do a fake run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to improve initial layout.

        Returns:
            DAGCircuit: The same DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Basic swap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")

        canonical_register = dag.qregs["q"]
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer["graph"]
            for gate in subdag.two_qubit_ops():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    # update current_layout
                    for swap in range(len(path) - 2):
                        # create the swap operation
                        current_layout.swap(path[swap], path[swap + 1])

        self.property_set["final_layout"] = current_layout
        return dag

logger = logging.getLogger(__name__)

class RandomSabreSwap(TransformationPass):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of the SWAP-based heuristic search from the SABRE qubit
    mapping paper [1] (Algorithm 1). The heuristic aims to minimize the number
    of lossy SWAPs inserted and the depth of the circuit.

    This algorithm starts from an initial layout of virtual qubits onto physical
    qubits, and iterates over the circuit DAG until all gates are exhausted,
    inserting SWAPs along the way. It only considers 2-qubit gates as only those
    are germane for the mapping problem (it is assumed that 3+ qubit gates are
    already decomposed).

    In each iteration, it will first check if there are any gates in the
    ``front_layer`` that can be directly applied. If so, it will apply them and
    remove them from ``front_layer``, and replenish that layer with new gates
    if possible. Otherwise, it will try to search for SWAPs, insert the SWAPs,
    and update the mapping.

    The search for SWAPs is restricted, in the sense that we only consider
    physical qubits in the neighborhood of those qubits involved in
    ``front_layer``. These give rise to a ``swap_candidate_list`` which is
    scored according to some heuristic cost function. The best SWAP is
    implemented and ``current_layout`` updated.

    This transpiler pass adds onto the SABRE algorithm in that it will run
    multiple trials of the algorithm with different seeds. The best output,
    determined by the trial with the least amount of SWAPed inserted, will
    be selected from the random trials.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(self, coupling_map, bridging_factor, bridge_gate, heuristic="basic", seed=None, fake_run=False, trials=None):
        r"""SabreSwap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'lookahead' or 'decay').
            seed (int): random seed used to tie-break among candidate swaps.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
            trials (int): The number of seed trials to run sabre with. These will
                be run in parallel (unless the PassManager is already running in
                parallel). If not specified this defaults to the number of physical
                CPUs on the local system. For reproducible results it is recommended
                that you set this explicitly, as the output will be deterministic for
                a fixed number of trials.

        Raises:
            TranspilerError: If the specified heuristic is not valid.

        Additional Information:

            The search space of possible SWAPs on physical qubits is explored
            by assigning a score to the layout that would result from each SWAP.
            The goodness of a layout is evaluated based on how viable it makes
            the remaining virtual gates that must be applied. A few heuristic
            cost functions are supported

            - 'basic':

            The sum of distances for corresponding physical qubits of
            interacting virtual qubits in the front_layer.

            .. math::

                H_{basic} = \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'lookahead':

            This is the sum of two costs: first is the same as the basic cost.
            Second is the basic cost but now evaluated for the
            extended set as well (i.e. :math:`|E|` number of upcoming successors to gates in
            front_layer F). This is weighted by some amount EXTENDED_SET_WEIGHT (W) to
            signify that upcoming gates are less important that the front_layer.

            .. math::

                H_{decay}=\frac{1}{\left|{F}\right|}\sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]
                    + W*\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'decay':

            This is the same as 'lookahead', but the whole cost is multiplied by a
            decay factor. This increases the cost if the SWAP that generated the
            trial layout was recently used (i.e. it penalizes increase in depth).

            .. math::

                H_{decay} = max(decay(SWAP.q_1), decay(SWAP.q_2)) {
                    \frac{1}{\left|{F}\right|} \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]\\
                    + W *\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]
                    }
        """

        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.coupling_map = coupling_map
            self.target = None
        if self.coupling_map is not None and not self.coupling_map.is_symmetric:
            # A deepcopy is needed here if we don't own the coupling map (i.e. we were given it,
            # rather than calculated it from the Target), to avoid modifications updating shared
            # references in passes which require directional constraints.
            if isinstance(coupling_map, CouplingMap):
                self.coupling_map = deepcopy(self.coupling_map)
            self.coupling_map.make_symmetric()
        self._neighbor_table = None
        if self.coupling_map is not None:
            self._neighbor_table = NeighborTable(
                rustworkx.adjacency_matrix(self.coupling_map.graph)
            )

        self.heuristic = heuristic
        self.seed = seed
        if trials is None:
            self.trials = CPU_COUNT
        else:
            self.trials = trials

        self.fake_run = fake_run
        self._qubit_indices = None
        self._clbit_indices = None
        self.dist_matrix = None

        self.bridge_gate = bridge_gate
        
        if bridging_factor == -1:
            self.bridging_factor = random.random()
        else:
            self.bridging_factor = bridging_factor

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None
        """

        if self.coupling_map is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        num_dag_qubits = len(dag.qubits)
        num_coupling_qubits = self.coupling_map.size()
        if num_dag_qubits < num_coupling_qubits:
            raise TranspilerError(
                f"Fewer qubits in the circuit ({num_dag_qubits}) than the coupling map"
                f" ({num_coupling_qubits})."
                " Have you run a layout pass and then expanded your DAG with ancillas?"
                " See `FullAncillaAllocation`, `EnlargeWithAncilla` and `ApplyLayout`."
            )
        if num_dag_qubits > num_coupling_qubits:
            raise TranspilerError(
                f"More qubits in the circuit ({num_dag_qubits}) than available in the coupling map"
                f" ({num_coupling_qubits})."
                " This circuit cannot be routed to this device."
            )

        if self.heuristic == "basic":
            heuristic = Heuristic.Basic
        elif self.heuristic == "lookahead":
            heuristic = Heuristic.Lookahead
        elif self.heuristic == "decay":
            heuristic = Heuristic.Decay
        else:
            raise TranspilerError("Heuristic %s not recognized." % self.heuristic)
        disjoint_utils.require_layout_isolated_to_component(
            dag, self.coupling_map if self.target is None else self.target
        )

        self.dist_matrix = self.coupling_map.distance_matrix

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        self._qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
        layout_mapping = {
            self._qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()
        }
        initial_layout = NLayout(layout_mapping, len(dag.qubits), self.coupling_map.size())

        sabre_dag, circuit_to_dag_dict = _build_sabre_dag(
            dag,
            self.coupling_map.size(),
            self._qubit_indices,
        )
        sabre_start = time.perf_counter()
        *sabre_result, final_permutation = _build_sabre_dag(
            len(dag.qubits),
            sabre_dag,
            self._neighbor_table,
            self.dist_matrix,
            heuristic,
            initial_layout,
            self.trials,
            self.seed,
        )
        sabre_stop = time.perf_counter()
        logging.debug("Sabre swap algorithm execution complete in: %s", sabre_stop - sabre_start)

        self.property_set["final_layout"] = Layout(dict(zip(dag.qubits, final_permutation)))
        if self.fake_run:
            return dag
        return _apply_sabre_result(
            dag.copy_empty_like(),
            dag,
            sabre_result,
            initial_layout,
            dag.qubits,
            circuit_to_dag_dict,
            self.bridge_gate,
            self.bridging_factor
        )


def _build_sabre_dag(dag, num_physical_qubits, qubit_indices):
    from qiskit.converters import circuit_to_dag

    # Maps id(block): circuit_to_dag(block) for all descendant blocks
    circuit_to_dag_dict = {}

    def recurse(block, block_qubit_indices):
        block_id = id(block)
        if block_id in circuit_to_dag_dict:
            block_dag = circuit_to_dag_dict[block_id]
        else:
            block_dag = circuit_to_dag(block)
            circuit_to_dag_dict[block_id] = block_dag
        return process_dag(block_dag, block_qubit_indices)

    def process_dag(block_dag, wire_map):
        dag_list = []
        node_blocks = {}
        for node in block_dag.topological_op_nodes():
            cargs_bits = set(node.cargs)
            if node.op.condition is not None:
                cargs_bits.update(condition_resources(node.op.condition).clbits)
            if isinstance(node.op, SwitchCaseOp):
                target = node.op.target
                if isinstance(target, Clbit):
                    cargs_bits.add(target)
                elif isinstance(target, ClassicalRegister):
                    cargs_bits.update(target)
                else:  # Expr
                    cargs_bits.update(node_resources(target).clbits)
            cargs = {block_dag.find_bit(x).index for x in cargs_bits}
            if isinstance(node.op, ControlFlowOp):
                node_blocks[node._node_id] = [
                    recurse(
                        block,
                        {inner: wire_map[outer] for inner, outer in zip(block.qubits, node.qargs)},
                    )
                    for block in node.op.blocks
                ]
            dag_list.append(
                (
                    node._node_id,
                    [wire_map[x] for x in node.qargs],
                    cargs,
                )
            )
        return SabreDAG(num_physical_qubits, block_dag.num_clbits(), dag_list, node_blocks)

    return process_dag(dag, qubit_indices), circuit_to_dag_dict


#MODIFIED: Add bridging
def _apply_sabre_result(
    out_dag,
    in_dag,
    sabre_result,
    initial_layout,
    physical_qubits,
    circuit_to_dag_dict,
    bridge_gate,
    bridging_factor
):
    """Apply the ``SabreResult`` to ``out_dag``, mutating it in place.  This function in effect
    performs the :class:`.ApplyLayout` transpiler pass with ``initial_layout`` and the Sabre routing
    simultaneously, though it assumes that ``out_dag`` has already been prepared as containing the
    right physical qubits.

    Mutates ``out_dag`` in place and returns it.  Mutates ``initial_layout`` in place as scratch
    space.

    Args:
        out_dag (DAGCircuit): the physical DAG that the output should be written to.
        in_dag (DAGCircuit): the source of the nodes that are being routed.
        sabre_result (tuple[SwapMap, Sequence[int], NodeBlockResults]): the result object from the
            Rust run of the Sabre routing algorithm.
        initial_layout (NLayout): a Rust-space mapping of virtual indices (i.e. those of the qubits
            in ``in_dag``) to physical ones.
        physical_qubits (list[Qubit]): an indexable sequence of :class:`.Qubit` objects representing
            the physical qubits of the circuit.  Note that disjoint-coupling handling can mean that
            these are not strictly a "canonical physical register" in order.
        circuit_to_dag_dict (Mapping[int, DAGCircuit]): a mapping of the Python object identity
            (as returned by :func:`id`) of a control-flow block :class:`.QuantumCircuit` to a
            :class:`.DAGCircuit` that represents the same thing.
    """

    # The swap gate is a singleton instance, so we don't need to waste time reconstructing it each
    # time we need to use it.
    swap_singleton = SwapGate()

    #MODIFIED: Add alternative gate options
    if bridge_gate == "cx":
        alt_singleton = CXGate()
    elif bridge_gate == "dcx":
        alt_singleton = DCXGate()
    elif bridge_gate == "iswap":
        alt_singleton = iSwapGate()
    elif bridge_gate == "cz":
        alt_singleton = CZGate()

    def empty_dag(block):
        empty = DAGCircuit()
        empty.add_qubits(out_dag.qubits)
        for qreg in out_dag.qregs.values():
            empty.add_qreg(qreg)
        empty.add_clbits(block.clbits)
        for creg in block.cregs:
            empty.add_creg(creg)
        empty._global_phase = block.global_phase
        return empty

    #MODIFIED: Add bridge gate instead of swap based on rng
    def apply_swaps(dest_dag, swaps, layout):
        for a, b in swaps:
            qubits = (physical_qubits[a], physical_qubits[b])
            layout.swap_physical(a, b)

            rng = random.random()
            if rng > bridging_factor:
                dest_dag.apply_operation_back(swap_singleton, qubits, (), check=False)
            else:
                dest_dag.apply_operation_back(alt_singleton, qubits, (), check=False)

    def recurse(dest_dag, source_dag, result, root_logical_map, layout):
        """The main recursive worker.  Mutates ``dest_dag`` and ``layout`` and returns them.

        ``root_virtual_map`` is a mapping of the (virtual) qubit in ``source_dag`` to the index of
        the virtual qubit in the root source DAG that it is bound to."""
        swap_map, node_order, node_block_results = result
        for node_id in node_order:
            node = source_dag._multi_graph[node_id]
            if node_id in swap_map:
                apply_swaps(dest_dag, swap_map[node_id], layout)
            if not isinstance(node.op, ControlFlowOp):
                dest_dag.apply_operation_back(
                    node.op,
                    [
                        physical_qubits[layout.virtual_to_physical(root_logical_map[q])]
                        for q in node.qargs
                    ],
                    node.cargs,
                    check=False,
                )
                continue

            # At this point, we have to handle a control-flow node.
            block_results = node_block_results[node_id]
            mapped_block_dags = []
            idle_qubits = set(dest_dag.qubits)
            for block, block_result in zip(node.op.blocks, block_results):
                block_root_logical_map = {
                    inner: root_logical_map[outer] for inner, outer in zip(block.qubits, node.qargs)
                }
                block_dag, block_layout = recurse(
                    empty_dag(block),
                    circuit_to_dag_dict[id(block)],
                    (
                        block_result.result.map,
                        block_result.result.node_order,
                        block_result.result.node_block_results,
                    ),
                    block_root_logical_map,
                    layout.copy(),
                )
                apply_swaps(block_dag, block_result.swap_epilogue, block_layout)
                mapped_block_dags.append(block_dag)
                idle_qubits.intersection_update(block_dag.idle_wires())

            mapped_blocks = []
            for mapped_block_dag in mapped_block_dags:
                # Remove wires that are idle in all blocks.
                mapped_block_dag.remove_qubits(*idle_qubits)
                mapped_blocks.append(dag_to_circuit(mapped_block_dag))

            # Apply the control flow gate to the dag.
            mapped_node = node.op.replace_blocks(mapped_blocks)
            mapped_node_qargs = mapped_blocks[0].qubits if mapped_blocks else ()
            dest_dag.apply_operation_back(mapped_node, mapped_node_qargs, node.cargs, check=False)
        return dest_dag, layout

    root_logical_map = {bit: index for index, bit in enumerate(in_dag.qubits)}
    return recurse(out_dag, in_dag, sabre_result, root_logical_map, initial_layout)[0]