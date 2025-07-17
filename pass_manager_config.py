import pprint

from qiskit.transpiler.passmanager_config import PassManagerConfig

# Adapted from PassManagerConfig
#MODIFICATION: Adding bridge_gate, final_layout, pad_ancilla, and bridging_factor
class CustomPassManagerConfig:
    """Pass Manager Configuration."""

    def __init__(
        self,
        initial_layout=None,
        final_layout=None,
        basis_gates=None,
        inst_map=None,
        coupling_map=None,
        layout_method=None,
        routing_method=None,
        translation_method=None,
        scheduling_method=None,
        instruction_durations=None,
        backend_properties=None,
        approximation_degree=None,
        seed_transpiler=None,
        timing_constraints=None,
        unitary_synthesis_method="default",
        unitary_synthesis_plugin_config=None,
        target=None,
        hls_config=None,
        init_method=None,
        optimization_method=None,
        pad_ancilla=None,
        bridging_factor=None,
        bridge_gate="cx",
    ):
        """Initialize a PassManagerConfig object

        Args:
            initial_layout (Layout): Initial position of virtual qubits on
                physical qubits.
            basis_gates (list): List of basis gate names to unroll to.
            inst_map (InstructionScheduleMap): Mapping object that maps gate to schedule.
            coupling_map (CouplingMap): Directed graph represented a coupling
                map.
            layout_method (str): the pass to use for choosing initial qubit
                placement. This will be the plugin name if an external layout stage
                plugin is being used.
            routing_method (str): the pass to use for routing qubits on the
                architecture. This will be a plugin name if an external routing stage
                plugin is being used.
            translation_method (str): the pass to use for translating gates to
                basis_gates. This will be a plugin name if an external translation stage
                plugin is being used.
            scheduling_method (str): the pass to use for scheduling instructions. This will
                be a plugin name if an external scheduling stage plugin is being used.
            instruction_durations (InstructionDurations): Dictionary of duration
                (in dt) for each instruction.
            backend_properties (BackendProperties): Properties returned by a
                backend, including information on gate errors, readout errors,
                qubit coherence times, etc.
            approximation_degree (float): heuristic dial used for circuit approximation
                (1.0=no approximation, 0.0=maximal approximation)
            seed_transpiler (int): Sets random seed for the stochastic parts of
                the transpiler.
            timing_constraints (TimingConstraints): Hardware time alignment restrictions.
            unitary_synthesis_method (str): The string method to use for the
                :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass. Will
                search installed plugins for a valid method. You can see a list of
                installed plugins with :func:`.unitary_synthesis_plugin_names`.
            target (Target): The backend target
            hls_config (HLSConfig): An optional configuration class to use for
                :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
                Specifies how to synthesize various high-level objects.
            init_method (str): The plugin name for the init stage plugin to use
            optimization_method (str): The plugin name for the optimization stage plugin
                to use.
        """
        self.initial_layout = initial_layout
        self.final_layout = final_layout
        self.basis_gates = basis_gates
        self.inst_map = inst_map
        self.coupling_map = coupling_map
        self.init_method = init_method
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.translation_method = translation_method
        self.optimization_method = optimization_method
        self.scheduling_method = scheduling_method
        self.instruction_durations = instruction_durations
        self.backend_properties = backend_properties
        self.approximation_degree = approximation_degree
        self.seed_transpiler = seed_transpiler
        self.timing_constraints = timing_constraints
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
        self.target = target
        self.hls_config = hls_config
        self.pad_ancilla = pad_ancilla
        self.bridging_factor = bridging_factor
        self.bridge_gate = bridge_gate

    #MODIFICATION:
    @classmethod
    def from_pass_manager_config(cls, pmc: PassManagerConfig):
        
        res = cls()

        res.initial_layout = pmc.initial_layout
        res.basis_gates = pmc.basis_gates
        res.inst_map = pmc.inst_map
        res.coupling_map = pmc.coupling_map
        res.init_method = pmc.init_method
        res.layout_method = pmc.layout_method
        res.routing_method = pmc.routing_method
        res.translation_method = pmc.translation_method
        res.optimization_method = pmc.optimization_method
        res.scheduling_method = pmc.scheduling_method
        res.instruction_durations = pmc.instruction_durations
        res.backend_properties = pmc.backend_properties
        res.approximation_degree = pmc.approximation_degree
        res.seed_transpiler = pmc.seed_transpiler
        res.timing_constraints = pmc.timing_constraints
        res.unitary_synthesis_method = pmc.unitary_synthesis_method
        res.unitary_synthesis_plugin_config = pmc.unitary_synthesis_plugin_config
        res.target = pmc.target
        res.hls_config = pmc.hls_config

        return res

    def __str__(self):
        newline = "\n"
        newline_tab = "\n\t"
        if self.backend_properties is not None:
            backend_props = pprint.pformat(self.backend_properties.to_dict())
            backend_props = backend_props.replace(newline, newline_tab)
        else:
            backend_props = str(None)
        return (
            "Pass Manager Config:\n"
            f"\tinitial_layout: {self.initial_layout}\n"
            f"\tfinal_layout: {self.final_layout}\n"
            f"\tbasis_gates: {self.basis_gates}\n"
            f"\tinst_map: {str(self.inst_map).replace(newline, newline_tab)}\n"
            f"\tcoupling_map: {self.coupling_map}\n"
            f"\tlayout_method: {self.layout_method}\n"
            f"\trouting_method: {self.routing_method}\n"
            f"\ttranslation_method: {self.translation_method}\n"
            f"\tscheduling_method: {self.scheduling_method}\n"
            f"\tinstruction_durations: {str(self.instruction_durations).replace(newline, newline_tab)}\n"
            f"\tbackend_properties: {backend_props}\n"
            f"\tapproximation_degree: {self.approximation_degree}\n"
            f"\tseed_transpiler: {self.seed_transpiler}\n"
            f"\ttiming_constraints: {self.timing_constraints}\n"
            f"\tunitary_synthesis_method: {self.unitary_synthesis_method}\n"
            f"\tunitary_synthesis_plugin_config: {self.unitary_synthesis_plugin_config}\n"
            f"\tpad_ancilla: {self.pad_ancilla}\n"
            f"\tbridging_factor: {self.bridging_factor}\n"
            f"\tbridge_gate: {self.bridge_gate}\n"
            f"\ttarget: {str(self.target).replace(newline, newline_tab)}\n"
        )
