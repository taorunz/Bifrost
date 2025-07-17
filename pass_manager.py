# from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager import StagedPassManager

from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
# from qiskit.transpiler.passes.layout import NoiseAdaptiveLayout
# possibly needs to be handled manually with qiskit.transpiler.layout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager
from qiskit.transpiler.passes import ApplyLayout

from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements, Layout2qDistance, FullAncillaAllocation, EnlargeWithAncilla, SabreSwap
from qiskit.transpiler.passes import CheckMap

from pass_manager_config import CustomPassManagerConfig
from layout import CustomLayout
from routing import RandomBridgeSwap, RandomSabreSwap


# Main modifications are in Layout and Routing passes
# Adapted from level_0_pass_manager
# pass_manager_config: dict
# param1, param2
def custom_pass_manager(pass_manager_config: CustomPassManagerConfig) -> StagedPassManager:

    plugin_manager = PassManagerStagePluginManager()
    basis_gates = pass_manager_config.basis_gates
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    final_layout = pass_manager_config.final_layout
    init_method = pass_manager_config.init_method
    layout_method = pass_manager_config.layout_method or "dense"
    routing_method = pass_manager_config.routing_method or "stochastic"
    translation_method = pass_manager_config.translation_method or "translator"
    optimization_method = pass_manager_config.optimization_method
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    target = pass_manager_config.target
    hls_config = pass_manager_config.hls_config
    pad_ancilla = pass_manager_config.pad_ancilla
    bridging_factor = pass_manager_config.bridging_factor
    bridge_gate = pass_manager_config.bridge_gate

    # Initialize all the passes to None

    unroll_3q = None
    init = None
    layout=None
    routing=None
    translation=None
    pre_opt = None
    optimization = None
    sched=None

    # Layout Setup

    if target is None:
        coupling_map_layout = coupling_map
    else:
        coupling_map_layout = target

    _given_layout = SetLayout(initial_layout)

    _choose_layout_and_score = [TrivialLayout(coupling_map),
                                Layout2qDistance(coupling_map,
                                                 property_name='trivial_layout_score')]

    def _choose_layout_condition(property_set):
        return not property_set['layout']

    if layout_method == 'trivial':
        _improve_layout = TrivialLayout(coupling_map)
    elif layout_method == 'dense':
        _improve_layout = DenseLayout(coupling_map, backend_properties)
    # elif layout_method == 'noise_adaptive':
    #   _improve_layout = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == 'sabre':
        _improve_layout = SabreLayout(coupling_map, max_iterations=2, seed=seed_transpiler)

    def _not_perfect_yet(property_set):
        return property_set['trivial_layout_score'] is not None and \
               property_set['trivial_layout_score'] != 0
    
    # Routing Setup

    if routing_method == "none":
        routing_pm = None
    elif routing_method in {"random_bridge", "random_sabre", "random_sabre_old", "random_sabre_new"}:
        if routing_method == "random_bridge":
            routing_pass = RandomBridgeSwap(coupling_map, bridging_factor=bridging_factor, bridge_gate=bridge_gate)
        elif routing_method == "random_sabre":
            routing_pass = RandomSabreSwap(coupling_map, bridging_factor=bridging_factor, bridge_gate=bridge_gate)

        routing_pm = PassManager()
        if target is not None:
            routing_pm.append(CheckMap(target, property_set_field="routing_not_needed"))
        else:
            routing_pm.append(CheckMap(coupling_map, property_set_field="routing_not_needed"))

        def _swap_condition(property_set):
            return not property_set["routing_not_needed"]

        # if use_barrier_before_measurement:
        # routing_pm.append([BarrierBeforeFinalMeasurements(), routing_pass], condition=_swap_condition)
        # else:
        if _swap_condition:
            routing_pm.append([routing_pass])
    
    else:
        routing_pm = plugin_manager.get_passmanager_stage(
            "routing", routing_method, pass_manager_config, optimization_level=0
        )

    # Build pass manager

    if coupling_map or initial_layout:

        # Unroll

        unroll_3q = common.generate_unroll_3q(
            target,
            basis_gates,
            approximation_degree,
            unitary_synthesis_method,
            unitary_synthesis_plugin_config,
            hls_config,
        )

        # Set Layout
        if layout_method != "none":
            def _swap_mapped(property_set):
                return property_set["final_layout"] is None
            
            layout = PassManager()
            layout.append(_given_layout)
            if _choose_layout_condition:
                layout.append(_choose_layout_and_score)
            if _not_perfect_yet:
                layout.append(_improve_layout)
            # layout.append(_choose_layout_and_score, condition=_choose_layout_condition)
            # layout.append(_improve_layout, condition=_not_perfect_yet)

            if pad_ancilla:
                embed = PassManager([FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()])
                # embed = common.generate_embed_passmanager(coupling_map_layout)
            else:
                embed = PassManager([ApplyLayout()])
            
            # layout.append(
            #    [pass_ for x in embed.passes() for pass_ in x["passes"]], condition=_swap_mapped
            # )
            if _swap_mapped:
                layout.append(
                    [pass_ for x in embed._tasks for pass_ in x]
                )
        
        # Set Routing
        
        routing = routing_pm
    
    # Translation
    
    if translation_method == "none":
        pass
    elif translation_method not in {"translator", "synthesis", "unroller"}:
        translation = plugin_manager.get_passmanager_stage(
            "translation", translation_method, pass_manager_config, optimization_level=0
        )
    else:
        translation = common.generate_translation_passmanager(
            target,
            basis_gates,
            translation_method,
            approximation_degree,
            coupling_map,
            backend_properties,
            unitary_synthesis_method,
            unitary_synthesis_plugin_config,
            hls_config,
        )
    
    # Pre-Opt (For asymetric coupling maps)

    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        pre_opt = common.generate_pre_op_passmanager(target, coupling_map)
        pre_opt += translation
    else:
        pre_opt = None
    
    # Scheduling

    if scheduling_method == "none":
        pass
    elif scheduling_method is None or scheduling_method in {"alap", "asap"}:
        sched = common.generate_scheduling(
            instruction_durations, scheduling_method, timing_constraints, inst_map, target=target
        )
    else:
        sched = plugin_manager.get_passmanager_stage(
            "scheduling", scheduling_method, pass_manager_config, optimization_level=0
        )
    
    # Init
    init = common.generate_control_flow_options_check(
        layout_method=layout_method,
        routing_method=routing_method,
        translation_method=translation_method,
        optimization_method=optimization_method,
        scheduling_method=scheduling_method,
        basis_gates=basis_gates,
        target=target,
    )
    if init_method is not None:
        init += plugin_manager.get_passmanager_stage(
            "init", init_method, pass_manager_config, optimization_level=0
        )
    elif unroll_3q is not None:
        init += unroll_3q

    # Debugging

    # print()
    # print("unroll_3q:", unroll_3q)
    # print("init:", init)
    # print("layout:", layout)
    # print("routing:", routing)
    # print("translation:", translation)
    # print("pre_optimization", pre_opt)
    # print("optimization:", optimization)
    # print("scheduling:", sched)

    return StagedPassManager(
        init=init,
        layout=layout,
        routing=routing,
        translation=translation,
        pre_optimization=pre_opt,
        optimization=optimization,
        scheduling=sched,
    )

def padding_pass_manager(pass_manager_config: CustomPassManagerConfig) -> StagedPassManager:

    plugin_manager = PassManagerStagePluginManager()
    basis_gates = pass_manager_config.basis_gates
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    final_layout = pass_manager_config.final_layout
    init_method = pass_manager_config.init_method
    layout_method = pass_manager_config.layout_method or "trivial"
    routing_method = pass_manager_config.routing_method or "stochastic"
    translation_method = pass_manager_config.translation_method or "translator"
    optimization_method = pass_manager_config.optimization_method
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    target = pass_manager_config.target
    hls_config = pass_manager_config.hls_config
    pad_ancilla = pass_manager_config.pad_ancilla
    bridging_factor = pass_manager_config.bridging_factor
    bridge_gate = pass_manager_config.bridge_gate

    # Initialize all the passes to None

    unroll_3q = None
    init = None
    layout=None
    routing=None
    translation=None
    pre_opt = None
    optimization = None
    sched=None

    # Layout

    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
            return not property_set["layout"]

    if target is None:
        coupling_map_layout = coupling_map
    else:
        coupling_map_layout = target

    _choose_layout = TrivialLayout(coupling_map_layout)

    def _swap_mapped(property_set):
        return property_set["final_layout"] is None
    
    layout = PassManager()
    layout.append(_given_layout)
    if _choose_layout_condition:
        layout.append(_choose_layout)

    embed = common.generate_embed_passmanager(coupling_map_layout)
    
    #layout.append(
    #    [pass_ for x in embed.passes() for pass_ in x["passes"]], condition=_swap_mapped
    #)
    if _swap_mapped:
        layout.append(
            [pass_ for x in embed._tasks for pass_ in x]
        )

    # Scheduling

    if scheduling_method == "none":
        pass
    elif scheduling_method is None or scheduling_method in {"alap", "asap"}:
        sched = common.generate_scheduling(
            instruction_durations, scheduling_method, timing_constraints, inst_map, target=target
        )
    else:
        sched = plugin_manager.get_passmanager_stage(
            "scheduling", scheduling_method, pass_manager_config, optimization_level=0
        )

    # Init
    init = common.generate_control_flow_options_check(
        layout_method=layout_method,
        routing_method=routing_method,
        translation_method=translation_method,
        optimization_method=optimization_method,
        scheduling_method=scheduling_method,
        basis_gates=basis_gates,
        target=target,
    )
    if init_method is not None:
        init += plugin_manager.get_passmanager_stage(
            "init", init_method, pass_manager_config, optimization_level=0
        )
    elif unroll_3q is not None:
        init += unroll_3q

    return StagedPassManager(
        init=init,
        layout=layout,
        routing=routing,
        translation=translation,
        pre_optimization=pre_opt,
        optimization=optimization,
        scheduling=sched,
    )