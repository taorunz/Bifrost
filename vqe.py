import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.optimize import minimize

from qiskit import qpy
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import EstimatorOptions as RuntimeEstimatorOptions
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from qiskit.transpiler.layout import Layout
from qiskit.circuit import QuantumRegister
from qiskit.compiler import transpile
from qiskit_ibm_runtime import Session
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.providers import JobV1

from transformations import transpile_transform, qiskit_transpile_transform
from utils import transpile_operator, split_circuit_by_barrier, change_mapping, qiskit_backend, qiskit_service, molecule_driver, seed
from transpiler import modified_transpile

random.seed(seed)

def get_estimators(backend, 
                   training_real_quantum_computer=False, 
                   testing_real_quantum_computer=False, 
                   approximation=True, 
                   skip_transpilation=False):
    
    runtimeoptions = RuntimeEstimatorOptions(default_shots = 1000)

    train_session = None
    test_session = None

    # FOR NOW: Using deprecated ESTIMATORV1 for training, under AerEstimator

    if training_real_quantum_computer:
        train_session = Session(backend=backend)
        training_estimator = RuntimeEstimator(mode = train_session, options=runtimeoptions)
    else:
        training_estimator = AerEstimator(approximation=approximation, skip_transpilation=skip_transpilation, run_options = {"shots": 1000})
    
    if testing_real_quantum_computer:
        test_session = Session(backend=backend)
        testing_estimator = RuntimeEstimator(mode = test_session, options=runtimeoptions)
    else:
        testing_estimator = AerEstimator(approximation=approximation, skip_transpilation=skip_transpilation, run_options = {"shots": 1000})

    return training_estimator, testing_estimator, train_session, test_session

def get_real_energy(hamiltonian, print_energy=True):

    real_solver = NumPyMinimumEigensolver()
    real_result = real_solver.compute_minimum_eigenvalue(hamiltonian)
    real_energy = real_result.eigenvalue

    if print_energy:
        print()
        print("--Real Eigensolver--")
        print("Real Energy:", real_energy)
        print()

    return real_energy

def run_real_qc(qcirc, parameters, hamiltonian, estimator):
    batch_size = len(parameters)
    job = estimator.run(batch_size * [qcirc], batch_size * [hamiltonian], parameters)
    job_id = job.job_id()

def cost_func(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    if isinstance(estimator, AerEstimator):
        result = estimator.run(ansatz, hamiltonian, params).result()
        energy = result.values[0]
    else:
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]

    return energy

def VQE(ansatz, hamiltonian, estimator):
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)

    res = minimize(
        cost_func,
        x0,
        args=(ansatz, hamiltonian, estimator),
        method="slsqp",
    )

    return res

def experiment(backend, 
               driver, 
               service,
               transformations,
               ansatz_type,
               ansatz_reps=1,
               training_real_quantum_computer=False, 
               testing_real_quantum_computer=False, 
               skip_transpilation=True, 
               runs=1, 
               run_repeats=1,
               real_qc_runs=1,
               runtime_dir="runtime/vqe/", 
               summary_only=False,
               restore_mapping=False,
               save_threshold=0,
               plot=True):

    # Qiskit Setup

    problem = driver.run()

    mapper = JordanWignerMapper()

    hamiltonian = mapper.map(problem.second_q_ops()[0])

    real_energy = get_real_energy(hamiltonian)

    training_estimator, testing_estimator, train_session, test_session = get_estimators(backend, training_real_quantum_computer, testing_real_quantum_computer, approximation=True, skip_transpilation=skip_transpilation)

    # VQE

    jobs = []
    for _ in tqdm(range(runs), position=0, desc="run"):

        hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
        hf.barrier()

        ansatz = hf

        if ansatz_type == "uccsd":
            ansatz = ansatz.compose(UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper))
        elif ansatz_type == "hardware":
            ansatz = ansatz.compose(EfficientSU2(hamiltonian.num_qubits, entanglement="circular", reps=ansatz_reps))
        elif ansatz_type == "twolocal":
            ansatz = ansatz.compose(TwoLocal(hamiltonian.num_qubits, 'ry', 'cz', entanglement="full", reps=ansatz_reps))
        else:
            print("Ansatz not implemented")
            return

        qubits = ansatz.qubits

        for transform in transformations:
            ansatz = transform(ansatz)
        
        if transformations:
            if restore_mapping:
                init_layout = Layout.from_intlist(list(range(ansatz.num_qubits)), QuantumRegister(ansatz.num_qubits, name='q'))
                ansatz, transpile_layout = change_mapping(ansatz, ansatz.layout.final_layout, init_layout, backend.configuration().coupling_map)
                hamiltonian_t = transpile_operator(hamiltonian, transpile_layout, qubits)
            else:
                hamiltonian_t = transpile_operator(hamiltonian, ansatz.layout, qubits)
        else:
            hamiltonian_t = hamiltonian

        qcirc = ansatz

        cnots = qcirc.count_ops().get('cx') or 0
        cnots += qcirc.count_ops().get('ecr') or 0

        avg_energy = 0
        for _ in tqdm(range(run_repeats), position=1, leave=False, desc="rep"):
            result = VQE(ansatz, hamiltonian_t, training_estimator)
            energy = result.fun
            avg_energy += energy
        avg_energy /= run_repeats

        params_list = result.x
        energy = result.fun

        cnots = qcirc.count_ops().get('cx') or 0
        cnots += qcirc.count_ops().get('ecr') or 0

        if not cnots:
            qcirc_t = transpile(qcirc, backend=backend, optimization_level=0)
            cnots = qcirc_t.count_ops().get('cx') or 0
            cnots += qcirc_t.count_ops().get('ecr') or 0

        num_parameters = ansatz.num_parameters

        parameters = np.reshape(params_list, (-1, num_parameters)).tolist()
        batch_size = len(parameters)

        if testing_real_quantum_computer and not skip_transpilation:
            qcirc = transpile(qcirc, backend, optimization_level=0, routing_method="sabre")
            hamiltonian_t = transpile_operator(hamiltonian, qcirc.layout, qubits)

            cnots = qcirc.count_ops().get('cx') or 0
            cnots += qcirc.count_ops().get('ecr') or 0
        elif testing_real_quantum_computer and not training_real_quantum_computer:
            qcirc = modified_transpile(qcirc, backend, pm_type="pad")
            hamiltonian_t = transpile_operator(hamiltonian, qcirc.layout, qubits)

        if testing_real_quantum_computer:
            job = None
        else:
            job = testing_estimator.run(batch_size * [qcirc], batch_size * [hamiltonian_t], parameters)

        jobs.append((avg_energy, cnots, qcirc, hamiltonian_t, parameters, job))
    
    if train_session:
        train_session.close()

    jobs.sort(key=lambda x: x[:2])

    print()
    if testing_real_quantum_computer:
        print("--Results--")
        print()
        last_job_id = None
        for idx, (energy, cnots, qcirc, hamiltonian_t, parameters, _) in tqdm(enumerate(jobs), position=0, desc="real qc run"):
            job_id = "N/A"
            if idx < real_qc_runs:
                if last_job_id:
                    last_job = service.job(last_job_id)
                    while last_job.status() not in JOB_FINAL_STATES:
                        time.sleep(60)
                batch_size = len(parameters)
                # job = testing_estimator.run(batch_size * [qcirc], batch_size * [hamiltonian_t], parameters)
                job = testing_estimator.run(pubs = [(qcirc, [hamiltonian_t] , parameters)])
                job_id = job.job_id()

            error = energy - real_energy

            last_job_id = job_id

            print()
            print(f"--Job {idx}--")
            print("Job ID:", job_id)
            print("Ground Energy:", energy)
            print("Error:", error)
            print("CNOT Count:", cnots)
            
        test_session.close()
    else:
        print("--Results--")
        print()

        energy_tally = 0
        error_tally = 0
        cnot_tally = 0
        saved_tally = 0
        saved_error_tally = 0

        errors = []
        cnot_list = []
        for idx, (energy, cnots, qcirc, hamiltonian_t, parameters, job) in tqdm(enumerate(jobs)):
            error = energy - real_energy

            energy_tally += energy
            error_tally += error
            cnot_tally += cnots

            errors.append(error)
            cnot_list.append(cnots)

            if not summary_only:
                print()
                print(f"--Job {idx}--")
                print("Ground Energy:", energy)
                print("Error:", error)
                print("CNOT Count:", cnots)

            if error < save_threshold:
                qpy_file = os.path.join(runtime_dir, f"qcirc_{saved_tally}.qpy")
                with open(qpy_file, "wb") as f:
                    qpy.dump(qcirc, f)
                
                hamil_file = os.path.join(runtime_dir, f"hamil_{saved_tally}.txt")
                with open(hamil_file, "w") as f: 
                    for pauli in hamiltonian_t.to_list():
                        f.write(f"{pauli[0]} {pauli[1]}\n")

                saved_tally += 1
                saved_error_tally += error

                if not summary_only:
                    print("Circuit saved to file")
            
            print()
        
        if summary_only:
            print("Suppressed individual job result printing")
            print()
        
        print("--Summary--")
        print("Avg. Ground Energy:", energy_tally/runs)
        print("Avg. Error:", error_tally/runs)
        print("Avg. CNOT Count:", cnot_tally/runs)
        if saved_tally:
            print("Saved Circuit Count:", saved_tally)
            print("Avg. Saved Error:", saved_error_tally/saved_tally)
        print("Runs:", runs)

        if plot:
            plt.scatter(cnot_list, errors, s=5)
            plt.show()

        return errors

def repeat_experiment(backend,
                      driver,
                      training_real_quantum_computer=False, 
                      testing_real_quantum_computer=False, 
                      skip_transpilation=True,
                      reps=1,
                      run_repeats=1,
                      from_dir="runtime/vqe/",
                      runtime_dir = "runtime/repeated_vqe/",
                      summary_only=False,
                      plot=True,
                      split_barrier=True):
    
    # Qiskit Setup

    problem = driver.run()

    mapper = JordanWignerMapper()

    hamiltonian = mapper.map(problem.second_q_ops()[0])

    real_energy = get_real_energy(hamiltonian)

    training_estimator, testing_estimator = get_estimators(backend, training_real_quantum_computer, testing_real_quantum_computer, approximation=True, skip_transpilation=skip_transpilation)

    jobs = []
    runs = int(len(os.listdir(from_dir))/2)
    for idx in tqdm(range(runs), position=0, desc="runs"):
        ansatz_path = os.path.join(from_dir, f"qcirc_{idx}.qpy")
        hamil_path = os.path.join(from_dir, f"hamil_{idx}.txt")

        with open(ansatz_path, "rb") as f:
            ansatz = qpy.load(f)[0]
        
        hamil_list = []
        with open(hamil_path, "r") as f:
            tup_lines = f.readlines()
            for line in tup_lines:
                split_line = line.split(" ")
                hamil_list.append((split_line[0], complex(split_line[1])))
        hamil = SparsePauliOp.from_list(hamil_list)

        offset = len(ansatz.parameters)
        num_params = offset*reps

        param_vec = ParameterVector('θ', num_params)

        ansatz, repeatable_circ = split_circuit_by_barrier(ansatz)

        for rep in range(reps):
            # Seems like saving the circuit loses layout info:
            # ansatz = change_mapping(ansatz, ansatz.layout.final_layout, ansatz.layout.initial_layout, backend.configuration().coupling_map)

            bind_dict = dict(zip(repeatable_circ.parameters, param_vec[rep*offset : (rep+1)*offset]))
            appendable_circ = repeatable_circ.assign_parameters(bind_dict)
            ansatz = ansatz.compose(appendable_circ)

        vqe = VQE(training_estimator, ansatz, SLSQP(maxiter=10000, ftol=1e-9))
        
        best_energy = float("inf")
        avg_energy = 0
        for _ in tqdm(range(run_repeats), position=1, leave=False, desc="rep"):
            vqe.initial_point = np.random.uniform(-np.pi, np.pi, vqe.ansatz.num_parameters)
            
            result = vqe.compute_minimum_eigenvalue(hamil)
            energy = result.eigenvalue

            if energy < best_energy:
                best_energy = energy
            avg_energy += energy
        avg_energy /= run_repeats

        qcirc = result.optimal_circuit
        params = result.optimal_parameters
        
        cnots = qcirc.count_ops().get('cx') or 0

        num_parameters = ansatz.num_parameters
        params_list = result.optimizer_result.x

        parameters = np.reshape(params_list, (-1, num_parameters)).tolist()
        batch_size = len(parameters)

        job = testing_estimator.run(batch_size * [qcirc], batch_size * [hamil], parameters)

        jobs.append((avg_energy, cnots, best_energy))
    
    print()
    if testing_real_quantum_computer:
        print("Writing job_ids to file")
        print()
        print("--CNOT Results--")
        job_file = os.path.join(runtime_dir, "running_jobs.txt")
        with open(job_file, "w") as f:
            for (job, cnots) in jobs:
                job_id = job.job_id()
                f.write(f"{job_id}\n")
                print(cnots)
            
            f.close()
    else:
        print("--Results--")
        print()

        energy_tally = 0
        error_tally = 0
        cnot_tally = 0

        errors = []
        cnot_list = []
        for idx, (avg_energy, cnots, best_energy) in tqdm(enumerate(jobs)):
            # estimator_result = job.result()
            # values = estimator_result.values
            # energy = values[0] if len(values) == 1 else values

            error = avg_energy - real_energy
            best_err = best_energy - real_energy

            energy_tally += avg_energy
            error_tally += error
            cnot_tally += cnots

            errors.append(error)
            cnot_list.append(cnots)

            if not summary_only:
                print()
                print(f"--Job {idx}--")
                print("Avg. Ground Energy:", avg_energy)
                print("Avg. Error:", error)
                print("CNOT Count:", cnots)
                print("Best Error:", best_err)
            
            print()
        
        if summary_only:
            print("Suppressed individual job result printing")
            print()
        
        print("--Summary--")
        print("Avg. Ground Energy:", energy_tally/runs)
        print("Avg. Error:", error_tally/runs)
        print("Avg. CNOT Count:", cnot_tally/runs)
        print("Runs:", runs)

        if plot:
            plt.scatter(cnot_list, errors, s=5)
            plt.show()

        return errors
            
def cross_domain_experiment(backend,
                      driver,
                      training_real_quantum_computer=False, 
                      testing_real_quantum_computer=False, 
                      skip_transpilation=True,
                      run_repeats=1,
                      save_dir="runtime/vqe/",
                      runtime_dir = "runtime/repeated_vqe/",
                      summary_only=False,
                      plot=True):
    
    # Qiskit Setup

    problem = driver.run()

    mapper = JordanWignerMapper()

    hamiltonian = mapper.map(problem.second_q_ops()[0])

    real_energy = get_real_energy(hamiltonian)

    training_estimator, testing_estimator = get_estimators(backend, training_real_quantum_computer, testing_real_quantum_computer, approximation=True, skip_transpilation=skip_transpilation)

    # #Need to construct a base ansatz for hamiltonian transpilation
    # base_ansatz = EfficientSU2(8, reps=1)
    # qubits = base_ansatz.qubits
    # base_t = transpile(base_ansatz, backend)
    # layout = base_t.layout
    # hamil_t = transpile_operator(hamiltonian, layout, qubits)

    jobs = []
    runs = int(len(os.listdir(save_dir))/2)
    for idx in tqdm(range(runs), position=0, desc="runs"):
        ansatz_path = os.path.join(save_dir, f"qcirc_{idx}.qpy")
        hamil_path = os.path.join(save_dir, f"hamil_{idx}.txt")

        with open(ansatz_path, "rb") as f:
            ansatz = qpy.load(f)[0]

        hamil_list = []
        with open(hamil_path, "r") as f:
            tup_lines = f.readlines()
            for line in tup_lines:
                split_line = line.split(" ")
                hamil_list.append((split_line[0], complex(split_line[1])))
        hamil = SparsePauliOp.from_list(hamil_list)

        vqe = VQE(training_estimator, ansatz, SLSQP(maxiter=10000, ftol=1e-9))
        
        best_energy = float("inf")
        avg_energy = 0
        for _ in tqdm(range(run_repeats), position=1, leave=False, desc="rep"):
            vqe.initial_point = np.random.uniform(-np.pi, np.pi, vqe.ansatz.num_parameters)

            result = vqe.compute_minimum_eigenvalue(hamil)
            energy = result.eigenvalue

            if energy < best_energy:
                best_energy = energy
            avg_energy += energy
        avg_energy /= run_repeats

        qcirc = result.optimal_circuit
        params = result.optimal_parameters
        
        cnots = qcirc.count_ops().get('cx') or 0

        num_parameters = ansatz.num_parameters
        params_list = result.optimizer_result.x

        parameters = np.reshape(params_list, (-1, num_parameters)).tolist()
        batch_size = len(parameters)

        job = testing_estimator.run(batch_size * [qcirc], batch_size * [hamil], parameters)

        jobs.append((avg_energy, cnots, best_energy))
    
    print()
    if testing_real_quantum_computer:
        print("Writing job_ids to file")
        print()
        print("--CNOT Results--")
        job_file = os.path.join(runtime_dir, "running_jobs.txt")
        with open(job_file, "w") as f:
            for (job, cnots) in jobs:
                job_id = job.job_id()
                f.write(f"{job_id}\n")
                print(cnots)
            
            f.close()
    else:
        print("--Results--")
        print()

        energy_tally = 0
        error_tally = 0
        cnot_tally = 0

        errors = []
        cnot_list = []
        for idx, (avg_energy, cnots, best_energy) in tqdm(enumerate(jobs)):
            # estimator_result = job.result()
            # values = estimator_result.values
            # energy = values[0] if len(values) == 1 else values

            error = avg_energy - real_energy
            best_err = best_energy - real_energy

            energy_tally += avg_energy
            error_tally += error
            cnot_tally += cnots

            errors.append(error)
            cnot_list.append(cnots)

            if not summary_only:
                print()
                print(f"--Job {idx}--")
                print("Avg. Ground Energy:", avg_energy)
                print("Avg. Error:", error)
                print("CNOT Count:", cnots)
                print("Best Error:", best_err)
            
            print()
        
        if summary_only:
            print("Suppressed individual job result printing")
            print()
        
        print("--Summary--")
        print("Avg. Ground Energy:", energy_tally/runs)
        print("Avg. Error:", error_tally/runs)
        print("Avg. CNOT Count:", cnot_tally/runs)
        print("Runs:", runs)

        if plot:
            plt.scatter(cnot_list, errors, s=5)
            plt.show()

        return errors

def run_vqe():
    parser = argparse.ArgumentParser(prog='VQE', description='Run VQE experiments')

    #For running with main.py
    parser.add_argument('vqe', nargs='?')
    
    parser.add_argument('-b', '--backend', dest='backend', default='ibm_kyiv', choices=["ibm_kyoto", "ibm_sherbrooke", "ibm_guadalupe", "ibm_kyiv"])
    parser.add_argument('-c', '--computer', dest='computer', default='fake', choices=["fake", "1", "2", "3", "4", "5", "6"])
    parser.add_argument('-m', '--molecule', dest='benchmark', required=True, choices=["h2", "h4", "h6", "lih", "beh2"])
    parser.add_argument('-r', '--routing', dest="routing", default='random_bridge')
    parser.add_argument('-bf', '--bridge_factor', dest='bridging_factor', default='-1')
    parser.add_argument('-g', '--gate', dest='bridge_gate', default='cx', choices=['cx', 'dcx', 'iswap', 'cz'])
    parser.add_argument('-a', '--ansatz', dest='ansatz', default='hardware', choices=['hardware', 'uccsd', 'twolocal'])
    parser.add_argument('-rp', '--reps', dest='reps', default='1')
    parser.add_argument('-rn', '--runs', dest='runs', default='50')
    parser.add_argument('-rr', '--real_runs', dest='real_runs', default='1')
    parser.add_argument('-d', '--dir', dest='runtime_dir', default='runtime/vqe')

    args = parser.parse_args()

    backend_name = args.backend
    computer = args.computer
    benchmark = args.benchmark
    routing = args.routing
    bridging_factor = float(args.bridging_factor)
    bridge_gate = args.bridge_gate
    ansatz = args.ansatz
    ansatz_reps = int(args.reps)
    runs = int(args.runs)
    real_runs = int(args.real_runs)
    runtime_dir = args.runtime_dir
    
    print(args)

    use_real_qc = False
    service = None
    if computer != "fake":
        use_real_qc = True
        service = qiskit_service(computer)
    
    backend = qiskit_backend(backend_name, computer)

    driver = molecule_driver(benchmark)

    #Non-guadalupe machines are too large to support non-trivial transpilation due to ancilla padding and simulation memory constraints
    #However, due to the large size of these machines, trivial layout results in same (or equivalent) circuits as non-trivial layout for all used benchmarks
    if backend_name == "ibm_guadalupe":
        transformations = [transpile_transform(backend=backend, routing_method=routing, bridging_factor=bridging_factor, bridge_gate=bridge_gate, pad_ancilla=True, seed_transpiler=seed)] 
    else:
        transformations = [transpile_transform(backend=backend, layout_method="trivial", routing_method=routing, bridging_factor=bridging_factor, bridge_gate=bridge_gate, pad_ancilla=False, seed_transpiler=seed)] 

    experiment(backend=backend, 
                driver=driver, 
                service=service,
                ansatz_type=ansatz, 
                ansatz_reps=ansatz_reps,
                transformations=transformations, 
                training_real_quantum_computer=False, 
                testing_real_quantum_computer=use_real_qc, 
                skip_transpilation=True,
                runs=runs, 
                real_qc_runs=real_runs,
                run_repeats=1,
                summary_only=False,
                restore_mapping=False,
                save_threshold=0,
                plot = False,
                runtime_dir=runtime_dir)

def run_vqe_repeat():
    parser = argparse.ArgumentParser(prog='VQE', description='Run VQE experiments')

    #For running with main.py
    parser.add_argument('vqe_repeat', nargs='?')
    
    parser.add_argument('-b', '--backend', dest='backend', default='ibm_kyiv', choices=["ibm_kyoto", "ibm_sherbrooke", "ibm_guadalupe", "ibm_kyiv"])
    parser.add_argument('-c', '--computer', dest='computer', default='fake', choices=["fake", "1", "2", "3", "4", "5"])
    parser.add_argument('-m', '--molecule', dest='benchmark', default='h4', choices=["h2", "h4", "h6", "lih", "beh2"])
    parser.add_argument('-d', '--dir', dest='from_dir', default="runtime/vqe/")
    parser.add_argument('-rp', '--reps', dest='reps', default='1')
    parser.add_argument('-rn', '--runs', dest='runs', default='1')

    args = parser.parse_args()

    backend = args.backend
    computer = args.computer
    benchmark = args.benchmark
    from_dir = args.from_dir
    ansatz_reps = int(args.reps)
    runs = int(args.runs)

    use_real_qc = False
    if computer != "fake":
        use_real_qc = True

    backend = qiskit_backend(backend, computer)

    driver = molecule_driver(benchmark)

    print(args)

    repeat_experiment(backend=backend,
                      driver=driver,
                      training_real_quantum_computer=False, 
                      testing_real_quantum_computer=use_real_qc, 
                      skip_transpilation=True,
                      reps=ansatz_reps,
                      run_repeats=runs,
                      from_dir=from_dir,
                      runtime_dir = "runtime/repeated_vqe/",
                      summary_only=False,
                      plot=False,
                      split_barrier=True)


if __name__ == "__main__":
    run_vqe()