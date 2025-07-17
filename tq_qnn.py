import sys
import os
import argparse
import random

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeGuadalupe
from qiskit.circuit import ParameterVector, QuantumRegister
from qiskit.transpiler.layout import Layout
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver

import torchquantum as tq
from torchquantum.torchquantum.dataset import MNIST, Vowel

from transpiler import modified_transpile
from learning import train, test
from utils import example_accuracy, torch_device, qiskit2tq, split_circuit_by_barrier, change_mapping, transpile_operator, seed

random.seed(seed)

torch.set_num_threads(2)
# None: 77.9
# 1: 67.8
# 2: 64.2
# 3: 62.9
# 4: 70.4
# 5: 65.3
# 6: 69.1
# 7: 75.6
# 8: 72.2
# 9: 79.1
# 10: 80.5

class TQNet(tq.QuantumModule):
    def __init__(self, 
                 layers: list[QuantumCircuit],
                 encoder = None, 
                 num_classes = 8,
                 use_softmax = False):
        super().__init__()

        self.layers = tq.QuantumModuleList()
        
        for layer in layers:
            tq_layer = qiskit2tq(layer)

            self.layers.append(tq_layer)

        self.encoder = encoder
        self.num_classes = num_classes
        self.use_softmax = use_softmax

        self.service = "TorchQuantum"
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, device, x, use_qiskit=False):
        bsz = x.shape[0]
        device.reset_states(bsz)
        
        x = F.avg_pool2d(x, 6)
        x = x.view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.layers, self.measure, x
            )
        else:
            if self.encoder:
                self.encoder(device, x)

            for layer in self.layers:
                layer(device)

            meas = self.measure(device)

        if self.num_classes == 4:
            meas = meas.reshape(bsz, 4, 2).sum(-1).squeeze()
        elif self.num_classes == 2:
            meas = meas.reshape(bsz, 2, 4).sum(-1).squeeze()

        if self.use_softmax:
            meas = F.log_softmax(meas, dim=1)
        
        return meas

def train_test_model(model,
                    tq_device,
                    dataset,
                    epochs=1):
    
    loss_fn = F.nll_loss
    acc_fn = example_accuracy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    train_dl = torch.utils.data.DataLoader(dataset['train'], batch_size=32, sampler=torch.utils.data.RandomSampler(dataset['train']))
    val_dl = torch.utils.data.DataLoader(dataset['valid'], batch_size=32, sampler=torch.utils.data.RandomSampler(dataset['valid']))
    test_dl = torch.utils.data.DataLoader(dataset['test'], batch_size=32, sampler=torch.utils.data.RandomSampler(dataset['test']))

    print("--Training--", file=sys.stderr)
    train_losses = train(model, train_dl, epochs, loss_fn, optimizer, val_dl=val_dl, acc_fn=acc_fn, device=tq_device)

    print(file=sys.stderr)
    print("--Testing--", file=sys.stderr)
    loss, accuracy = test(model, eval_dl=test_dl, acc_fn=acc_fn, loss_fn=loss_fn, device=tq_device)
    
    print("Loss:", loss, file=sys.stderr)
    print("Accuracy:", accuracy, file=sys.stderr)
    print(file=sys.stderr)

    return loss, accuracy


def get_dataset(dataset_name):

    data_name, num_classes = dataset_name.split("_")
    num_classes = int(num_classes)
    digits = range(num_classes)
    
    match data_name:
        case "mnist":
            dataset = MNIST(
                root=f"./data/mnist_data",
                train_valid_split_ratio=[.95, .05],
                digits_of_interest=digits,
                n_test_samples=300,
            )

        case "fashion":
            dataset = MNIST(
                root=f"./data/fashion_data",
                fashion=True,
                train_valid_split_ratio=[.95, .05],
                digits_of_interest=digits,
                n_test_samples=300,
            )

        case "vowel":
            dataset = Vowel(
                root=f"./data/vowel_data",
                train_valid_split_ratio=[.95, .05],
                digits_of_interest=digits,
                n_test_samples=300,
            )

    return dataset, num_classes

def QuantumEmbedding(n_qubits, reps):

    ansatz = QuantumCircuit(n_qubits)
    for idx in range(reps):
        ansatz = ansatz.compose(TwoLocal(n_qubits, ['rx'], ['rzz'], reps=1, parameter_prefix=f'θ_{idx}', skip_final_rotation_layer=True))
        ansatz = ansatz.compose(TwoLocal(n_qubits, ['ry'], [], reps=1, parameter_prefix=f'Φ_{idx}', skip_final_rotation_layer=True))
    
    return ansatz

def experiment(backend,
               dataset_name,
               ansatz_name = "hardware",
               custom_transpile=False,
               sabre=False,
               routing="random_bridge",
               bridging_factor=-1,
               bridge_gate="cx",
               epochs = 1,
               runs = 1,
               run_repeats=1,
               ansatz_reps=1,
               topn=0,
               restore_mapping=False,
               cnot_max=None,
               runtime_dir="runtime/qnn_circs/",
               summary_only=False,
               plot=True):
    
    # Hamiltonian Setup
    r = 1.5
    driver = PySCFDriver(f"H 0.0 0.0 0.0; H 0.0 0.0 {r}; H 0.0 0.0 {2*r}; H 0.0 0.0 {3*r}")
    problem = driver.run()
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(problem.second_q_ops()[0])
    
    n_qubits = 8

    tq_device = tq.QuantumDevice(n_wires=n_qubits).to(torch_device)

    if (n_qubits == 4):
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
    elif (n_qubits == 2):
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["2x8_rxryrzrxryrzrxry"])
    elif (n_qubits == 8):
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["8x2_ryz"])

    jobs = []
    for _ in tqdm(range(runs), position=0, desc="run"):

        best_accuracy = 0

        while best_accuracy == 0:
            if ansatz_name == "hardware":
                ansatz = EfficientSU2(n_qubits, reps=ansatz_reps)
            elif ansatz_name == "quantum_embedding":
                ansatz = QuantumEmbedding(n_qubits, reps=ansatz_reps)
            
            qubits = ansatz.qubits

            if custom_transpile:
                ansatz = modified_transpile(ansatz, backend, routing_method=routing, bridging_factor=bridging_factor, bridge_gate=bridge_gate)
                if cnot_max:
                    cnots = ansatz.count_ops().get('cx')
                    while (cnots > cnot_max):
                        ansatz = modified_transpile(ansatz, backend, routing_method=routing, bridging_factor=bridging_factor, bridge_gate=bridge_gate)
            else:
                if not sabre:
                    ansatz = transpile(ansatz, backend, optimization_level=0)
                else:
                    ansatz = transpile(ansatz, backend, optimization_level=0, routing_method="sabre")
            
            if restore_mapping:
                init_layout = Layout.from_intlist(list(range(ansatz.num_qubits)), QuantumRegister(ansatz.num_qubits, name='q'))
                ansatz, transpile_layout = change_mapping(ansatz, ansatz.layout.final_layout, init_layout, backend.configuration().coupling_map)
            else:
                transpile_layout = ansatz.layout
            
            hamil_t = transpile_operator(hamiltonian, transpile_layout, qubits)

            cnots = ansatz.count_ops().get('cx') or 0

            dataset, num_classes = get_dataset(dataset_name)

            avg_accuracy = 0
            for _ in tqdm(range(run_repeats), position=1, leave=False, desc="rep"):
                params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
                ansatz_b = ansatz.bind_parameters(params)

                model = TQNet([ansatz_b], encoder, num_classes=num_classes, use_softmax=True).to(torch_device)

                try:
                    _, accuracy = train_test_model(model, tq_device, dataset, epochs=epochs)
                except IndexError as e:
                    print(f"{e}", file=sys.stderr)
                    accuracy = 0.0

                if accuracy > best_accuracy:
                    best_accuracy = accuracy

                avg_accuracy += accuracy
            avg_accuracy /= run_repeats

        jobs.append((avg_accuracy, cnots, ansatz, best_accuracy, hamil_t))
    
    jobs.sort(key=lambda x: (-x[0],x[1]))

    print("--Results--")
    print()

    acc_tally = 0
    cnot_tally = 0
    saved_tally = 0
    saved_acc_tally =0
    acc_list = []
    cnot_list = []
    to_save = topn
    for idx, job in enumerate(jobs):
        accuracy = job[0]
        cnots = job[1]
        qcirc = job[2]
        best_acc = job[3]

        acc_tally += accuracy
        cnot_tally += cnots

        acc_list.append(accuracy)
        cnot_list.append(cnots)

        if not summary_only:
            print(f"--Run {idx}--")
            print("Avg. Accuracy:", accuracy)
            print("CNOTs:", cnots)
            print("Best Accuracy:", best_acc)
            print()
        
        if to_save:
            cur_dir = runtime_dir

            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

            qpy_file = os.path.join(cur_dir, f"qcirc_{saved_tally}.qpy")
            with open(qpy_file, "wb") as f:
                qpy.dump(qcirc, f)
            
            hamil_file = os.path.join(cur_dir, f"hamil_{saved_tally}.txt")
            with open(hamil_file, "w") as f: 
                for pauli in hamil_t.to_list():
                    f.write(f"{pauli[0]} {pauli[1]}\n")

            saved_tally += 1
            saved_acc_tally += accuracy
            
            to_save -= 1
    
    if summary_only:
        print("Suppressed individual job result printing")
        print()

    print("--Summary--")
    print("Avg. Accuracy:", acc_tally/runs)
    print("Avg. CNOT Count:", cnot_tally/runs)
    print()

    if plot:
        plt.scatter(cnot_list, acc_list, s=5)
        plt.show()
    
    return acc_list, cnot_list

def repeat_experiment(dataset_name,
                      ansatz_reps=1,
                      run_reps=1,
                      epochs=1,
                      from_vqe=False,
                      save_dir="runtime/qnn_circs/",
                      summary_only=False,
                      plot=True):
    
    n_qubits = 8

    tq_device = tq.QuantumDevice(n_wires=n_qubits).to(torch_device)

    if (n_qubits == 4):
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
    elif (n_qubits == 2):
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["2x8_rxryrzrxryrzrxry"])
    elif (n_qubits == 8):
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["8x2_ryz"])
    
    jobs = []

    runs = len(os.listdir(save_dir))/2
    # if from_vqe: #Temporary fix
    runs = int(runs/2)

    for idx in tqdm(range(5,runs+5), position=0, desc="run"):
        ansatz_path = os.path.join(save_dir, f"qcirc_{idx}.qpy")

        with open(ansatz_path, "rb") as f:
            ansatz = qpy.load(f)[0]

        offset = len(ansatz.parameters)
        num_params = offset*ansatz_reps

        param_vec = ParameterVector('p', num_params)

        if from_vqe:
            ansatz, repeatable_circ = split_circuit_by_barrier(ansatz)
        else:
            repeatable_circ = ansatz

        for rep in range(ansatz_reps):
            # Seems like saving the circuit loses layout info:
            # ansatz = change_mapping(ansatz, ansatz.layout.final_layout, ansatz.layout.initial_layout, backend.configuration().coupling_map)

            bind_dict = dict(zip(repeatable_circ.parameters, param_vec[rep*offset : (rep+1)*offset]))
            appendable_circ = repeatable_circ.assign_parameters(bind_dict)
            ansatz = ansatz.compose(appendable_circ)

        cnots = ansatz.count_ops().get('cx') or 0
        dataset, num_classes = get_dataset(dataset_name)

        best_accuracy = 0
        avg_accuracy = 0
        for _ in tqdm(range(run_reps), position=1, leave=False, desc="rep"):
            params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
            ansatz = ansatz.bind_parameters(params)

            model = TQNet([ansatz], encoder, num_classes=num_classes, use_softmax=True).to(torch_device)

            _, accuracy = train_test_model(model, tq_device, dataset, epochs=epochs)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            avg_accuracy += accuracy
        avg_accuracy /= run_reps

        jobs.append((avg_accuracy, cnots, best_accuracy, ansatz))
    
    jobs.sort(key=lambda x: (-x[0],x[1]))

    print("--Results--")
    print()

    accuracy_tally = 0
    cnot_tally = 0

    accuracy_list = []
    cnot_list = []
    for idx, job in tqdm(enumerate(jobs)):
        accuracy = job[0]
        cnots = job[1]
        best_accuracy = job[2]
        qcirc = job[3]

        accuracy_tally += accuracy
        cnot_tally += cnots

        accuracy_list.append(accuracy)
        cnot_list.append(cnots)

        if not summary_only:
            print(f"--Job {idx}--")
            print("Avg. Accuracy:", accuracy)
            print("CNOT Count:", cnots)
            print("Best Accuracy:", best_accuracy)
        
        print()
    
    if summary_only:
        print("Suppressed individual job result printing")
        print()
    
    print("--Summary--")
    print("Avg. Accuracy:", accuracy_tally/runs)
    print("Avg. CNOT Count:", cnot_tally/runs)
    print("Runs:", runs)

    if plot:
        plt.scatter(cnot_list, accuracy_list, s=5)
        plt.show()

    return accuracy_list, cnots

def run_qnn():
    parser = argparse.ArgumentParser(prog='VQE', description='Run VQE experiments')

    #For running with main.py
    parser.add_argument('qnn', nargs='?')
    
    parser.add_argument('-b', '--benchmark', dest='benchmark', default='mnist_4', choices=["mnist_2", "mnist_4", "fashion_2", "fashion_4"])
    parser.add_argument('-r', '--routing', dest="routing", default='random_bridge')
    parser.add_argument('-g', '--gate', dest='bridge_gate', default='cx', choices=['cx', 'dcx', 'iswap', 'cz'])
    parser.add_argument('-a', '--ansatz', dest='ansatz', default='hardware', choices=['hardware', 'quantum_embedding'])
    parser.add_argument('--reps', dest='reps', default='1')
    parser.add_argument('--runs', dest='runs', default='50')
    parser.add_argument('-e', '--epochs', dest='epochs', default='10')

    args = parser.parse_args()

    benchmark = args.benchmark
    routing = args.routing
    bridge_gate = args.bridge_gate
    ansatz = args.ansatz
    ansatz_reps = int(args.reps)
    runs = int(args.runs)
    epochs = int(args.epochs)

    print(args)

    backend = FakeGuadalupe()

    # TopN
    experiment(backend,
            benchmark,
            ansatz_name=ansatz,
            custom_transpile=True,
            routing=routing,
            epochs=epochs,
            runs=runs,
            ansatz_reps=ansatz_reps,
            run_repeats=1,
            topn=runs/5,
            restore_mapping=False,
            summary_only=False,
            plot=False,
            bridge_gate=bridge_gate,
            bridging_factor=-1)

if __name__ == "__main__":
    run_qnn()
    
    # Standard Repeat
    # repeat_experiment("mnist_4",
    #                   reps=8,
    #                   epochs=10,
    #                   split_barrier=False,
    #                   summary_only=False,
    #                   plot=True)
    
    # Repeat from VQE H4
    # repeat_experiment("fashion_4",
    #                   run_reps=5,
    #                   epochs=10,
    #                   from_vqe=True,
    #                   save_dir="runtime/vqe_h4/",
    #                   summary_only=False,
    #                   plot=False)