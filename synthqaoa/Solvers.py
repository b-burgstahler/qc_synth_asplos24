import networkx as nx
import numpy as np
from typing import Optional, List, Callable

from qiskit import IBMQ, Aer, QuantumCircuit, QuantumRegister
from qiskit.result import Counts
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer.noise import NoiseModel

from bqskit import Circuit, MachineModel
from bqskit.compiler import CompilationTask, Compiler, BasePass
from bqskit.passes import LayerGenerator, SimpleLayerGenerator,WideLayerGenerator, SetRandomSeedPass, QuickPartitioner,GreedyPartitioner,ForEachBlockPass,ScanningGateRemovalPass,QSearchSynthesisPass,UnfoldPass, QFASTDecompositionPass
from bqskit.ext import bqskit_to_qiskit, model_from_backend, qiskit_to_bqskit
from bqskit.utils.random import seed_random_sources

from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

import sys
sys.path.append('/home/bburgst/synthesis/') 
from synthqaoa.customPasses import PrintCNOTsPass, PrintGateCountsPass
#from synthqaoa.customPasses import PrintCNOTsPass, PrintGateCountsPass
#print(sys.modules.keys())
import logging
logger = logging.getLogger(__name__)
logger.info('solvers.py')

# ch = logging.StreamHandler()
# fh = logging.FileHandler('solver.log')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# fh.setFormatter(formatter)

# ch.setLevel(logging.INFO)
# fh.setLevel(logging.DEBUG)

# logger.addHandler(ch)
# logger.addHandler(fh)



        
        

def max_cut_brute_force(G: nx.Graph, shots=1024):
    n = G.number_of_nodes()
    w = nx.adjacency_matrix(G).todense()

    best_cost_brute = 0

    counts: dict[str, int] = {}
    count = 0

    for b in range(2**n):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
        cost = 0
        for i in range(n):
            for j in range(n):
                cost = cost + w[i, j]*x[i]*(1-x[j])
        if best_cost_brute < cost:
            best_cost_brute = cost
            count = 0
        if best_cost_brute == cost:
            count += 1
        counts["".join([str(i) for i in x])] = cost

    for (key, val) in counts.items():
        if val == best_cost_brute:
            counts[key] = shots // count
        else:
            counts[key] = 0

    return counts

class QAOASolver:
    def __init__(self, qaoa:QuantumCircuit, 
                 #params:List[Parameter],
                objf:Callable[[Counts],float],
                noise_model:NoiseModel = None):
        self.qc_qaoa = qaoa ## need how cicuit and params are linked.
        # self.qc_params = params
        # self.graph = graph
        self.compute_expectation = objf
        self.all_circuits: list[QuantumCircuit] = []
        #self.param_history = []
        self.noise_model = noise_model
        #self.init_circuit()

    # def init_circuit(self): ## expect to recieve a parameterized circuit from Ellis
    #     nqubits = self.graph.number_of_nodes()

    #     self.qc_qaoa = QuantumCircuit(nqubits)

    #     beta = Parameter("$\\beta$")
    #     qc_mix = QuantumCircuit(nqubits)
    #     for i in range(0, nqubits):
    #         qc_mix.rx(2 * beta, i)

    #     gamma = Parameter("$\\gamma$")
    #     qc_p = QuantumCircuit(nqubits)
    #     for pair in list(self.graph.edges()):  # pairs of nodes
    #         qc_p.rzz(2 * gamma, pair[0], pair[1])

    #     qc_0 = QuantumCircuit(nqubits)
    #     qc_0.h(range(nqubits))

    #     self.qc_qaoa.append(qc_0, [i for i in range(0, nqubits)])
    #     self.qc_qaoa.append(qc_p, [i for i in range(0, nqubits)])
    #     self.qc_qaoa.append(qc_mix, [i for i in range(0, nqubits)])

    #     self.qc_params = [beta, gamma]

    #     self.qc_qaoa.measure_all()
    #     self.qc_qaoa = self.qc_qaoa.decompose()

    # /////////////////////////////
    #  ASSUME OBJ MAY BE SET FOR DIFFERENT TYPES OF PROBLEMS
    # /////////////////////////////

    # def maxcut_obj(self, x: str) -> float:
    #     """
    #     Given a bitstring as a solution, this function returns
    #     the number of edges shared between the two partitions
    #     of the graph.

    #     Args:
    #         x: str
    #         solution bitstring

    #         G: networkx graph

    #     Returns:
    #         obj: float
    #             Objective
    #     """
    #     obj = 0
    #     for i, j in self.graph.edges():
    #         if x[i] != x[j]:
    #             obj -= 1

    #     return obj

    # def compute_expectation(self, counts: dict) -> float:
    #     """
    #     Computes expectation value based on measurement results

    #     Args:
    #         counts: dict
    #                 key as bitstring, val as counts

    #     Returns:
    #         avg: float
    #             expectation value
    #     """

    #     avg = 0
    #     sum_count = 0
    #     for bitstring, count in counts.items():
    #         obj = self.obj(self.graph, bitstring)
    #         avg += obj * count
    #         sum_count += count

    #     return avg/sum_count

      # /////////////////////////////
    #  ASSUME OBJ MAY BE SET FOR DIFFERENT TYPES OF PROBLEMS
    # /////////////////////////////
    def __add_obj_counts(self, counts): ## helper function to appropriately name qubits based on qreg name
        qc = self.qc_qaoa
        obj = self.compute_expectation
        new_dict = {}
        temp = [qc.qregs[i].name for i in range(len(qc.qregs))]
        for key in [idx for idx in counts.keys()]:
            new_key = []
            for idx in range(len(key)):
                if int(key[idx]) == 1:
                    new_key.append(''.join(temp[idx]))
                else:
                    new_key.append('-')
            new_key.append(':  '+str(obj({key:counts[key]})))
            new_dict[''.join(new_key)] = counts[key]
        return new_dict

    def set_noise_model(self, model:NoiseModel=None):
        self.noise_model = model

    def create_qaoa_circ(self, theta: List[float]) -> QuantumCircuit:
        """
        Creates a parametrized qaoa circuit

        Args:  
            G: networkx graph
            theta: list
                unitary parameters

        Returns:
            qc: qiskit circuit
        """
        qc = self.qc_qaoa.bind_parameters(
            {self.qc_qaoa.parameters[i]: theta[i] for i in range(len(theta))})

        return qc

    def get_expectation(self, shots:int=1024):
        """
        Runs parametrized circuit

        Args:
            G: networkx graph
            p: int,
            Number of repetitions of unitaries
        """

        backend = Aer.get_backend('qasm_simulator') #noise free.
        backend.shots = shots

        def execute_circ(theta):

            qc = self.create_qaoa_circ(theta)
            #print(str(theta) + " " + str(qc.depth()), end="\t")
            logger.info(str(theta) + " " + str(qc.depth()))
            #self.param_history.append(theta)
            self.all_circuits.append(qc)

            counts = backend.run(qc, seed_simulator=10,
                                 nshots=shots).result().get_counts()

            return self.compute_expectation(counts) # TODO computing expectation should be done by nchoosek defined circ

        return execute_circ

    def minimize(self, init_params: List[float], options: dict = {'maxiter': 10}, method='COBYLA'):
        expectation = self.get_expectation()
        
        # options['disp'] = True
        jac, hes = None, None
        if method == 'COBYLA':
            options['rhobeg'] = 1
        elif method == 'Nelder-Mead':
            N = len(init_params)
            first_vertex = np.ones([1,N])
            other_vertices = 2.14*np.eye(N)+np.ones([N,N])
            simplex=np.concatenate((first_vertex,other_vertices), axis = 0)
            options['initial_simplex'] = simplex
            options['return_all'] = True
        else:
            jac = 'cs'
            hes = 'cs'
        
        res = minimize(expectation,init_params, method=method, options=options, jac=jac, hess=hes)
        return res

    def get_found_circuit(self,res):
        if type(res) == OptimizeResult:
            qc_res = self.create_qaoa_circ(res.x)
        elif type(res) == np.ndarray():
            qc_res = self.create_qaoa_circ(res)
        else:
            qc_res = None
            raise Exception('Invalid result vector')
        temp = self.qc_qaoa.copy_empty_like()
        temp._parameters = []

        return temp.compose(qc_res)

    def run_on_simulator(self, qc_res, w_noise:bool=True):
        backend = Aer.get_backend('aer_simulator')
        backend.shots = 1024
        #qc_res = self.get_found_circuit(self,res)
        if w_noise:
            noise = self.noise_model
        else:
            noise = None
        counts = backend.run(qc_res, seed_simulator=10, noise_model=noise).result().get_counts()
        obj_value = self.compute_expectation(counts)
        #counts = self.__add_obj_counts(counts)
        return counts, obj_value


class QAOASynthesisSolver(QAOASolver):
    def __init__(self, qaoa:QuantumCircuit, 
                 #params:List[Parameter], 
                 objf:Callable[[Counts],float],#=maxcut_obj, 
                 machine_model: Optional[MachineModel] = None,
                 noise_model:NoiseModel = None):
        super().__init__(qaoa, 
                         #params,
                        objf,
                        noise_model=noise_model)
        self.machine_model = machine_model
        self.minimized_qc: Optional[QuantumCircuit] = None
        self.matched_circuit: Optional[QuantumCircuit] = None

        def _get_layer_gen(machine_model: MachineModel = self.machine_model) -> LayerGenerator:
            """Build a `model`-compliant layer generator."""
            if machine_model is None:
                return SimpleLayerGenerator()

            tq_gates = [
                gate for gate in machine_model.gate_set if gate.num_qudits == 2]
            mq_gates = [
                gate for gate in machine_model.gate_set if gate.num_qudits > 2]
            sq_gates = [
                gate for gate in machine_model.gate_set if gate.num_qudits == 1]

            if len(tq_gates) == 1 and len(mq_gates) == 0:
                # if CNOTGate() in tq_gates:
                #     return FourParamGenerator()
                # else:
                return SimpleLayerGenerator(tq_gates[0])

            return WideLayerGenerator(tq_gates + mq_gates)

        default_passes =[
            SetRandomSeedPass(seed=0),
            QuickPartitioner(),
            #GreedyPartitioner(),
            ForEachBlockPass([
                QSearchSynthesisPass(layer_generator=_get_layer_gen(), instantiate_options={'seed': 0}), 
                ScanningGateRemovalPass(instantiate_options={'seed': 0}),
                PrintGateCountsPass('block')]),
            UnfoldPass(),
            PrintGateCountsPass('FINAL')
        ]

        qfast_passes = [
            SetRandomSeedPass(seed=0),
            QFASTDecompositionPass(),
            ForEachBlockPass([
                QSearchSynthesisPass(layer_generator=_get_layer_gen(), instantiate_options={'seed': 0}), 
                ScanningGateRemovalPass(instantiate_options={'seed': 0}),
                PrintGateCountsPass('block')]),
            UnfoldPass(),
            PrintGateCountsPass('FINAL')
        ]

        self.passes = default_passes
    
    def create_qaoa_circ(self, theta: List[float]) -> QuantumCircuit:
        if self.minimized_qc:
            qc = self.minimized_qc.bind_parameters(
                {self.minimized_qc.parameters[i]: theta[i] for i in range(len(theta))})

            qc.measure_all()

            return qc

        qc = self.qc_qaoa.bind_parameters(
            {self.qc_qaoa.parameters[i]: theta[i] for i in range(len(theta))})
        qc.remove_final_measurements()
        # qc.draw(output='mpl')
        bqskit_circuit = qiskit_to_bqskit(qc)

        

        task = CompilationTask(bqskit_circuit, self.passes)

        with Compiler() as compiler:
            out_circuit = compiler.compile(task)

        # template = out_circuit
        qc = bqskit_to_qiskit(out_circuit)
        # qc.draw(output='mpl')
        qc.measure_all()
        return qc

    def is_circuit_similar(self, qc_same_depth: List[QuantumCircuit]) -> bool:

        
        for reference_qc in qc_same_depth:
            reference_dag = circuit_to_dag(reference_qc)
            reference_circuit_nodes = reference_dag.topological_op_nodes()

            for qc in qc_same_depth:
                if qc == reference_qc or qc.depth() != reference_qc.depth():
                    continue

                dag = circuit_to_dag(qc)

                is_similar = True

                for node in dag.topological_op_nodes():
                    try:
                        next = reference_circuit_nodes.__next__()
                    except: 
                        break
                    if next is None or node.name != next.name:
                        is_similar = False
                        break

                if is_similar:
                    # they are identical
                    self.matched_circuit = qc
                    print('using circuit of depth %d' % (reference_qc.depth() + 1))
                    return True
        logger.info("No equivalent circuits found for depth %d" %
              (reference_qc.depth() + 1))
        #print("No equivalent circuits found for depth %d" %(reference_qc.depth() + 1))

        return False

    def minimize(self, init_params: List[float], options: dict = {'maxiter': 10}, phase2_options: dict = {'maxiter': 10}, method='COBYLA'):
        seed_random_sources(0)
        res = super().minimize(init_params, options=options,method=method) # previously set to always be 5
        logger.info('Current params:'+ str(res))
        # print(res)

        qcs_by_depth = {}

        for qc in self.all_circuits:
            qc.remove_final_measurements()
            if qc.depth() in qcs_by_depth:
                qcs_by_depth[qc.depth()].append(qc)
            else:
                qcs_by_depth[qc.depth()] = [qc]

        depths = sorted(qcs_by_depth.keys())

        for depth in depths:
            if self.is_circuit_similar(qcs_by_depth[depth]):
                break
            if (depth == list(depths)[-1]):
                selection = np.argmin([(5*x.count_ops()['cx']+x.count_ops()['u3']) for x in self.all_circuits])
                self.matched_circuit = self.all_circuits[selection]
                print('using template with lowest CNOT count (lowest u3 breaks ties)')

        # if we find a short circuit that has a match, we use it
        if self.matched_circuit is not None:
            self.minimized_qc = QuantumCircuit(self.qc_qaoa.num_qubits)

            matched_dag = circuit_to_dag(self.matched_circuit)

            #self.qc_params = []
            i = 0
            params_values = []

            for node in matched_dag.topological_op_nodes():
                for param in node.op.params:
                    p = Parameter("alpha_" + str(i)) #len(self.qc_params)))
                    #self.qc_params.append(p)
                    params_values.append(param)
                    node.op.params[node.op.params.index(param)] = p
                    i = i + 1
                self.minimized_qc.append(node.op, node.qargs)
            print('begin template optimization')
            seed_random_sources(0)
            return super().minimize(params_values, options=phase2_options, method=method)

        return res


class QAOAInstantiationSynthesisSolver(QAOASynthesisSolver):
    def __init__(self, qaoa: QuantumCircuit,
                 # params:List[Parameter], 
                 objf: Callable[[Counts], float],  # =maxcut_obj,
                 machine_model: Optional[MachineModel] = None,
                 noise_model:NoiseModel = None):
        super().__init__(qaoa, 
                         #params, 
                         objf,
                         machine_model=machine_model,
                         noise_model=noise_model)
        self.machine_model = machine_model
        self.template: Optional[Circuit] = None

    def create_qaoa_circ(self, theta: List[float]) -> QuantumCircuit:
        if self.minimized_qc:
            qc = self.minimized_qc.bind_parameters({self.minimized_qc.parameters[i]: theta[i] for i in range(len(theta))})

            qc.measure_all()

            return qc

        qc = self.qc_qaoa.bind_parameters(
            {self.qc_qaoa.parameters[i]: theta[i] for i in range(len(theta))})
        qc.remove_final_measurements()
        bqskit_circuit = qiskit_to_bqskit(qc)

        out_circuit = None

        if self.template is not None:
            ## TODO, include all synthesized circuits for consideration as template?
            target_unitary = bqskit_circuit.get_unitary()
            #print('start inst')
            out_circuit = self.template.instantiate(target_unitary, seed=0)
            #print('end inst')
            distance = out_circuit.get_unitary().get_distance_from(target_unitary, 2) ## Changed from 1 to 2
            diff = np.linalg.norm(out_circuit.get_unitary()-target_unitary) ## new calc
            #print( "HS- {:.16E}".format(distance)+ "   " + "{:.16E}".format(diff) + " ")
            #print( "{:.16E}".format(diff), " ", end="")
            logger.info( "HS- {:.16E}".format(distance)+ "   " + "{:.16E}".format(diff) + " ")
            distance = (distance+diff)/2
            if distance > 1e-8:
                out_circuit = None
            else:
                # print("Using template ", end="")
                logger.info("Using template ", end="")

        if out_circuit is None:
            task = CompilationTask(bqskit_circuit, self.passes)
            with Compiler() as compiler:
                # print('start synth')
                out_circuit = compiler.compile(task)
                # print('end synth')
        self.template = out_circuit
        qc = bqskit_to_qiskit(out_circuit)
        qc.measure_all()

        return qc

# if __name__ == "__main__":
#     from examples import ALL
#     import nchoosek

#     IBMQ.load_account()
#     provider = IBMQ.get_provider(hub='ibm-q-ncsu', group='nc-state', project='grad-qc-class')
#     backend = provider.get_backend('ibmq_guadalupe')
#     noise_model = NoiseModel.from_backend(backend)


#     for example_name in ALL:
#         env = ALL[example_name]
#         qc, obj = nchoosek.solver.circuit_gen(env)
#         testSolver = QAOAInstantiationSynthesisSolver(qc,obj)
#         opt_params = testSolver.minimize([1.0,1.0],{'maxiter': 30},{'maxiter': 400})