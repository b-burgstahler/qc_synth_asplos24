import numpy as np
from qiskit import QuantumCircuit
from qiskit.result.counts import Counts
from typing import Optional, Callable, Union
from qiskit_aer.noise import NoiseModel
from nchoosek import Environment

def rekey_counts(counts: Counts, circ: Optional[QuantumCircuit]= None, objf: Optional[Callable]= None):
    ''
    qc = circ
    obj = objf
    new_dict = {}
    if qc:
        temp = [qc.qregs[i].name for i in range(len(qc.qregs))]
    for key in [idx for idx in counts.keys()]:
        new_key = []
        if qc:
            for idx in range(len(key)):
                if int(key[idx]) == 1:
                    new_key.append(''.join(temp[idx]))
                    new_key.append(',')
                else:
                    new_key.append('-')
                    new_key.append(',')
        else:
            new_key = list(key)
        if obj:
            new_key.append(':  '+str(obj({key: counts[key]})))
        new_dict[''.join(new_key)] = counts[key]

    #new_Counts = Counts(new_dict,counts.time_taken,counts.creg_sizes,counts.memory_slots)
    return new_dict

def create_nck_solution(env: Union[Environment, QuantumCircuit, list], counts: Counts, amount: int = 1):
    '''Format most likely count as a nck solution dictionary'''
    assert (amount >= 1)
    all_nck = []
    for sol, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        nck_sol = {}
        if isinstance(env, Environment):
            enum = enumerate(env.ports())
        elif isinstance(env, QuantumCircuit):
            enum = enumerate([env.qregs[i].name for i in range(len(env.qregs))])
        else: # env is ordered list
            enum = enumerate(env)
        for idx, port in enum:
            nck_sol[port] = True if int(sol[idx]) else False
        all_nck.append(nck_sol)
    return all_nck[0:amount]
