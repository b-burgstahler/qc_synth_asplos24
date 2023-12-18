# import os
import sys
import qiskit
import qiskit_aer
# from qiskit.utils import QuantumInstance
from qiskit_ibm_provider import IBMProvider

# from qiskit_aer.noise import NoiseModel
import datetime

# sys.path.append('NchooseK')
import nchoosek
from nchoosek.solver.common import Result as nckResult
# import re
import random

# from dimod import SimulatedAnnealingSampler
from synthqaoa.Solvers import (
    QAOAInstantiationSynthesisSolver,
    QAOASynthesisSolver,
    QAOASolver,
)
from synthqaoa import helpers
import pandas as pd
# import numpy as np

## benchmarking forked from https://github.com/ejwilson3/nchoosek_sc22/blob/main/run_problems.py, adapted to new solver


def min_vert_cover(V, E):
    env = nchoosek.Environment()
    for vert in V:
        env.register_port(vert)
    for edge in E:
        env.nck([edge[0], edge[1]], {1, 2})
    env.minimize(V)
    return env


def max_cut(V, E):
    env = nchoosek.Environment()
    for vert in V:
        env.register_port(vert)
    for edge in E:
        env.different(edge[0], edge[1], soft=True)
    return env


def clique_cover(V, E, nCliques):
    env = nchoosek.Environment()
    # One hot encoding: each vertex is part of one clique.
    for vert in V:
        N = []
        for clique in range(nCliques):
            var = vert + str(clique)
            env.register_port(var)
            N.append(var)
        env.nck(N, {1})
    # Check every vertex pair; if they aren't connected, add nCliques constraints
    for idx, vert1 in enumerate(V[:-1]):
        for vert2 in V[idx + 1 :]:
            if ((vert1, vert2) not in E) and ((vert2, vert1) not in E):
                for clique in range(nCliques):
                    var1 = vert1 + str(clique)
                    var2 = vert2 + str(clique)
                    env.nck([var1, var2], {0, 1})
    return env


def min_set_cover(E, S):
    env = nchoosek.Environment()
    for key in S:
        env.register_port(key)
        env.nck([key], {0}, soft=True)
    for elem in E:
        N = []
        for key in S:
            if elem in S[key]:
                N.append(key)
        if N == []:
            print("Set cannot be covered")
            return
        env.nck(N, {range(1, len(N) + 1)})
    return env


def exact_set_cover(E, S):
    env = nchoosek.Environment()
    for key in S:
        env.register_port(key)
    for elem in E:
        N = []
        for key in S:
            if elem in S[key]:
                N.append(key)
        if N == []:
            print("Set cannot be covered")
            return
        env.nck(N, {1})
    return env


def map_color(V, E, ncolors=4):
    env = nchoosek.Environment()
    for vert in V:
        verts = []
        for col in range(ncolors):
            env.register_port(vert + str(col))
            verts.append(vert + str(col))
        env.nck(verts, {1})
    for edge in E:
        for col in range(ncolors):
            env.nck([edge[0] + str(col), edge[1] + str(col)], {0, 1})
    return env


def SAT3(cons):
    vari = []
    env = nchoosek.Environment()
    for con in cons:
        for v in con:
            if v not in vari:
                vari.append(v)
                env.register_port(v)
                if v[0] == "!":
                    if (v[1:]) in vari:
                        env.nck([v, v[1:]], {1})
                elif ("!" + v) in vari:
                    env.nck([v, "!" + v], {1})
        env.nck(con, {1, 2, 3})
    return env


def run_env_2(
    env, job, port_order, check_env, solver, simulator, quantum_instance, num_reads, idx
):
    # qtime1 = datetime.datetime.now()
    # qc, objf = nchoosek.solver.circuit_gen(env, None)
    # qtime2 = datetime.datetime.now()
    assert solver == "qaoa_hard" or solver == "synth_hard" or solver == "inst_hard"
    job_ch = provider.retrieve_job(job)
    counts = job_ch.result().get_counts()
    if quantum_instance:
        backend = quantum_instance
        noise_model = qiskit_aer.noise.NoiseModel.from_backend(backend)
    else:
        noise_model = None

    solutions = helpers.create_nck_solution(port_order, counts, amount=5)

    # res = env.solve(solver=solver)
    res = nckResult()
    res.tallies = None 
    res.solutions = solutions
    res.variables = port_order
    # res.solver_times = (stime1, stime2)
    # res.qubo_times = (qtime1, qtime2)
    res.depth = job_ch.circuits()[0].depth()
    res.qubit_names = list(env.ports())
    res.jobIDs = [job]
    res.qubits = nchoosek.solver.circuit_gen(env, None)[0].num_qubits

    ch = check_env.solve(solver="z3")
    if not ch or not res or not res.solutions:
        res = None
        count = None
        v = None
        valq = None
        check_quality = None
        qualities = None
        validations = None
    else:
        check = ch.solutions[0]
        # This checks how many runs satisfied the constraints
        v = 0
        # This checks how many runs optimized the constraints
        count = 0

        valq = []
        qualities = []
        validations = []

        # Compare the quality of the results to this value
        check_quality = check_env.quality(check)
        for r_idx, result in enumerate(res.solutions):
            valq_temp = env.valid(result)
            quality = env.quality(result)
            if res.tallies:
                add = res.tallies[r_idx]
            else:
                add = 1
            if valq_temp:  # valid implies all hard constraints met
                v += add
            if (
                quality == check_quality
            ):  # quality imples number of soft constraints met.
                count += add
            valq.append(valq_temp)
            qualities.append(quality)
            nck_validation = env.validation(result)
            custom_validation = (
                len(nck_validation.hard_passed),
                len(nck_validation.hard_failed),
                len(nck_validation.soft_passed),
                len(nck_validation.soft_failed),
            )
            validations.append(custom_validation)
    return res, count, v, valq, check_quality, qualities, validations


def write_out(
    times,
    results,
    total_counts,
    envs,
    good_counts,
    n_qubs,
    opt_counts,
    n_jobs,
    depths,
    start,
    n_probs,
    time_filename,
    results_filename,
    data_filename,
    hard_inds,
    check_quals,
    soft_counts,
    all_constraint_val,
):
    with open(time_filename, "a") as f:
        f.write(" \n")
        for time in times:
            f.write(str(time) + "\n")
    with open(results_filename, "a") as f:
        for idx, result in enumerate(results):
            f.write(str(result) + "\n")
            f.write(str(total_counts[idx]) + "\n")
    print(
        str(good_counts)
        + ", \n"
        + str(n_qubs)
        + ", \n"
        + str(opt_counts)
        + ", \n"
        + str(n_jobs)
        + ", \n"
        + str(depths)
        + ", \n"
        + str(hard_inds)
        + ", \n"
        + str(check_quals)
        + ", \n"
        + str(soft_counts)
        + ", \n"
        + str(all_constraint_val)
    )
    for idx, env in enumerate(envs):
        good = []
        bad = []
        name = str(start + idx % n_probs)
        if env and results[idx]:
            item = (
                str(good_counts[idx])
                + ", "
                + str(len(env.ports()))
                + ", "
                + str(len(env.constraints()))
                + ", "
                + str(n_qubs[idx])
                + ", "
                + str(opt_counts[idx])
                + ", "
                + str(n_jobs[idx])
                + ", "
                + str(depths[idx])
                + ", "
                + str(hard_inds[idx])
                + ", "
                + str(check_quals[idx])
                + ", "
                + str(soft_counts[idx])
                + ", "
                + str(all_constraint_val[idx])
                + ", "
                + str(int(name) + 1)
                + "\n"
            )
            if good_counts[idx] > 0:
                good.append(item)
            else:
                bad.append(item)
        with open(name + "_" + solver + "_" + data_filename, "a") as f:
            f.write("\n")
            for item in good:
                f.write(item)
            for item in bad:
                f.write(item)


def run_graph(
    V,
    E,
    nCliques,
    solver,
    simulator=False,
    color=True,
    time_filename="times.dat",
    results_filename="results.dat",
    data_filename="output.dat",
    quantum_instance=None,
    nQubs=2000,
    num_reads=100,
):
    # list of environments
    envs = []
    # list of lists of results
    results = []
    # list of lists of how often the result showed up; relevant only to ocean solver
    total_counts = []
    # list of environments for comparing soft constraints
    check_envs = []
    # list of results for comparing soft constraints
    check = []
    # list of times used in the solver; used primarily for qiskit solver in order to collect the jobs
    times = []
    # list of how many results satisfied all hard constraints. The number will only be more than one for the ocean solver
    good_counts = []
    # list of how many results had the maximum number of soft constraints met. The number will only be more than one for the ocean solver
    opt_counts = []
    # list of number of physical qubits used
    n_qubs = []
    # list of number of jobs used for qiskit solver
    n_jobs = []
    # list of circuit depths used for qiskit solver
    depths = []
    job_ids = []

    # list of valids
    hard_inds = []
    # list of check env qualities
    check_quals = []
    # list of soft constraints satisfied
    soft_counts = []
    # validations list
    all_constraints_validation = []

    try:
        envs.append(min_vert_cover(V, E))
        check_envs.append(min_vert_cover(V, E))
    except ValueError:
        envs.append(None)
        check_envs.append(None)
    try:
        envs.append(max_cut(V, E))
        check_envs.append(max_cut(V, E))
    except ValueError:
        envs.append(None)
        check_envs.append(None)
    # clique cover uses more qubits than the others
    if len(V) * nCliques <= nQubs and color:
        try:
            envs.append(clique_cover(V, E, nCliques))
            check_envs.append(clique_cover(V, E, nCliques))
        except ValueError:
            envs.append(None)
            check_envs.append(None)
    else:
        envs.append(None)
        check_envs.append(None)
    if len(V) * 4 <= nQubs and color:
        try:
            envs.append(map_color(V, E))
            check_envs.append(map_color(V, E))
        except ValueError:
            envs.append(None)
            check_envs.append(None)
    else:
        envs.append(None)
        check_envs.append(None)
    times.append(datetime.datetime.now())

    # Run all the problems and record the results
    for idx, env in enumerate(envs):
        # Add None to the lists in order to keep indices in line
        if not env:
            results.append(None)
            total_counts.append(None)
            opt_counts.append(None)
            good_counts.append(None)
            n_qubs.append(None)
            n_jobs.append(None)
            depths.append(None)
            hard_inds.append(None)
            check_quals.append(None)
            soft_counts.append(None)
            all_constraints_validation.append(None)
            times.append(datetime.datetime.now())
            continue
        # Run the actual problem and add it to the lists
        # res, count, v = run_env(env, check_envs[idx], solver, simulator, quantum_instance, num_reads, idx)

        row = df[(df["qubit_count"] == len(env.ports())) & (df["idx"] == idx)].head(1)
        job = row["job"].values[0]
        port_order_str = row["port_order"].values[0]
        port_order = (
            port_order_str.replace("[", "")
            .replace("'", "")
            .replace("]", "")
            .split(", ")
        )

        res, count, v, valq, check_quality, qualities, validations = run_env_2(
            env,
            job,
            port_order,
            check_envs[idx],
            solver,
            simulator,
            quantum_instance,
            num_reads,
            idx,
        )
        # res, count, v = run_env(env, check_envs[idx], quantum_instance)
        print(str(idx + 1) + "/" + str(sum(x is not None for x in envs)) + " complete")

        if res:
            results.append(res.solutions)
            total_counts.append(res.tallies)
            opt_counts.append(count)
            good_counts.append(v)
            n_qubs.append(res.qubits)
            depths.append(res.depth)
            hard_inds.append(valq)
            check_quals.append(check_quality)
            soft_counts.append(qualities)
            all_constraints_validation.append(validations)
            if res.jobIDs:
                n_jobs.append(len(res.jobIDs))
            else:
                n_jobs.append(0)
            job_ids.append(res.jobIDs)
            times.append(datetime.datetime.now())
        else:
            results.append(None)
            total_counts.append(None)
            opt_counts.append(None)
            good_counts.append(None)
            n_qubs.append(None)
            depths.append(None)
            n_jobs.append(None)
            job_ids.append(None)
            hard_inds.append(None)
            check_quals.append(None)
            soft_counts.append(None)
            all_constraints_validation.append(None)
            times.append(datetime.datetime.now())
    write_out(
        times,
        results,
        total_counts,
        envs,
        good_counts,
        n_qubs,
        opt_counts,
        n_jobs,
        depths,
        0,
        len(envs),
        time_filename,
        results_filename,
        data_filename,
        hard_inds,
        check_quals,
        soft_counts,
        all_constraints_validation,
    )

    return envs


def run_other(
    E,
    S,
    constraints,
    solver,
    simulator=False,
    time_filename="times.dat",
    results_filename="results.dat",
    data_filename="output.dat",
    quantum_instance=None,
    num_reads=100,
    p=1,
):
    # list of environments
    envs = []
    # list of lists of results
    results = []
    # list of lists of how often the result showed up; relevant only to ocean solver
    total_counts = []
    # list of environments for comparing soft constraints
    check_envs = []
    # list of results for comparing soft constraints
    check = []
    # list of times used in the solver; used primarily for qiskit solver in order to collect the jobs
    times = []
    # list of how many results satisfied all hard constraints. The number will only be more than one for the ocean solver
    good_counts = []
    # list of how many results had the maximum number of soft constraints met. The number will only be more than one for the ocean solver
    opt_counts = []
    # list of number of physical qubits used
    n_qubs = []
    # list of number of jobs used for qiskit solver
    n_jobs = []
    # list of circuit depths used for qiskit solver
    depths = []
    job_ids = []

    # list of valids
    hard_inds = []
    # list of check env qualities
    check_quals = []
    # list of soft constraints satisfied
    soft_counts = []

    all_constraints_validation = []

    # Not sure how many ancillary bits they will need. Min set cover especially.
    # try:
    #     envs.append(min_set_cover(E, S))
    #     check_envs.append(min_set_cover(E, S))
    # except:
    #     envs.append(None)
    #     check_envs.append(None)
    # try:
    #     envs.append(exact_set_cover(E, S))
    #     check_envs.append(exact_set_cover(E, S))
    # except:
    #     envs.append(None)
    #     check_envs.append(None)
    try:
        envs.append(SAT3(constraints))
        check_envs.append(SAT3(constraints))
    except:
        envs.append(None)
        check_envs.append(None)

    times.append(datetime.datetime.now())

    # Run all the problems and record the results
    for idx, env in enumerate(envs):
        # Add None to the lists in order to keep indices in line
        if not env:
            results.append(None)
            total_counts.append(None)
            opt_counts.append(None)
            good_counts.append(None)
            depths.append(None)
            n_qubs.append(None)
            n_jobs.append(None)
            job_ids.append(None)
            times.append(datetime.datetime.now())
            continue
        # Run the actual problem and add it to the lists
        # res, count, v, valq, check_quality, qualities, validations = run_env(
        #     env, check_envs[idx], solver, simulator, quantum_instance, num_reads, idx
        # )

        row = df[
            (df["qubit_count"] == nchoosek.solver.circuit_gen(env, None)[0].num_qubits)
            & (df["idx"] == idx)
        ].head(1)
        job = row["job"].values[0]
        port_order_str = row["port_order"].values[0]
        port_order = (
            port_order_str.replace("[", "")
            .replace("'", "")
            .replace("]", "")
            .split(", ")
        )

        res, count, v, valq, check_quality, qualities, validations = run_env_2(
            env,
            job,
            port_order,
            check_envs[idx],
            solver,
            simulator,
            quantum_instance,
            num_reads,
            idx,
        )
        print(str(idx + 1) + "/" + str(sum(x is not None for x in envs)) + " complete")
        results.append(res.solutions)
        total_counts.append(res.tallies)
        opt_counts.append(count)
        good_counts.append(v)
        n_qubs.append(res.qubits)
        depths.append(res.depth)
        hard_inds.append(valq)
        check_quals.append(check_quality)
        soft_counts.append(qualities)
        all_constraints_validation.append(validations)
        if res.jobIDs:
            n_jobs.append(len(res.jobIDs))
        else:
            n_jobs.append(0)
        job_ids.append(res.jobIDs)
        times.append(datetime.datetime.now())
    write_out(
        times,
        results,
        total_counts,
        envs,
        good_counts,
        n_qubs,
        opt_counts,
        n_jobs,
        depths,
        4,
        len(envs),
        time_filename,
        results_filename,
        data_filename,
        hard_inds,
        check_quals,
        soft_counts,
        all_constraints_validation,
    )

    return envs


if __name__ == "__main__":
    # qiskit.IBMQ.load_account()
    # provider = qiskit.IBMQ.get_provider(hub='ibm-q-ncsu', group='nc-state', project='grad-qc-class')

    IBMProvider()
    provider = IBMProvider(instance="ibm-q-ncsu/nc-state/grad-qc-class")
    #########################Change machine here####################################
    backend = provider.get_backend("ibm_hanoi")
    ################################################################################
    p = 3  # number of QAOA repetitions
    nqubs = 21

    if len(sys.argv) == 1:
        solver = "synth"
    else:
        solver = str(sys.argv[1])  ## NOTE: _nono corresponds to 'no noise'
        if (
            solver != "qaoa"
            and solver != "synth"
            and solver != "inst"
            and solver != "qaoa_nono"
            and solver != "synth_nono"
            and solver != "inst_nono"
            and solver != "synth_hard"
        ):
            print(solver + " solver not implemented; using synthesis only")
            solver = "synth"

    if len(sys.argv) < 3:
        if solver == "synth_hard":
            print("using hardware")
            simulator = False
            nqubs = backend.configuration().n_qubits
        else:
            print("defaulting to simulator")
            simulator = True
    else:  ## new solver name help separate result files, rather than grouping simulator and hardware.
        ty = str(sys.argv[2])
        if ty == "simulator":
            simulator = True
        elif ty == "hardware":
            simulator = False
            nqubs = backend.configuration().n_qubits
        else:
            print("not valid evaluation type, defaulting to simulator")
            simulator = True

    quantum_instance = backend

    #############
    # INSERT Results from elsewhere, overrides
    #############
    solver = "inst_hard"
    print(solver)
    filename = "jobIDS_ports_" + solver + ".tsv"
    #####
    simulator = False
    nqubs = backend.configuration().n_qubits

    df = pd.read_csv(filename, delimiter="\t", header=None)
    df.columns = ["idx", "job", "port_order"]
    df["qubit_count"] = df["port_order"].str.count(", ") + 1
    cols = ["qubit_count", "idx", "job", "port_order"]
    df = df[cols]
    for i, col in enumerate(cols):
        if i < 2:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(object)
    ##############

    V = [ "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "x", "y", "z", "s", "t", "u", "v", "w", "aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll", "mm", "nn", "oo", "pp", "qq", "rr", "ss", "tt", "uu", "vv", "ww", "xx", "yy", "zz", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak", "al", "am", "an", "ao", "ap", "aq", "ar", "as", "at", "au", "av", "aw", "ax", "ay", "az", "ba", "bc", "bd", "be", "bf", "bg", "bh", "bi", "bj", "bk", "bl", "bm", "bn", "bo", "bp", "bq", "br", "bs", "bt", "bu", "bv", "bw", "bx", "by", "bz", "ca", "cb", "cd",
    ]
    E = [ ("a", "b"), ("a", "c"), ("b", "c"), ("d", "e"), ("d", "f"), ("e", "f"), ("a", "d"), ("b", "e"), ("g", "h"), ("h", "i"), ("g", "i"), ("g", "b"), ("h", "c"), ("j", "k"), ("j", "l"), ("k", "l"), ("j", "c"), ("k", "a"), ("m", "n"), ("n", "o"), ("m", "o"), ("m", "f"), ("n", "f"), ("p", "r"), ("p", "q"), ("q", "r"), ("p", "i"), ("q", "i"), ("x", "p"), ("x", "q"), ("x", "r"), ("y", "m"), ("y", "n"), ("y", "o"), ("z", "j"), ("z", "k"), ("z", "l"), ("s", "n"), ("t", "i"), ("s", "t"), ("s", "u"), ("t", "u"), ("v", "w"), ("v", "aa"), ("w", "aa"), ("v", "a"), ("w", "k"), ("bb", "cc"), ("bb", "dd"), ("cc", "dd"), ("bb", "aa"), ("cc", "s"), ("ee", "ff"), ("ee", "gg"), ("ff", "gg"), ("ee", "h"), ("ff", "r"), ("hh", "ii"), ("ii", "jj"), ("jj", "kk"), ("kk", "hh"), ("jj", "j"), ("kk", "ll"), ("ll", "mm"), ("mm", "nn"), ("nn", "oo"), ("oo", "ll"), ("pp", "qq"), ("qq", "rr"), ("rr", "ss"), ("ss", "pp"), ("qq", "z"), ("tt", "uu"), ("vv", "ww"), ("ww", "xx"), ("uu", "vv"), ("xx", "tt"), ("tt", "ww"), ("vv", "r"), ("xx", "o"), ("yy", "zz"), ("zz", "ab"), ("ab", "ac"), ("ac", "ad"), ("ad", "yy"), ("yy", "ab"), ("yy", "ad"), ("zz", "ac"), ("ab", "b"), ("ad", "l"), ("ae", "af"), ("af", "ag"), ("ag", "ah"), ("ah", "ai"), ("ai", "ae"), ("ae", "ag"), ("ae", "ah"), ("af", "ai"), ("ag", "g"), ("ah", "h"), ("aj", "ak"), ("ak", "al"), ("al", "am"), ("am", "an"), ("an", "aj"), ("aj", "al"), ("aj", "am"), ("al", "ll"), ("am", "mm"), ("ak", "an"), ("ao", "ap"), ("ap", "aq"), ("aq", "ar"), ("ar", "as"), ("at", "au"), ("au", "av"), ("aw", "ax"), ("av", "aw"), ("ax", "ao"), ("as", "at"), ("ao", "ar"), ("ap", "as"), ("aq", "at"), ("ar", "au"), ("as", "av"), ("at", "aw"), ("au", "ax"), ("av", "ao"), ("aw", "ap"), ("ax", "aq"), ("ao", "oo"), ("aq", "qq"), ("as", "ss"), ("au", "uu"), ("aw", "ww"), ("ay", "az"), ("az", "ba"), ("ba", "bc"), ("bc", "bd"), ("bi", "ay"), ("bd", "be"), ("be", "bf"), ("bf", "bg"), ("bg", "bh"), ("bh", "bi"), ("ay", "bc"), ("az", "bd"), ("ba", "be"), ("bc", "bf"), ("bd", "bg"), ("be", "bh"), ("bf", "bi"), ("bg", "ay"), ("bh", "az"), ("bi", "bc"), ("ay", "yy"), ("ba", "aa"), ("bd", "dd"), ("bf", "ff"), ("bh", "hh"), ("bj", "bk"), ("bk", "bl"), ("bl", "bm"), ("bm", "bn"), ("bn", "bo"), ("bo", "bp"), ("bp", "bq"), ("bq", "br"), ("br", "bs"), ("bs", "bj"), ("bj", "bm"), ("bk", "bn"), ("bl", "bo"), ("bm", "bp"), ("bn", "bq"), ("bo", "br"), ("bp", "bs"), ("bq", "bj"), ("br", "bk"), ("bs", "bl"), ("bj", "jj"), ("bl", "al"), ("bn", "an"), ("bp", "ap"), ("bs", "as"), ("bt", "bu"), ("bu", "bv"), ("bv", "bw"), ("bw", "bx"), ("bx", "by"), ("by", "bz"), ("bz", "ca"), ("ca", "cb"), ("cb", "cd"), ("cd", "bt"), ("bt", "bw"), ("bu", "bx"), ("bv", "by"), ("bw", "bz"), ("bx", "ca"), ("by", "cb"), ("bz", "cd"), ("ca", "bt"), ("cb", "bu"), ("cd", "bv"), ("bt", "at"), ("bv", "av"), ("bx", "ax"), ("bz", "az"), ("cb", "ab"),
    ]
    edge_length = [ 3, 8, 13, 18, 23, 28, 37, 42, 47, 52, 57, 62, 67, 72, 80, 90, 100, 110, 135, 160, 185, 210,
    ]
    for i, j in enumerate(
        [ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 37, 41, 45, 50, 55, 60, 65, 75, 85, 95, 105,
        ]
    ):
        # for i, j in enumerate([3]):
        if j > nqubs:
            break
        print("===============\nstart with " + str(j) + " qubits")
        run_graph(
            V[:j],
            E[: edge_length[i]],
            i + 1,
            solver,
            simulator=simulator,
            quantum_instance=quantum_instance,
            nQubs=nqubs,
            color=False,
        )
        print(str(j) + "qubits complete\n===============")

    print("======BEGIN GRAPHS take 2=========")
    E = [ ("a", "b"), ("a", "c"), ("b", "c"), ("d", "e"), ("d", "f"), ("e", "f"), ("a", "d"), ("b", "e"), ("g", "h"), ("h", "i"), ("g", "i"), ("g", "b"), ("h", "c"), ("j", "k"), ("j", "l"), ("k", "l"), ("j", "c"), ("k", "a"), ("i", "l"), ("f", "l"), ("f", "i"), ("j", "d"), ("j", "g"), ("g", "d"), ("k", "e"), ("k", "h"), ("e", "h"), ("l", "a"), ("c", "d"), ("f", "g"), ("i", "j"), ("a", "i"), ("b", "l"), ("c", "f"), ("b", "h"), ("b", "k"), ("e", "g"), ("l", "c"), ("f", "h"), ("i", "k"), ("a", "g"), ("d", "i"), ("j", "e"), ("j", "b"), ("a", "e"), ("c", "g"), ("g", "k"), ("g", "l"), ("d", "h"), ("f", "a"), ("i", "c"), ("d", "k"), ("b", "f"), ("l", "h"), ("b", "i"), ("c", "k"), ("i", "e"), ("f", "j"), ("b", "d"), ("c", "e"), ("f", "k"), ("l", "d"), ("l", "e"),
    ]
    edge_length = [24, 31, 37, 43, 48, 54, 60, 63]
    for i, j in enumerate([12, 12, 12, 12, 12, 12, 12, 12]):
        if j > nqubs:
            break
        print("===============\nstart with " + str(j) + " qubits")
        if i > 2:
            run_graph(
                V[:j],
                E[: edge_length[i]],
                4,
                solver,
                simulator,
                False,
                quantum_instance=quantum_instance,
                nQubs=nqubs,
            )
        else:
            run_graph(
                V[:j],
                E[: edge_length[i]],
                4,
                solver,
                simulator=simulator,
                quantum_instance=quantum_instance,
                nQubs=nqubs,
            )
        print(str(j) + "qubits complete\n===============")

    print("======BEGIN OTHERS=========")
    E = []
    S = {}
    bins = []
    limit = 27
    step = 3
    seed = 6
    random.seed(seed)
    for i in range(3, limit):
        constraints = []
        bins = [str(x) for x in range(i)]
        # for j in range(step):
        #     bins.append(str(i - j))
        #     E.append("v" + str(i - j))
        #     for sub in S:
        #         if int(sub[1:]) % step == 0:
        #             continue
        #         if random.random() < 0.2:
        #             S[sub].append("v" + str(i - j))
        # for j in range(step):
        #     if j == step - 1:
        #         var = []
        #         for k in range(step):
        #             var.append("v" + str(i - k))
        #         S["s" + str(i - j)] = var
        #         continue
        #     S["s" + str(i - j)] = []
        #     for k in range(i):
        #         if random.random() < 0.3:
        #             S["s" + str(i - j)].append("v" + str(k))
        for k in range(len(bins)):
            a = "!" if random.random() < 0.5 else ""
            b = "!" if random.random() < 0.5 else ""
            c = "!" if random.random() < 0.5 else ""
            choice = random.sample(bins, 3)
            constraints.append([a + choice[0], b + choice[1], c + choice[2]])
        if i <= 24:
            with open("3sat_" + solver + "_" + "inputs.dat", "a") as f:
                f.write(str(seed) + "\n")
                # f.write(str(E) + "\n")
                # f.write(str(S) + "\n")
                f.write("".join(["v" + str(x) for x in bins]) + "\n")
                f.write(str(constraints) + "\n")
            print(
                "===============\nstart with "
                + str(len(constraints))
                + " constraints and "
                + str(len(set([i for x in constraints for i in x])))
                + " unique variables"
            )
            run_other(
                E,
                S,
                constraints,
                solver,
                simulator=simulator,
                quantum_instance=quantum_instance,
                p=p,
            )
            print(
                str(str(len(set([i for x in constraints for i in x]))))
                + " unique vars complete\n==============="
            )
