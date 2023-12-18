# Approximate Synthesis of Parameteric Circuits for Variational Algorithms

Programs to run QAOA based testing of synthesis based searches for alternate representations of the standard QAOA built.

Benchmarks are built by adapting the existing NchooseK benchmark sets, but the workflow does not necessarily depend on NchooseK.

For full operation, you will need to register an account with IBM in order to have access to their physical machines [here](https://quantum-computing.ibm.com/). 


## Install Requirements (Core functionality)
    z3-solver
    'qiskit[visualization]'
    'bqskit[ext]'

## Additional Install requirements ()
    pandas
    pyarrow


## Workflow Instructions:
Each of the shell script is set up run the benchmarks within a particular SLURM environment, but each individual job may be run locally instead.

The file `run_synth_problems.py` contains the core logic for creating and running the benchmarks -- and is where the particular hardware backend of may be chosen (search for 'change machine here').  Any desired customizations can be made adjacent to that line -- setting up your IBM provider, setting the number of QAOA repetitions to run, etc.

Examples of how theses may be run for each of the problem types is included in the shell scripts -- 3 methods: qaoa, synth, and inst ; each with 3 variations nono (noise free simulation), nothing (noisy simulation), and hard (hardware submission)

For *simulator* runs (both ideal and noise free) `run_synth_problems.py` generates '.dat' files of all results.

For *hardware* runs, 'run_synth_problems.py' generates .txt files which record the necessary information to access job data once the jobs complete (jobIDS_ports_*.txt).  However, they must be converted to .tsv files for input into 'analyze_synth_problems.py'

```0	<IBMCircuitJob('cnjnahj4bkt000881fzg')>	['1', '2', '!1', '0', '!2', '_anc1', '_anc2', '_anc3']```

should be replaced with 

```0	cnjnahj4bkt000881fzg	['1', '2', '!1', '0', '!2', '_anc1', '_anc2', '_anc3']```

Now to get the hardware results, `analyze_synth_problems.py` is run.  But first, the same customizations near 'change machine here' should be made. Additionally, some overrides are added under 'insert results from elsewhere' to access each of the `.tsv` files with jobnames.

Now there should be a  `.dat` files for every configuration.  We care only about the files preceeded by 0 (minvert cover), 1 (maxcut), and 4 (3sat).  The others were NchooseK benchmarks which we have ommitted as they require more qubits.


Now, the `.dat` files are consolidated and saved as a single parquet file using `consolidate_data.py`.  This parquet file becomes an input into `make_plots.ipynb` for generating all plots.

It should be carefully noted that while all problems can have plots for correct solutions, *min vertex cover* has both hard and soft constraints, *max cut* has only soft constraints, and *3sat* has only hard constraints.