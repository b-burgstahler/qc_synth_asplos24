import pandas as pd
import re
from ast import literal_eval

solvers = [
    "qaoa_nono",
    "qaoa",
    "qaoa_hard",
    "synth_nono",
    "synth",
    "synth_hard",
    "inst_nono",
    "inst",
    "inst_hard",
]

problems = [0, 1, 4]  # minvertcover, maxcut, 3sat
post = '_full'

items = []

for problem in problems:
    for solver in solvers:
        with open(str(problem) + "_" + solver + "_output.dat", "r") as file:
            for line in file:
                line = line.replace("\n", "")
                if line != "":
                    temp = line.replace("[", '"[').replace("]", ']"')
                    x = re.search("\([0-9]*, [0-9]*\)", temp)
                    (start, stop) = x.span()
                    temp = temp.replace(
                        temp[start:stop], '"' + temp[start:stop] + '"', 1
                    )
                    temp_arr = temp.split('"')
                    for i, sub in enumerate(temp_arr):
                        if sub[0] != "[" and sub[0] != "(":
                            temp_arr[i] = sub.replace(", ", "\t")
                        else:
                            temp_arr[i] = '"' + sub + '"'
                    temp = "".join(temp_arr)
                    temp = temp + "\t " + solver + "\n"
                    items.append(temp)

with open("consolidated" + post + ".tsv", "w") as file:
    file.writelines(items)


names = [
    "good_counts",
    "ports",
    "constraints",
    "num_qubits",
    "opt_counts",
    "n_jobs",
    "depths",
    "hard_indicator",
    "z3_soft",
    "soft_counts",
    "validation",
    "problem_id",
    "solver",
]
dtypes = {
    "good_counts": int,
    "ports": int,
    "constraints": int,
    "num_qubits": int,
    "opt_counts": int,
    "n_jobs": int,
    "depths": int,
    "problem_id": int,
}

test = pd.read_csv(
    "consolidated" + post + ".tsv",
    sep="\t",
    quotechar='"',
    quoting=1,
    header=None,
    names=names,
)
for col in ["hard_indicator", "z3_soft", "soft_counts", "validation"]:
    test[col] = test[col].apply(literal_eval)

test = test[(test["num_qubits"].notnull())]
test.to_parquet("consolidated" + post + ".par")
