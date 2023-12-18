import logging

from bqskit.ir.gates import CXGate
from bqskit.compiler import BasePass


logger = logging.getLogger(__name__)
logger.info('customPasses.py')
# logger.setLevel(logging.DEBUG)

# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# fh = logging.FileHandler('passes.log')
# fh.setLevel(logging.DEBUG)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# fh.setFormatter(formatter)

# new_fh = logging.FileHandler('passes_info.log')
# new_fh.setLevel(logging.INFO)
# new_fh.setFormatter(formatter)

# logger.addHandler(new_fh)
# logger.addHandler(ch)
# logger.addHandler(fh)



class PrintCNOTsPass(BasePass):
    def run(self, circuit, data) -> None:
        logger.info("Current CNOT count:", circuit.count(CXGate()))

class PrintGateCountsPass(BasePass):
    def __init__(self,tag:str='') -> None:
        super().__init__()
        self.start = tag

    def run(self,circuit,data,) -> None:
        res = self.start+" Gate Counts\t"
        for gate in circuit.gate_set:
            new = f"{gate}: {circuit.count(gate)}\t"
            res = res + new
        if self.start == 'FINAL':
            # print(res)
            logger.info(res)
        else:
            # print(res)
            logger.debug(res)