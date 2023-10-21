from utils import current_path

current_path(__file__)


import pprint
import qsharp
from TestCode import AddMain

with qsharp.capture_diagnostics() as diagnostics:
    # print(AddMain.simulate())
    # print(AddMain.simulate_sparse())
    # print(AddMain.toffoli_simulate())
    print(AddMain.simulate_noise())

pprint.pprint(diagnostics)