import sys

from vqe import run_vqe, run_vqe_repeat
from tq_qnn import run_qnn

to_run = sys.argv[1]

if to_run == "vqe":
    run_vqe()
elif to_run == "vqe_repeat":
    run_vqe_repeat()
elif to_run == "qnn":
    run_qnn()