from solver import Solver
from util import *
import argparse

args = parse_commandline_args()
solver = Solver(args)
# solve using Ant Colony System
solver.solve()
solver.print_and_save()


