# from .circuit import NodalAnalysis
# from . import differential
from .analysis import FrequencyAnalysis
from .netlist import Netlist

from .params import init_mpl

init_mpl()  # define custom matplotlib styling
