# flake8: noqa
from .circuit import NodalAnalysis
from .analysis import FrequencyAnalysis
from .netlist import Netlist

from .params import init_mpl

init_mpl()  # define custom matplotlib styling
