from .measure_contructor import *
from .samplers import *

avail_measures = {
    "jointm": MeasureJointM,
}

avail_sampler = {
    "dpp": DPPSampler,
    'df': MaxMinDistSampler,
    'cf': UniformCoverageSampler,
    'random': RandomSampler,
    'uniform': UniformSampler,
}