from .flow_gen import FlowGenerator

def make_flowgen(cfg):
    return FlowGenerator(cfg)