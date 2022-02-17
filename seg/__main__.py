import argparse

from .utils import get_logger, get_cur_time, init_logger, read_yaml
from .solver import SegSolver


logger = get_logger('seg')


parser = argparse.ArgumentParser(description='Segment Main')

parser.add_argument('--cfg_file', default=None, required=True, type=str, 
                     help='config file of the experiment')
parser.add_argument('--log_file', default=f'{get_cur_time()}_expr.log', required=False, type=str, 
                     help='[optional]name of log file, default is <cur_time>_expr.log')


if __name__ == '__main__':
    args = parser.parse_args()

    # Init the logger of this package which is the parent logger of all other loggers.
    # We can't use the logger before this init logger function.
    init_logger('seg', args.log_file)

    cfg = read_yaml(args.cfg_file)

    solver = SegSolver(cfg)
    solver.run()
