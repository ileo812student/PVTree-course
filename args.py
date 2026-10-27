import argparse

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Compare results')

    parser.add_argument('--step', type=float, default=1.0)  
    parser.add_argument('--step2d', type=float, default=1.0)
    parser.add_argument('--step2d_no', type=float, default=10000)
    parser.add_argument('--num_point', type=int, default=100)
    parser.add_argument('--x_range', type=int, default=70)
    parser.add_argument('--y_range', type=int, default=14)   
    parser.add_argument('--z_range', type=int, default=80)
    parser.add_argument('--r_ori', type=float, default=0.5)
    parser.add_argument('--ratioE', type=float, default=0.065)
    parser.add_argument('--gamma', type=float, default=3.0)
    parser.add_argument('--ratioQ', type=float, default=0.5)

    parser.add_argument('--num-identities', type=int, default=20,
                        help='number of synthetic identities (must be divisible by 4)')
    parser.add_argument('--samples-per-identity', type=int, default=7)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-root', type=str, default='./pv_pattern_results')
    parser.add_argument('--bezier-root', type=str, default='./bezierpalm')
    parser.add_argument('--resume', action='store_true',
                        help='skip valid existing outputs and deterministically fill missing files')
    parser.add_argument('--force', action='store_true',
                        help='allow overwriting generated files (never enabled by default)')
    parser.add_argument('--workers', type=int, default=1,
                        help='parallel workers for independent identity generation')

    args = parser.parse_args(argv)
    return args
