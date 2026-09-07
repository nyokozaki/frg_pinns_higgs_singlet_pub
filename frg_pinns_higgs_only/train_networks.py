import argparse
import hyperparams as hp
import singlet_UV_rho_2D as singlet_UV


def parse_block_range(s):
    try:
        start, end = s.split('-')
        start = int(start)
        end = int(end)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid block range: {s!r}. Use format like 0-4"
        )

    if start > end:
        raise argparse.ArgumentTypeError(
            f"invalid block range: {s!r}. start must be <= end"
        )

    return list(range(start, end + 1))


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['rho'])
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of training epochs')

parser.add_argument('--rho-blocks', type=parse_block_range, default=[0, 1, 2, 3, 4],
                    help='contiguous rho block range, e.g. 0-4')
parser.add_argument('--tag', type=str, default='default',
                    help='output directory suffix (writes to ./data_UV_<tag>/). '
                         'Use distinct tags to run multiple trainings concurrently '
                         'from the same folder without clobbering each other.')
parser.add_argument('--no-loss-ext', action='store_true',
                    help='disable loss_extensions (loss_sign_and_mag_couplings) by '
                         'forcing hyperparams.w_sign = hyperparams.w_mag = 0')

args = parser.parse_args()

if args.no_loss_ext:
    hp.w_sign = 0.0
    hp.w_mag = 0.0

if args.mode == 'rho':
    ret = singlet_UV.train(args.epochs, rho_blocks=args.rho_blocks, tag=args.tag)
