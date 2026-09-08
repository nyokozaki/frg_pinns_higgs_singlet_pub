# Usage:
#   python train_networks.py rho   --epochs 10000                    # all 5 rho blocks x 2 t blocks
#   python train_networks.py rho   --epochs 10000 --rho-blocks 2-4    # train/resume a subset of the rho blocks
#   python train_networks.py sigma --epochs 10000 --sigma-blocks 0-2  # same for the sigma version
#   python train_networks.py 2D    --epochs 10000                     # (irho,isigma) 25 cells x 2 blocks
#                                                                      # interface-continuity loss with the
#                                                                      # left/bottom neighbors only,
#                                                                      # saved to ./data_UV_2D/ (=round 1 when --round omitted)
#   python train_networks.py 2D --round 2 --epochs 5000               # round 2: a smoothing pass that
#                                                                      # warm-starts each cell from the
#                                                                      # completed round-1 grid, while also
#                                                                      # adding interface-continuity loss with
#                                                                      # the right/top neighbors.
#                                                                      # the source (--src-dir) defaults to ./data_UV_2D,
#                                                                      # the destination (--out-dir) defaults to
#                                                                      # ./data_UV_2D_round2 (round 1 is not overwritten).
#                                                                      # cells that cannot be warm-started because
#                                                                      # round 1 is not complete for all 25 cells
#                                                                      # are skipped with a warning.
import argparse
import singlet_UV_rho_2D as singlet_UV
import singlet_UV_sigma as singlet_UV_sigma


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
parser.add_argument('mode', choices=['rho', 'sigma', '2D'])
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of training epochs')

parser.add_argument('--rho-blocks', type=parse_block_range, default=[0, 1, 2, 3, 4],
                    help='contiguous rho block range, e.g. 0-4')
parser.add_argument('--sigma-blocks', type=parse_block_range, default=[0, 1, 2, 3, 4],
                    help='contiguous sigma block range, e.g. 0-4')

parser.add_argument('--round', type=int, choices=[1, 2], default=1,
                    help='2D mode only: 1 = normal left/bottom-only sweep into ./data_UV_2D/; '
                         '2 = second smoothing pass that warm-starts from a completed round-1 '
                         'grid and also enforces right/top interface continuity, writing to a '
                         'separate directory (default ./data_UV_2D_round2/) so round-1 data is '
                         'never overwritten.')
parser.add_argument('--src-dir', type=str, default=None,
                    help='2D mode with --round 2 only: directory to warm-start from and to read '
                         'right/top neighbor models from (default ./data_UV_2D).')
parser.add_argument('--out-dir', type=str, default=None,
                    help='2D mode with --round 2 only: directory to write round-2 output to '
                         '(default ./data_UV_2D_round2).')

args = parser.parse_args()

if args.mode == 'rho':
    ret = singlet_UV.train(args.epochs, rho_blocks=args.rho_blocks)
elif args.mode == 'sigma':
    ret = singlet_UV_sigma.train(args.epochs, sigma_blocks=args.sigma_blocks)
elif args.mode == '2D':
    if args.round == 1:
        ret = singlet_UV.train_rho_sigma(args.epochs, rho_blocks=args.rho_blocks, sigma_blocks=args.sigma_blocks)
    else:
        src_dir = args.src_dir or "./data_UV_2D"
        out_dir = args.out_dir or "./data_UV_2D_round2"
        ret = singlet_UV.train_rho_sigma_round2(
            args.epochs, rho_blocks=args.rho_blocks, sigma_blocks=args.sigma_blocks,
            src_dir=src_dir, out_dir=out_dir
        )
