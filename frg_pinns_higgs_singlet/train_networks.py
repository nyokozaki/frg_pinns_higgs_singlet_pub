# Usage:
#   python train_networks.py rho   --epochs 10000                    # 全5 rhoブロック x 2 tブロック
#   python train_networks.py rho   --epochs 10000 --rho-blocks 2-4    # rhoブロックの一部を学習/再開
#   python train_networks.py sigma --epochs 10000 --sigma-blocks 0-2  # sigma版も同様
#   python train_networks.py 2D    --epochs 10000                     # (irho,isigma)25セル x 2ブロック
#                                                                      # 左/下隣接のみ境界連続性loss、
#                                                                      # ./data_UV_2D/ に保存(--round省略時=1)
#   python train_networks.py 2D --round 2 --epochs 5000               # 2周目: 完成済み1周目グリッドを
#                                                                      # 各セルの初期値としてウォームスタート
#                                                                      # しつつ、右/上隣接との境界連続性loss
#                                                                      # も追加した平滑化パス。
#                                                                      # 参照元(--src-dir)は既定 ./data_UV_2D、
#                                                                      # 保存先(--out-dir)は既定
#                                                                      # ./data_UV_2D_round2 (1周目は上書きしない)。
#                                                                      # 1周目が全25セル分揃っていないと
#                                                                      # ウォームスタートできないセルは
#                                                                      # 警告付きでスキップされる。
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
