import os.path as osp
import argparse
import pathlib
import os

from rlkit.util.slurm_util import create_sbatch_script
import rlkit

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str)
    parser.add_argument('--label', type=str)
    parser.add_argument('--job_name', type=str, default=None)

    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--gpu', action='store_true')

    parser.add_argument('--mem', type=int, default=6) # 9
    parser.add_argument('--max_hours', type=int, default=504) #168
    parser.add_argument('--partition', type=str, default="titans")
    parser.add_argument('--exclude', type=str, default=None)

    args = parser.parse_args()

    args.no_gpu = (not args.gpu)

    if args.job_name is None:
        args.job_name = "options_{}_{}".format(args.env, args.label)

    args.conda_env = 'rpl'
    args.exp_dir = pathlib.Path(__file__).parent.absolute()
    args.slurm_template = os.path.abspath(
        osp.join(os.path.dirname(rlkit.__file__), os.pardir, 'experiments/rlkit_base_template.sbatch')
    )
    args.python_script = os.path.join(args.exp_dir, "train.py")

    create_sbatch_script(args, use_variants=False)
