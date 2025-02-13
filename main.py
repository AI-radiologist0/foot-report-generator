# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------
#
# from model.model import check_model_build, run_visualize_feature_map_func, DeepNetwork
import argparse


from torchinfo import summary

# The import statement below will be refactored soon.
import _init_path
from model import get_fcn
from config import cfg
from config import update_config
from utils.tools import check_device


def parse_args():
    desc = "Pytorch implementation of DeepNetwork"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--cfg',
                        default='experiments/test.yaml',
                        help='experiment configure file name',
                        required=False,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def run_fn(config):
    device = check_device()

    # init model
    model = get_fcn(config, is_train=True)

    model_stat = summary(model,
                         input_size=(1, 784),
                         device='cuda',
                         verbose=1,
                         col_width=16,
                         col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
                         row_settings=["var_names"])


"""main"""


def main():
    args = parse_args()
    update_config(cfg, args)

    # logger
    # create logger

    # run
    run_fn(cfg)
    # check_model_build(args=args)
    # run_visualize_feature_map_func(args)


if __name__ == '__main__':
    main()
