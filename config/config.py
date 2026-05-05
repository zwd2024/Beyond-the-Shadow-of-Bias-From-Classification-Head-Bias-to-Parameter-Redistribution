import argparse

parser = argparse.ArgumentParser(description='The newly added parameters')
# 遗忘参数(Unlearning parameter)
parser.add_argument('--unlearn-class', type=str, default="3,4,5", help='0,1,...,n')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='')
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                    help='')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='LR',
                    help='')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='Learning rate step gamma (default: 0.7) after 50 epochs')
parser.add_argument('--epochs-or-steps', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
# 上述参数，如果你按照readme文件进行复现，不需要额外添加，只需要添加下面的参数配置
# If you reproduce the above parameters according to the readme file, there is no need to add them separately. Just add the parameter configuration below

# BiasShift
parser.add_argument('--bias-shift-value', type=float, default=-0.01, help="bias-shift value")

# TS-BGRM
parser.add_argument('--unlearning-epochs', type=int, default=8,help="Adjust the first phase epoch of TS-BGRM")
parser.add_argument('--ft-epoch', type=int, default=1, help="Adjust the second phase epoch of FT-BGRM")

# LB-HR
parser.add_argument('--bias-min', type=float, default=-0.01, help="lower bound of bias")
parser.add_argument('--unlearning-loss', type=float, default=1, help="Multiplying the gradient of LB-HR by a coefficient is equivalent to changing the learning rate.")

