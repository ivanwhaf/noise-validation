import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, help='batch size', default=64)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-epochs', type=int, help='total epochs', default=100)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()

for i in range(10):
    args.lr *= 0.1
    print(args.lr)
