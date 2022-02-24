import argparse


parser = argparse.ArgumentParser("TITANIC")

parser.add_argument('--device', type=str, default='cpu',
                    help='device to run')

parser.add_argument('--lr', type=float, default=0.5,
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=800,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=20000,
                    help='total epochs')
parser.add_argument('--emb_size', type=int, default=64,
                    help='embedding size')
parser.add_argument('--hidden_nodes', type=int, default=8,
                    help='hidden nodes in every hidden layer')


args,_ = parser.parse_known_args()