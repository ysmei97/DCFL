import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default = 'FedAvg')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--task', type=str, default = 'Class_NonIID')
    parser.add_argument('--device', type=str, default = 'cuda:0')
    parser.add_argument('--pretrain', type=str, default = 'False')
    parser.add_argument('--synthetic_number', type=int, default = 600)
    parser.add_argument('--client_number', type=int, default = 20)
    args = parser.parse_args()
    return args