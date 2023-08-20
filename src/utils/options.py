import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--distill_ep', type=int, default=1,
                        help="the number of distillation epochs: E")
    parser.add_argument('--global_ep', type=int, default=1,
                        help="the number of global epochs: E")
    parser.add_argument('--pre_ep', type=int, default=20,
                        help="the number of pre-train epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--pre_bs', type=int, default=128,
                        help="local batch size during pre-training: B")
    parser.add_argument('--global_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
     
    parser.add_argument('--gpu', type=int, default=0, help="the index of gpus")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--name', type=str, help='process name')
    parser.add_argument('--cifar100', action='store_true', help="whether to use cifar100")
    parser.add_argument('--mnist', action='store_true', help="whether to use Mnist")
    parser.add_argument('--fmnist', action='store_true', help="whether to use fmnist")
    parser.add_argument('--femnist', action='store_true', help="whether to use fEMNIST(FEMNIST experiments)")
    parser.add_argument('--cifar10cnn', action='store_true', help=" ")
    parser.add_argument('--cifar10cifar100', action='store_true', help="")
    parser.add_argument('--cifar100supcls', action='store_true', help="")
    parser.add_argument('--model', type=str, default='vgg19')#res18
    
    parser.add_argument('--cp_path', type=str, default='../cases')
    parser.add_argument('--restored', action='store_true', help="restored the checkpoint")

    parser.add_argument('--mix', type=int, default=0.5, help="")
    parser.add_argument('--noidd', type=float, default=0.9,
                        help='about noidd')  
    parser.add_argument('--method', type=str, default='local')
    parser.add_argument('--v2', action='store_true', help="whether to use Mnist")
    parser.add_argument('--labelnum', type=int, default=5, help="the index of gpus")                  
    args = parser.parse_args()
    return args

