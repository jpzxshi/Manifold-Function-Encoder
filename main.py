import os, random
import numpy as np
import torch
import learner as ln
from postprocess import postprocess

class Poisson_varying_domains_data(ln.data.Data_MIONet_Cartesian):
    '''Data for 2d Poisson equation.
    '''
    def __init__(self, path):
        super(Poisson_varying_domains_data, self).__init__()
        X_train, X_test = np.load(path + 'X_train.npz'), np.load(path + 'X_test.npz')
        # X_train:([train_num, dim_zomega], [train_num, dim_zf], [train_num, num_loc, dim_loc])
        self.X_train = (X_train['zomega'], X_train['zk'], X_train['zfg'], X_train['position'])
        # y_train:[train_num, num_loc]
        self.y_train = np.load(path + 'y_train.npy')
        # X_test:([test_num, dim_zomega], [test_num, dim_zf], [test_num, num_loc, dim_loc])
        test_num_half = 100 # If an out-of-memory error occurs, the amount of test data can be reduced,
        test_range = [X_test['zomega'].shape[0] // 2 - test_num_half, 
                      X_test['zomega'].shape[0] // 2 + test_num_half]
        self.X_test = (X_test['zomega'][test_range[0]:test_range[1]], 
                       X_test['zk'][test_range[0]:test_range[1]], 
                       X_test['zfg'][test_range[0]:test_range[1]], 
                       X_test['position'][test_range[0]:test_range[1]])
        # y_test:[test_num, num_loc]
        self.y_test = np.load(path + 'y_test.npy')[test_range[0]:test_range[1]]
        
def rel_error(data, net):
    # print test relative error
    net.eval()
    zomega, zk, zfg, position = data.X_test
    with torch.no_grad():
        y_pred = net.predict([zomega, zk, zfg, position])

    y_true = data.y_test
    denom = torch.sqrt(torch.tensor(y_true.shape[1], dtype=y_true.dtype, device=y_true.device))
    rmse_pred = torch.norm(y_true - y_pred, dim=1) / denom
    rmse_true = torch.norm(y_true, dim=1) / denom

    error = torch.mean(rmse_pred / rmse_true)
    
    print('test relative error: ', error.detach().item())


def main():    #### device
    device = 'gpu' # 'cpu' or 'gpu'

    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    #### data
    path = './data/' # the directory of the dataset
    #### MIONet
    mode = 12
    latent_dim = mode ** 2
    sizes = [
        [latent_dim] + [500] * 4,
        [latent_dim] + [500] * 4,
        [2 * latent_dim, -500], # -500 means the last layer is without bias
        [2] + [500] * 4,
        ]
    activation = 'relu'
    #### training
    lr = 1e-5
    iterations = 5000000
    batch_size = 10
    print_every = 1000
    
    callback = rel_error
    #callback = None
    
    training_args = {
        'criterion': 'MSE',
        'optimizer': 'Adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': 'best_only',
        'callback': callback,
        'dtype': 'float',
        'device': device,
    }

    ln.Brain.Start()
    data = Poisson_varying_domains_data(path)
    
    net = ln.nn.MIONet_Cartesian(sizes, activation, bias=False)
    #intervals, dpis = [[0,1]] * 2, [100, 100]
    #net = ln.nn.MIONet_precomp(sizes, intervals, dpis, activation=activation, bias=False)
    
    ln.Brain.Init(data, net)
    ln.Brain.Run(**training_args)
    #change the training argument(s) and run a second round
    #training_args['lr'] = 1e-5
    #ln.Brain.Run(**training_args)
    ln.Brain.Restore()
    ln.Brain.Output(data=False)
    postprocess(data, ln.Brain.Best_model())
    ln.Brain.End()

if __name__ == '__main__':
    main()

