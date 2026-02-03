import os, random
import numpy as np
from mfe import MFE
#from tqdm import trange

def data(n_mode, train_num, test_num, seed=42):
    # X_train:([train_num, dim_zomega], [train_num, dim_zf], [train_num, num_loc, dim_loc])
    # y_train:[train_num, num_loc]
    # X_test:([test_num, dim_zomega], [test_num, dim_zf], [test_num, num_loc, dim_loc])
    # y_test:[test_num, num_loc]
    # reproducibility: set seeds for Python random and NumPy
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    mfe = MFE()
    zdim = n_mode * n_mode
    
    train_num_half = train_num // 2
    test_num_half = test_num // 2
    num_half = train_num_half + test_num_half
    num_samples = 10000
    
    star_data = np.load('./data/star_raw_data.npz')
    zomega_star = np.zeros([num_half, zdim])
    zf_star = np.zeros([num_half, zdim])
    zk_star = np.zeros([num_half, zdim])
    zg_star = np.zeros([num_half, zdim])
    position_star = np.zeros([num_half, num_samples, 2])
    u_star = np.zeros([num_half, num_samples])
    for i in range(num_half):
        print('Processing No. {} ...'.format(i))
        points = star_data['points_{}'.format(i)][:, 0:2]
        line = star_data['line_{}'.format(i)]
        triangle = star_data['triangle_{}'.format(i)]        
        k = star_data['k_{}'.format(i)]
        f = star_data['f_{}'.format(i)]
        g = star_data['g_{}'.format(i)]
        u = star_data['u_{}'.format(i)]
        (zomega_star[i, :], zk_star[i, :], zf_star[i, :], zg_star[i, :]) = mfe.encode(
            points, line, triangle, k, f, g, n_mode)
        
        sample = np.random.choice(points.shape[0], num_samples, replace=True)
        position_star[i, :, :] = points[sample, :]
        u_star[i, :] = u[sample]
        
    annular_data = np.load('./data/annular_raw_data.npz')
    zomega_annular = np.zeros([num_half, zdim])
    zf_annular = np.zeros([num_half, zdim])
    zk_annular = np.zeros([num_half, zdim])
    zg_annular = np.zeros([num_half, zdim])
    position_annular = np.zeros([num_half, num_samples, 2])
    u_annular = np.zeros([num_half, num_samples])
    for i in range(num_half):
        print('Processing No. {} ...'.format(i))
        points = annular_data['points_{}'.format(i)][:, 0:2]
        line = annular_data['line_{}'.format(i)]
        triangle = annular_data['triangle_{}'.format(i)]        
        k = annular_data['k_{}'.format(i)]
        f = annular_data['f_{}'.format(i)]
        g = annular_data['g_{}'.format(i)]
        u = annular_data['u_{}'.format(i)]
        (zomega_annular[i, :], zk_annular[i, :], zf_annular[i, :], zg_annular[i, :]) = mfe.encode(
            points, line, triangle, k, f, g, n_mode)
        
        sample = np.random.choice(points.shape[0], num_samples, replace=True)
        position_annular[i, :, :] = points[sample, :]
        u_annular[i, :] = u[sample]
    
    train_zomega = np.concatenate((zomega_star[:train_num_half, :], zomega_annular[:train_num_half, :]), axis=0)
    train_zk = np.concatenate((zk_star[:train_num_half, :], zk_annular[:train_num_half, :]), axis=0)
    train_zf = np.concatenate((zf_star[:train_num_half, :], zf_annular[:train_num_half, :]), axis=0)
    train_zg = np.concatenate((zg_star[:train_num_half, :], zg_annular[:train_num_half, :]), axis=0)
    train_zfg = np.concatenate((train_zf, train_zg), axis=1)
    train_position = np.concatenate((position_star[:train_num_half, :], position_annular[:train_num_half, :]), axis=0)
    train_u = np.concatenate((u_star[:train_num_half, :], u_annular[:train_num_half, :]), axis=0)
    
    test_zomega = np.concatenate((zomega_star[-test_num_half:, :], zomega_annular[-test_num_half:, :]), axis=0)
    test_zk = np.concatenate((zk_star[-test_num_half:, :], zk_annular[-test_num_half:, :]), axis=0)
    test_zf = np.concatenate((zf_star[-test_num_half:, :], zf_annular[-test_num_half:, :]), axis=0)
    test_zg = np.concatenate((zg_star[-test_num_half:, :], zg_annular[-test_num_half:, :]), axis=0)
    test_zfg = np.concatenate((test_zf, test_zg), axis=1)
    test_position = np.concatenate((position_star[-test_num_half:, :], position_annular[-test_num_half:, :]), axis=0)
    test_u = np.concatenate((u_star[-test_num_half:, :], u_annular[-test_num_half:, :]), axis=0)
    
    np.savez_compressed("./data/X_train.npz", zomega=train_zomega, zk=train_zk, 
                        zfg=train_zfg, position=train_position)
    np.save("./data/y_train.npy", train_u)
    np.savez_compressed("./data/X_test.npz", zomega=test_zomega, zk=test_zk, 
                        zfg=test_zfg, position=test_position)
    np.save('./data/y_test.npy', test_u)
    
def main():
    legendre_mode = 12
    train_num = 9000
    test_num = 1000
    data(legendre_mode, train_num, test_num)
    print('Data prepared!')

if __name__ == '__main__':
    main()
