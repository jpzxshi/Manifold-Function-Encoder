import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri   
from mfe import MFE

def postprocess(data, net):
    fig, axes = plt.subplots(4, 3, figsize=(14, 15.5), constrained_layout=True)
    
    star_data = np.load('./data/star_raw_data.npz')
    star_test_no = star_data['num'] - 1
    points = star_data['points_{}'.format(star_test_no)][:, 0:2]
    line = star_data['line_{}'.format(star_test_no)] 
    triangle = star_data['triangle_{}'.format(star_test_no)]      
    k = star_data['k_{}'.format(star_test_no)]
    f = star_data['f_{}'.format(star_test_no)]
    g = star_data['g_{}'.format(star_test_no)]
    u = star_data['u_{}'.format(star_test_no)]
    mode = np.round(np.sqrt(data.X_test_np[0].shape[-1])).astype(np.int64)
    print('mode: ', mode)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    
    tpc = axes[0,0].tripcolor(triangulation, k.ravel(), shading ='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[0,0])
    axes[0,0].set_title('Input:k')
    axes[0,0].set_xticks([])
    axes[0,0].set_yticks([])
    axes[0,0].set_aspect('equal')
    axes[0,0].set_box_aspect(1)
    
    tpc = axes[0,1].tripcolor(triangulation, f.ravel(), shading ='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[0,1])
    axes[0,1].set_title('Input:f')
    axes[0,1].set_xticks([])
    axes[0,1].set_yticks([])
    axes[0,1].set_aspect('equal')
    axes[0,1].set_box_aspect(1)
    
    line_points = points[line[:, 0]]
    tpc = axes[0,2].scatter(line_points[:, 0], line_points[:, 1], c=g, cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[0,2])
    axes[0,2].set_title('Input:g')
    axes[0,2].set_xticks([])
    axes[0,2].set_yticks([])
    axes[0,2].set_aspect('equal')
    axes[0,2].set_box_aspect(1)
    
    #zomega = data.X_test[0][0]
    #zk = data.X_test[1][0]
    #zfg = data.X_test[2][0]
    mfe = MFE()
    zomega, zk, zf, zg = mfe.encode(points, line, triangle, k, f, g, mode)
    zfg = np.concatenate((zf, zg), axis=-1)
    u_rec = net.predict([zomega, zk, zfg, points[:,:2]]).cpu().detach().numpy()
    
    tpc = axes[1,0].tripcolor(triangulation, u_rec.ravel(), shading='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[1,0])
    axes[1,0].set_title('Output:Prediction of u')
    axes[1,0].set_xticks([])
    axes[1,0].set_yticks([])
    axes[1,0].set_aspect('equal')
    axes[1,0].set_box_aspect(1)
    
    tpc = axes[1,1].tripcolor(triangulation, u.ravel(), shading='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[1,1])
    axes[1,1].set_title('Reference of u')
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])
    axes[1,1].set_aspect('equal')
    axes[1,1].set_box_aspect(1)
    
    tpc = axes[1,2].tripcolor(triangulation, np.abs(u.ravel()-u_rec.ravel()), shading='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[1,2])
    axes[1,2].set_title('Error')
    axes[1,2].set_xticks([])
    axes[1,2].set_yticks([])
    axes[1,2].set_aspect('equal')
    axes[1,2].set_box_aspect(1)
    
    rel_error = np.sqrt(np.sum((u.ravel()-u_rec.ravel()) ** 2)) / np.sqrt(np.sum((u.ravel()) ** 2))
    print('reconstruction error: ', rel_error)  
    
    ################################################
    ################################################
    ################################################
    
    annular_data = np.load('./data/annular_raw_data.npz')
    annular_test_no = annular_data['num'] - 1
    points = annular_data['points_{}'.format(annular_test_no)][:, 0:2]
    line = annular_data['line_{}'.format(annular_test_no)] 
    triangle = annular_data['triangle_{}'.format(annular_test_no)]      
    k = annular_data['k_{}'.format(annular_test_no)]
    f = annular_data['f_{}'.format(annular_test_no)]
    g = annular_data['g_{}'.format(annular_test_no)]
    u = annular_data['u_{}'.format(annular_test_no)]
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    
    tpc = axes[2,0].tripcolor(triangulation, k.ravel(), shading ='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[2,0])
    axes[2,0].set_title('Input:k')
    axes[2,0].set_xticks([])
    axes[2,0].set_yticks([])
    axes[2,0].set_aspect('equal')
    axes[2,0].set_box_aspect(1)
    
    tpc = axes[2,1].tripcolor(triangulation, f.ravel(), shading ='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[2,1])
    axes[2,1].set_title('Input:f')
    axes[2,1].set_xticks([])
    axes[2,1].set_yticks([])
    axes[2,1].set_aspect('equal')
    axes[2,1].set_box_aspect(1)
    
    line_points = points[line[:, 0]]
    tpc = axes[2,2].scatter(line_points[:, 0], line_points[:, 1], c=g, cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[2,2])
    axes[2,2].set_title('Input:g')
    axes[2,2].set_xticks([])
    axes[2,2].set_yticks([])
    axes[2,2].set_aspect('equal')
    axes[2,2].set_box_aspect(1)
    
    #zomega = data.X_test[0][500]
    #zk = data.X_test[1][500]
    #zfg = data.X_test[2][500]
    zomega, zk, zf, zg = mfe.encode(points, line, triangle, k, f, g, mode)
    zfg = np.concatenate((zf, zg), axis=-1)
    u_rec = net.predict([zomega, zk, zfg, points[:,:2]]).cpu().detach().numpy()
    
    tpc = axes[3,0].tripcolor(triangulation, u_rec.ravel(), shading='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[3,0])
    axes[3,0].set_title('Output:Prediction of u')
    axes[3,0].set_xticks([])
    axes[3,0].set_yticks([])
    axes[3,0].set_aspect('equal')
    axes[3,0].set_box_aspect(1)
    
    tpc = axes[3,1].tripcolor(triangulation, u.ravel(), shading='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[3,1])
    axes[3,1].set_title('Reference of u')
    axes[3,1].set_xticks([])
    axes[3,1].set_yticks([])
    axes[3,1].set_aspect('equal')
    axes[3,1].set_box_aspect(1)
    
    tpc = axes[3,2].tripcolor(triangulation, np.abs(u.ravel()-u_rec.ravel()), shading='gouraud', cmap='rainbow', rasterized=True)
    fig.colorbar(tpc, ax=axes[3,2])
    axes[3,2].set_title('Error')
    axes[3,2].set_xticks([])
    axes[3,2].set_yticks([])
    axes[3,2].set_aspect('equal')
    axes[3,2].set_box_aspect(1)
    
    rel_error = np.sqrt(np.sum((u.ravel()-u_rec.ravel()) ** 2)) / np.sqrt(np.sum((u.ravel()) ** 2))
    print('reconstruction error: ', rel_error)  

    plt.savefig('./prediction.pdf')
    plt.show()  
    