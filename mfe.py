import numpy as np
from numpy.polynomial.legendre import legvander, leggauss

def legendre_normalized(x, n):
    x = np.asarray(x)
    y = legvander(2 * x - 1, n-1) * np.sqrt(2 * np.arange(n) + 1)
    return y
    
class MFE:
    def __init__(self):
        super(MFE, self).__init__()
    
    def encode(self, points, line, triangle, k, f, g, mode):
        ##points : [points_num, 2]
        ##line : [line_num, 2]
        ##triangle : [triangle_num, 3]
        ##k : [points_num, ]
        ##f : [points_num, ]
        ##g : [line_num, ]
        
        ####2D
        vertices = points[triangle]  # [triangle_num, 3, 2]
        vec1 = vertices[:, 1, :] - vertices[:, 0, :] 
        vec2 = vertices[:, 2, :] - vertices[:, 0, :]  
        areas = 0.5 * np.abs(vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0])  # [triangle_num, ]
        
        bary_coords = np.array([
            [1/6, 1/6, 2/3],  
            [1/6, 2/3, 1/6],  
            [2/3, 1/6, 1/6] 
        ])  
        weights = np.array([1/3, 1/3 ,1/3])
        
        gauss_points = np.tensordot(bary_coords, vertices, axes=([1], [1])).transpose(1, 0, 2)  # [triangle_num, 6, 2]
        
        leg_x = legendre_normalized(gauss_points[..., 0], mode)  # [triangle_num, 6, mode]
        leg_y = legendre_normalized(gauss_points[..., 1], mode)  # [triangle_num, 6, mode]
        weighted_areas = areas[:, None] * weights[None, :] #/ areas.sum() # [triangle_num, 6]
        
        zomega = np.einsum('kli,klj,kl->ij', leg_x, leg_y, weighted_areas).reshape(-1)
        
        gauss_k = np.dot(k[triangle], bary_coords.T)  # [triangle_num, 6]
        zk = np.einsum('kli,klj,kl,kl->ij', leg_x, leg_y, gauss_k, weighted_areas, optimize=True).reshape(-1)
        
        gauss_f = np.dot(f[triangle], bary_coords.T)  # [triangle_num, 6]
        zf = np.einsum('kli,klj,kl,kl->ij', leg_x, leg_y, gauss_f, weighted_areas, optimize=True).reshape(-1)
        
        ###1D
        line_points = points[line]  # [line_num, 2, 2]
        line_g = np.stack([g, np.roll(g, -1)], axis=1) # [line_num, 2]
        vec = line_points[:, 1, :] - line_points[:, 0, :]  
        lengths = np.sqrt(vec[:, 0] **2 + vec[:, 1]**2 )  # [line_num, ]
        
        gauss_n = 5
        gauss_x, gauss_w = leggauss(gauss_n)    # (gauss_n,)
        t = 0.5 * (gauss_x + 1.0)               # (gauss_n,)         # map to [0,1]
        line_pts0 = line_points[:, 0, :][:, None, :]   # (line_num, 1, 2)
        line_pts1 = line_points[:, 1, :][:, None, :]   # (line_num, 1, 2)
        t_expanded = t[None, :, None]                  # (1, gauss_n, 1)
        
        gauss_points = (1.0 - t_expanded) * line_pts0 + t_expanded * line_pts1  # (line_num, gauss_n, 2)
        gauss_g = (1.0 - t[None, :]) * line_g[:, 0][:, None] + t[None, :] * line_g[:, 1][:, None]  # (line_num, gauss_n)
        weights = 0.5 * gauss_w[None, :] * lengths[:, None] #/ lengths.sum()  # (line_num, gauss_n)
        
        leg_x = legendre_normalized(gauss_points[..., 0], mode)  # (line_num, gauss_n, mode)
        leg_y = legendre_normalized(gauss_points[..., 1], mode)  # (line_num, gauss_n, mode)
        zg = np.einsum('kli,klj,kl,kl->ij', leg_x, leg_y, gauss_g, weights, optimize=True).reshape(-1)

        return zomega, zk, zf, zg

    def decode(self, points, line, zk, zf, zg, mode):
        leg_x = legendre_normalized(points[..., 0], mode) #  [points_num, mode]
        leg_y = legendre_normalized(points[..., 1], mode) #  [points_num, mode] 
        k = np.einsum('ki,kj,ij->k', leg_x, leg_y, zk.reshape([mode, mode]), optimize=True) #  [points_num,] 
        f = np.einsum('ki,kj,ij->k', leg_x, leg_y, zf.reshape([mode, mode]), optimize=True) #  [points_num,] 
        
        line_points = points[line[:, 0]]
        leg_x = legendre_normalized(line_points[..., 0], mode) #  [line_num, mode]
        leg_y = legendre_normalized(line_points[..., 1], mode) #  [line_num, mode] 
        g = np.einsum('ki,kj,ij->k', leg_x, leg_y, zg.reshape([mode, mode]), optimize=True) #  [line_num,] 
        
        return k, f, g 
        
def main():
    pass
    
if __name__ == '__main__':
    main()