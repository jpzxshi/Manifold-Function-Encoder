import os
import numpy as np
import pygmsh
from scipy.interpolate import RegularGridInterpolator#, interp1d
from gaussian_process import Gaussian_process, Gaussian_process_period
from poisson_equation_solver import solve_poisson_equation
from plot import plot_mesh, plot_k, plot_f, plot_g, plot_u

# generate star domains based on radius
def generate_star_domains_with_radius(num):
    intervals = [0, 1]
    mean = 1
    std = 0.2
    length_scale = 0.5
    features = 1001
    period = 1
    scale = 2.8 # 2.5 <= scale <=3, 2.8

    gp = Gaussian_process_period(intervals, mean, std, length_scale, features, period)
    gps = gp.generate(num * 2)
    
    theta = np.linspace(0, 2 * np.pi, 1001)
    x = gps * np.cos(theta)
    y = gps * np.sin(theta)
    scales = 2 * np.maximum(np.max(np.abs(x), axis=-1), np.max(np.abs(y), axis=-1))
    indices = np.where(scales < scale)[0]
    if indices.shape[0] < num:
        raise ValueError
    gps = (gps[indices][:num] / scale)[:, :1000]

    return gps # [num, 1000]

# transform radius to spatial coordinates
def radius_2_space(r):
    theta = np.linspace(0, 2 * np.pi, r.shape[-1], endpoint=False)
    x = r * np.cos(theta) + 0.5
    y = r * np.sin(theta) + 0.5
    points = np.concatenate((x[..., None], y[..., None]), axis=-1)
    return points # [*r.shape, 2]

# generate mesh given polygons
def generate_mesh(polygons, mesh_size):
    polygons = polygons.reshape(-1, polygons.shape[-2], polygons.shape[-1])
    data = {}
    for i in range(polygons.shape[0]):
        print('Generating mesh No. {} ...'.format(i))
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(polygons[i], mesh_size)
            mesh = geom.generate_mesh()
        data['points_{}'.format(i)] = mesh.points[:, :2]
        data['vertex_{}'.format(i)] = mesh.cells[2].data
        data['line_{}'.format(i)] = mesh.cells[0].data
        data['triangle_{}'.format(i)] = mesh.cells[1].data
    return data

# interpolate for function on [0,1]^2
def func_inter_rec(pts, func):
    x = np.linspace(0, 1, func.shape[0])
    y = np.linspace(0, 1, func.shape[1])
    Z = func
    f = RegularGridInterpolator((x, y), Z, method='cubic')
    return f(pts)

def generate_data(n):
    # generate n datapoints
    
    # generate area
    areas = generate_star_domains_with_radius(n) #  [n, 1000]
    areas_sample = areas[:, ::10] # [n, 100]
    areas_xy = radius_2_space(areas_sample) # [n, 100, 2]

    # generate mesh
    meshes = generate_mesh(areas_xy, 0.01)

    # generate random function
    gpk = Gaussian_process([[0, 1]] * 2, 1, 0.2, 0.2, 100)
    k = gpk.generate(n) # [n, 100, 100]
    gpf = Gaussian_process([[0, 1]] * 2, 0, 1, 0.2, 100)
    f = gpf.generate(n) # [n, 100, 100]
    gpg = Gaussian_process([[0, 1]] * 2, 0, 0.02, 0.5, 100)
    g = gpg.generate(n) # [n, 200]
    #gpg = Gaussian_process_period([0, 1], 0, 0.02, 1, 200, period=1)
    #g = gpg.generate(n) # [n, 200]
    
    # interpolation
    k_mesh = []
    f_mesh = []
    g_mesh = []
    for i in range(n):
        pts = meshes['points_{}'.format(i)]
        line = meshes['line_{}'.format(i)]
        pts_b = pts[line[:, 0]]
        #pts_b = pts[line[:, 0]] - get_barycenter(areas_xy[i])
        k_mesh.append(func_inter_rec(pts, k[i]))
        f_mesh.append(func_inter_rec(pts, f[i]))
        g_mesh.append(func_inter_rec(pts_b, g[i]))
        #g_mesh.append(func_inter_boundary(pts_b, g[i]))
    
    # solve
    solutions = []
    for i in range(n):
        print('Solving PDE No. {} ...'.format(i))
        mesh = {'elements': meshes['triangle_{}'.format(i)],
                'points': meshes['points_{}'.format(i)],
                'line': meshes['line_{}'.format(i)]}
        solutions.append(solve_poisson_equation(mesh, k_mesh[i], f_mesh[i], g_mesh[i]))
    
    # data
    data = {}
    data['num'] = n
    for i in range(n):
        data['points_{}'.format(i)] = meshes['points_{}'.format(i)]
        data['vertex_{}'.format(i)] = meshes['vertex_{}'.format(i)]
        data['line_{}'.format(i)] = meshes['line_{}'.format(i)]
        data['triangle_{}'.format(i)] = meshes['triangle_{}'.format(i)]
        data['k_{}'.format(i)] = k_mesh[i]
        data['f_{}'.format(i)] = f_mesh[i]
        data['g_{}'.format(i)] = g_mesh[i]
        data['u_{}'.format(i)] = solutions[i]
    
    return data

def plot(data):
    n = 0
    # plot
    mesh = {}
    mesh['points'] = data['points_{}'.format(n)]
    mesh['vertex'] = data['vertex_{}'.format(n)]
    mesh['line'] = data['line_{}'.format(n)]
    mesh['triangle'] = data['triangle_{}'.format(n)]
    k = data['k_{}'.format(n)]
    f = data['f_{}'.format(n)]
    g = data['g_{}'.format(n)]
    u = data['u_{}'.format(n)]
    
    plot_mesh(mesh)
    plot_k(mesh, k)
    plot_f(mesh, f)
    plot_g(mesh, g)
    plot_u(mesh, u)
    
def save_data(data):
    save_dir = './data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez_compressed(save_dir + '/star_raw_data.npz', **data)
    data = np.load(save_dir + '/star_raw_data.npz')
    #print(data.files)
    for i in range(min(data['num'], 2)):
        print('points_{}'.format(i), data['points_{}'.format(i)].shape, data['points_{}'.format(i)].dtype)
        print('vertex_{}'.format(i), data['vertex_{}'.format(i)].shape, data['vertex_{}'.format(i)].dtype)
        print('line_{}'.format(i), data['line_{}'.format(i)].shape, data['line_{}'.format(i)].dtype)
        print('triangle_{}'.format(i), data['triangle_{}'.format(i)].shape, data['triangle_{}'.format(i)].dtype)
        print('k_{}'.format(i), data['k_{}'.format(i)].shape, data['k_{}'.format(i)].dtype)
        print('f_{}'.format(i), data['f_{}'.format(i)].shape, data['f_{}'.format(i)].dtype)
        print('g_{}'.format(i), data['g_{}'.format(i)].shape, data['g_{}'.format(i)].dtype)
        print('u_{}'.format(i), data['u_{}'.format(i)].shape, data['u_{}'.format(i)].dtype)

def main():
    data = generate_data(5000)
    plot(data)
    save_data(data)

if __name__ == '__main__':
    main()