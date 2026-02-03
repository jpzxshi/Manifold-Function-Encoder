"""
@author: Pengzhan Jin (jpz@pku.edu.cn)
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_mesh(mesh):
    points = mesh['points']
    vertex = mesh['vertex'] # vertexes of (100-)polygon
    line = mesh['line'] # boundary lines of mesh (not polygon)
    triangle = mesh['triangle']
    
    vertex_points = points[vertex[:, 0]] 

    print(points.shape)
    #print(points)
    print(vertex.shape)
    #print(vertex)
    print(line.shape)
    #print(line)
    print(triangle.shape)
    #print(triangle)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    ax.triplot(triangulation, color='black', linewidth=0.5)
    plt.scatter(vertex_points[:, 0], vertex_points[:, 1], s=3)
    plt.title('mesh')
    plt.savefig('mesh.pdf')
    plt.show()
    
def plot_k(mesh, k):
    points = mesh['points']
    triangle = mesh['triangle']
    
    print(points.shape)
    print(triangle.shape)
    print(k.shape)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, k.ravel(), shading ='gouraud', cmap='rainbow')
    fig.colorbar(tpc)
    plt.title('k')
    plt.savefig('k.pdf')
    plt.show()
    
def plot_f(mesh, f):
    points = mesh['points']
    triangle = mesh['triangle']
    
    print(points.shape)
    print(triangle.shape)
    print(f.shape)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, f.ravel(), shading ='gouraud', cmap='rainbow')
    fig.colorbar(tpc)
    plt.title('f')
    plt.savefig('f.pdf')
    plt.show()
    
def plot_g(mesh, g):
    points = mesh['points']
    line = mesh['line']
    line_points = points[line[:, 0]]
    
    print(line.shape)
    print(g.shape)
    
    plt.scatter(line_points[:, 0], line_points[:, 1], c=g, cmap='rainbow')
    plt.colorbar()
    plt.title('g')
    plt.savefig('g.pdf')
    plt.show()
    
def plot_u(mesh, u):
    points = mesh['points']
    triangle = mesh['triangle']
    
    print(points.shape)
    print(triangle.shape)
    print(u.shape)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, u.ravel(), shading ='gouraud', cmap='rainbow')
    fig.colorbar(tpc)
    plt.title('u')
    plt.savefig('u.pdf')
    plt.show()

def main():
    pass

if __name__ == '__main__':
    main()