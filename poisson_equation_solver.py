import numpy as np

# Solve Possion equation: 
# - \nabla \cdot (k \nabla u) = f in \Omega, u = g on \partial\Omega

def A_e(pts, tri, ij, k):
    # pts: vertices
    # tri: index of triangle
    i = ij[0]
    j = ij[1]
    index_i = tri.index(i)
    index_j = tri.index(j)
    b_i = np.zeros((3))
    b_j = np.zeros((3))
    b_i[index_i] = 1
    b_j[index_j] = 1
    A = pts[tri]
    A[:, 2] = 1
    a_i = np.linalg.solve(A, b_i) # coefficients
    a_j = np.linalg.solve(A, b_j)

    a_k = np.linalg.solve(A, k[tri])
    x0, y0, x1, y1, x2, y2 = A[0][0], A[0][1], A[1][0], A[1][1], A[2][0], A[2][1]
    int_x = (x1 - x0) / 6 + (x2 - x0) / 6 + x0 / 2
    int_y = (y1 - y0) / 6 + (y2 - y0) / 6 + y0 / 2
    J = np.array([[x1 - x0, x2 - x0],
                  [y1 - y0, y2 - y0]])
    det_J = np.abs(np.linalg.det(J))
    int_x = det_J * int_x
    int_y = det_J * int_y
    return np.dot(a_i[:2], a_j[:2]) * np.abs(np.linalg.det(A)) / 2 * a_k[-1] + np.dot(a_i[:2], a_j[:2]) * int_x * a_k[0] \
           + np.dot(a_i[:2], a_j[:2]) * int_y * a_k[1]

# compute (phi_i, f)
def b_e(pts, tri, i, f):
    # take vertices of element and return contribution to b
    index_i = tri.index(i)
    b_i = np.zeros((3))
    b_i[index_i] = 1
    b = np.array([f[tri[0]],f[tri[1]], f[tri[2]]])#.reshape(3,1)
    A = pts[tri]
    A[:, 2] = 1
    s_i = np.linalg.solve(A, b_i)
    s = np.linalg.solve(A, b)
    a_i, t_i, c_i = s_i[0], s_i[1], s_i[2]
    a, t, c = s[0], s[1], s[2]
    x0, y0, x1, y1, x2, y2 = A[0][0], A[0][1], A[1][0], A[1][1], A[2][0], A[2][1]
    J = np.array([[x1 - x0, x2 - x0],
                  [y1 - y0, y2 - y0]])
    det_J = np.abs(np.linalg.det(J))
    int_x2 = x0**2/2 + (x1 - x0)**2/12 + (x2 - x0)**2/12 + (x1 - x0)*(x2 -x0)/12 + x0*(x1 - x0)/3 + x0*(x2 - x0)/3
    int_y2 = y0**2/2 + (y1 - y0)**2/12 + (y2 - y0)**2/12 + (y1 - y0)*(y2 -y0)/12 + y0*(y1 - y0)/3 + y0*(y2 - y0)/3
    int_xy = (x1 - x0)*(y1 - y0)/12 + (x1 - x0)*(y2 - y0)/24 + (x1 - x0)*y0/6 + (x2 - x0)*(y1 - y0)/24 + (x2 - x0)*(y2 - y0)/12 \
                 + (x2 - x0)*y0/6 + (y1 - y0)*x0/6 + (y2 - y0)*x0/6 + x0*y0/2
    int_x2 = det_J * int_x2
    int_y2 = det_J * int_y2
    int_xy = det_J * int_xy
    int_x = (x1-x0)/6 + (x2-x0)/6 + x0/2
    int_y = (y1-y0)/6 + (y2-y0)/6 + y0/2
    int_x = det_J * int_x
    int_y = det_J * int_y
    int_1 = det_J * (1 / 2)
    ans = a_i*a*int_x2 + t_i*t*int_y2 + (a_i*t + t_i*a)*int_xy + (a*c_i + c*a_i)*int_x + (t*c_i + c*t_i)*int_y + c*c_i*int_1
    return ans


def solve_poisson_equation(mesh, k, f, g):
    # - \nabla \cdot (k \nabla u) = f in \Omega, u = g on \partial\Omega
    # mesh : dict{'points', 'elements', 'line'}
    #   points: [N, 2] or [N, 3]
    #   elements: index of vertex of triangles [M, 3]
    #   line: index of boundary points [B, 2]
    # k, f: function values on points [N]
    # g: function values on boundary [B], corresponding to mesh['line'][:, 0]
    points = mesh['points']
    if points.shape[-1] == 2:
        points = np.hstack([points, np.zeros([points.shape[0], 1])])
    elements = mesh['elements']
    line = mesh['line']
    N = points.shape[0]
    #E = elements.shape[0]
    #Loop over elements and assemble LHS and RHS
    boundary = line[:, 0]
    A = np.zeros((N,N))
    b = np.zeros((N))
    g_e = np.zeros((N))
    boundary_list = boundary.tolist()
    for element in elements:
        element_list = element.tolist()
        l = len(element_list) # l = 3
        index_d = [] # [[element_list[0], element_list[0]], [element_list[0], element_list[1]]...]
        for i in range(l):
            a = element_list[i]
            b[a] = b[a] + b_e(points, element_list, a, f)
            for j in range(i,l):
                index_d.append([element_list[i], element_list[j]])

            if a in boundary_list:
                index_a = boundary_list.index(a)
                a_2 = element_list[(i+1)%3]
                a_3 = element_list[(i+2)%3]
                if a_2 not in boundary_list:
                    g_e[a_2] += g[index_a]*A_e(points, element_list, np.array([a, a_2]), k)
                if a_3 not in boundary_list:
                    g_e[a_3] += g[index_a]*A_e(points, element_list, np.array([a, a_3]), k)

        for ij in index_d:
            A[ij[0], ij[1]] = A[ij[0], ij[1]] + A_e(points, element_list, ij, k)
    # restore the whole matrix
    A = A + np.triu(A, k=1).T + np.tril(A, k=-1).T
    # Delete the boundary rows and columns of A and b here
    b = b - g_e
    A = np.delete(A, boundary, axis=0)
    A = np.delete(A, boundary, axis=1)
    b = np.delete(b, boundary, axis=0)
    # solve for the interior points
    u_init = np.linalg.solve(A, b)
    # restore the whole solution u including the zero boundary points
    u = np.zeros(N)
    u[np.delete(np.arange(N), boundary, axis=0)] = u_init
    u[boundary] = g
    return u # u.shape = (N,)

