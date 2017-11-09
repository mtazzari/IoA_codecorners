import numpy as np
from angdist import angdistcut_cython_par


#helper function to generate points
def gen_points_sphere(n):
    """
    generate random points on sphere
    """
    #angles
    costheta = 2*np.random.rand(n)-1
    theta = np.arccos(costheta)
    phi = 2*np.pi*np.random.rand(n)
    #unit vectors on the sphere
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    vec = np.array([x,y,z]).T
    return vec,theta,phi

# maximum separation: 
maxsep_deg = 1.
cos_maxsep = np.cos(np.deg2rad(maxsep_deg))

# similar to real use case
vec_obj_large,theta_obj_large,phi_obj_large = gen_points_sphere(1000000)
vec_ps_large,theta_ps_large,phi_ps_large = gen_points_sphere(100000)
#make them C contiguous again
vec_obj_large_c = np.ascontiguousarray(vec_obj_large)
vec_ps_large_c = np.ascontiguousarray(vec_ps_large)

result_cython_par_large = angdistcut_cython_par(vec_obj_large_c,vec_ps_large_c,cos_maxsep,num_threads=2)