import numpy as np
import matplotlib.pyplot as plt
# convert arrays of spherical(phi, theta, r) to cartesian(x,y,z)

def sph_to_cart(sphar, r_or_d='r'):
    
    theta = sphar[:,0]
    phi = sphar[:,1]
    r = sphar[:,2]
    
    if r_or_d == 'd':
        phi = np.radians(phi)
        theta = np.radians(theta)
    
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(phi)
    
    cartar = np.zeros((len(phi),3))
    
    cartar[:,0] = x
    cartar[:,1] = y
    cartar[:,2] = z
    
    return cartar
     
    
#change cartesian to spherical   
   
def cart_to_sph(cartar, r_or_d='r'):
    x = cartar[:,0]
    y = cartar[:,1]
    z = cartar[:,2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y,x)
    phi = np.arctan2(z,np.sqrt(x**2+y**2))   
    
    if r_or_d == 'd':
        phi = np.degrees(phi)
        theta = np.degrees(theta)
    
    sphar = np.zeros((len(x),3))
    
    sphar[:,0] = theta
    sphar[:,1] = phi
    sphar[:,2] = r
    
    return sphar



def rot3d(xyz, r_or_d = 'r', xang = 0, yang = 0, zang = 0, show_rotmap = 'n'):    
    
    if r_or_d == 'd':
        xang = np.radians(xang)
        yang = np.radians(yang)
        zang = np.radians(zang)
        
    if yang == 0:
        cay = 1
        say = 0
    else:
        cay = np.cos(yang)
        say = np.sin(yang)
        
    roty = np.array([[cay, 0, -say],
                     [0, 1, 0],
                     [say, 0, cay]])
                     
    if zang == 0:
        caz = 1
        saz = 0
    else:
        caz = np.cos(zang)
        saz = np.sin(zang)
        
    rotz = np.array([[caz, saz, 0],
                     [-saz, caz, 0],
                     [0,0,1]])
                     
    if xang == 0:
        cax = 1
        sax = 0
    else:
        cax = np.cos(xang)
        sax = np.sin(xang)
        
    rotx = np.array([[1, 0, 0],
                     [0, cax, sax],
                     [0, -sax, cax]])
                     
    rotmat = np.matmul(np.matmul(rotx, rotz), roty)
    
    s = np.shape(xyz)

    if len(s) == 2 and s[1] == 3:
        return np.matmul(xyz, rotmat)
    else:
        return np.matmul(np.transpose(rotmat), xyz)