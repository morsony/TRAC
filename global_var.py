import numpy as np


class sysvars(object):

    cl = 2.99792458e10
    mp = 1.6726e-24            
    me = 9.10938e-28
    re = 2.8179e-13
    q = np.sqrt(me*re*cl**2)
    
    
    sharedstuff = 0
    k = 0
    mingamma = 0
    gammaint = 0
    gamme = 0
    nodensity = 0
    thermal = 0
    eef = 0
    wjm = 0
    
    intergralstuff = 0
    intindex = 0
    intvalues = 0
    
    parameters = 0
    alpha1 = 0
    alpha2 = 0
    beta1 = 0
    beta2 = 0
    kappa1 = 0
    kappa2 = 0
    mu = 0
    delta = 0
    pindex = 0
    v = 0
    kappadelta1 = 0
    kappadelta2 = 0
    bsq = 0
    epsilon = 0
    
    integrationswitch = 0
    intswitch = 0
    
    intparms = 0
    omega = 0
    gamma = 0
    kb = 0
    
    extravar = 0
    tempvar1 = 0
    tempvar2 = 0
    
    thetaparams = 0
    theta = 0
    form = 0
    
    fxyparams = 0
    kx = 0
    fxybar = 0
    
    theta0_arr = 0
    eiso_theta_t = 0
    miso_theta_t = 0
    time_index = 0
    radii_spread = 0
    ntheta = 0
    nphi = 0
    neiso_t = 0
    neiso_theta = 0
    theta_index = 0

    rho_k = 0
    
    shock_thickness = 0.1 * cl
