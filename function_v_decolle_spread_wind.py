import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import interpolate
from global_var import sysvars as gv
import functools
from matplotlib import pyplot as plt

#function to handle energy spreading
# making a memoizable version - Brian, 5/11/2017
# cleaning up commetented out code - Brian, 3/21/2018
#@functools.lru_cache(maxsize=1024)
def e_iso_interp_spread_memoized(r,tobs,theta,phi):
    
    # we don't actually use Tlab for anything, or tobs any more. - Brian, 3/21/2018
    #Tlab = tobs+ r/gv.cl * np.cos(theta)

    #theta_arr0 = theta*180/np.pi*(gv.ntheta)/180
    theta_arr0 = (theta*180/np.pi)*(gv.ntheta)/180-.5
    phi_arr0 = phi*180/np.pi*(gv.nphi-1)/360

#    print('theta0_arr', p['theta0_arr'])
#    print('theta', p['theta'])
#    print('phi_arr', phi_arr)
#    print('theta_arr', theta_arr)

    #print('theta = ',theta*180/np.pi)
    #print('gv.ntheta = ',gv.ntheta)
    #print('theta_arr0 = ',theta_arr0)

    theta_arr = int(theta_arr0)
    phi_arr = int(phi_arr0)
    
    #print('theta_arr = ',theta_arr)

    theta0 = gv.theta0_arr[phi_arr, theta_arr]    

    #print('theta0 = ',theta0)
    
    if np.size(theta0) == 1:
        theta0 = theta0 + np.zeros((np.size(r)))   # np.size(Tlab) origionally - Brian, 3/21/2018

    if np.size(r) == 1:
        r = r + np.zeros((np.size(theta0)))
        

    theta_int0 = (theta0+90) * gv.neiso_theta/180
    theta_int = theta_int0
    
    #print('theta_int = ',theta_int)
    
    interp_time = interpolate.interp1d(gv.radii_spread, np.arange(len(gv.radii_spread)), bounds_error = False, fill_value = 'extrapolate')
    time_int = interp_time(r)

    
    if np.size(theta_int) == 1:
        theta_int = theta_int + np.zeros((np.size(time_int))) 

    theta_int = theta_int.astype(int)
    
    time_int = time_int.astype(int)
    time_int = abs(time_int)

    #print('theta_int = ',theta_int)

#   Vectorizing this procedure, use np.clip - Brian, 2/09/2017

    time_int=np.clip(time_int,0,gv.neiso_t-1)

    #print(time_int,theta_int)
    #print(np.size(time_int),np.size(theta_int))

    eiso0 = gv.eiso_theta_t[time_int, theta_int]
    miso0 = gv.miso_theta_t[time_int, theta_int]

    #print('eiso old = ',eiso0)
    #print('miso old = ',miso0)
    #print('theta_arr0 = ',theta_arr0)
    #print('theta0 = ',theta0)
    
    #print('theta_arr+1 = ',(theta_arr+1)%gv.ntheta)
    theta1 = gv.theta0_arr[phi_arr, (theta_arr+1)%gv.ntheta]    
    #print('gv.theta0_arr = ',gv.theta0_arr[phi_arr,:])

    if np.size(theta1) == 1:
        theta1 = theta1 + np.zeros((np.size(r)))   # np.size(Tlab) origionally - Brian, 3/21/2018

    #print('theta_diff = ',theta_int0-theta_int)
    #print('theta_diff = ',theta_arr0-theta_arr)
    #print('theta_int = ',theta_int)

    eiso1 = gv.eiso_theta_t[time_int, (theta_int+1)%1800]
    miso1 = gv.miso_theta_t[time_int, (theta_int+1)%1800]

    #eiso = eiso0 + (theta-theta0)*(eiso1-eiso0)/(theta1-theta0)
    #miso = miso0 + (theta-theta0)*(miso1-eiso0)/(theta1-theta0)
    eiso = eiso0 + (theta_arr0-theta_arr)*(eiso1-eiso0)
    miso = miso0 + (theta_arr0-theta_arr)*(miso1-miso0)

    #print('eiso new = ',eiso)
    #print('miso new = ',miso)

    leiso0 = np.log(eiso0)
    lmiso0 = np.log(miso0)
    leiso1 = np.log(eiso1)
    lmiso1 = np.log(miso1)
    
    #eiso = eiso0 + (theta-theta_arr0[theta_int])*(eiso1-eiso0)/(theta_arr0[theta_int+1]-theta_arr0[theta_int])
    #miso = miso0 + (theta-theta_arr0[theta_int])*(miso1-eiso0)/(theta_arr0[theta_int+1]-theta_arr0[theta_int])
    leiso = leiso0 + (theta_arr0-theta_arr)*(leiso1-leiso0)
    lmiso = lmiso0 + (theta_arr0-theta_arr)*(lmiso1-lmiso0)

    leiso = np.exp(leiso)
    lmiso = np.exp(lmiso)

    #print('eiso log = ',eiso)
    #print('miso log = ',miso)

    eiso = leiso
    miso = lmiso
    
#    print('eiso = ',eiso,theta0,r,time_int,theta_int)

#    print(theta_int,time_int,Tlab,tobs,r,theta,interp_time(r))
#    print(np.arange(len(gv.radii_spread)))
##    plt.plot(gv.radii_spread,gv.eiso_theta_t[:,0])

#    print('r, theta = ',r,theta,phi,theta0)

#    plt.plot(gv.radii_spread,intrp_time(gv.radii_spread))
#    plt.xscale('log')
##    plt.yscale('log')
#    plt.plot(gv.theta0_arr[:,np.arange(30)*60])
#    print(gv.eiso_theta_t[time_int,:])
#    plt.plot(gv.eiso_theta_t[0,:])
#    plt.yscale('log')
#    plt.show()


    # Try again using radius and direct interpolation rather than time and nearest neighbor. - Brian, 3/21/2018

    theta_arr = theta*180/np.pi*(gv.ntheta-1)/180
    phi_arr = phi*180/np.pi*(gv.nphi-1)/360
    
    theta_arr = int(theta_arr)
    phi_arr = int(phi_arr)
    
    theta0 = gv.theta0_arr[phi_arr, theta_arr]

    if np.size(theta0) == 1:
        theta0 = theta0 + np.zeros((np.size(r)))   # np.size(Tlab) origionally - Brian, 3/21/2018

    if np.size(r) == 1:
        r = r + np.zeros((np.size(theta)))
    
#    theta_int = (theta0+90) * gv.neiso_theta/180
#
#    if np.size(theta_int) == 1:
#        theta_int = theta_int + np.zeros((np.size(time_int)))


#    print('gv.theta_index = ',gv.theta_index)
#    print('gv.radii_spread = ',gv.radii_spread)

#    interp_eiso = interpolate.interp2d(gv.theta_index,gv.radii_spread,gv.eiso_theta_t, kind='linear', bounds_error = False, fill_value = 'extrapolate')
#    eiso = interp_eiso(theta0,r)

#    interp_miso = interpolate.interp2d(gv.theta_index,gv.radii_spread,gv.miso_theta_t, kind='linear', bounds_error = False, fill_value = 'extrapolate')
#    miso = interp_miso(theta0,r)

#    print(gv.theta_index.shape,gv.radii_spread.shape,gv.eiso_theta_t.shape)

    #interp_eiso = interpolate.RectBivariateSpline(gv.radii_spread,gv.theta_index,gv.eiso_theta_t)
#    interp_eiso = interpolate.RegularGridInterpolator((gv.radii_spread,gv.theta_index),gv.eiso_theta_t, method='nearest', bounds_error = False, fill_value = 0)
#    eiso2 = interp_eiso([[theta0],[r]])




    return [eiso,miso]




#function to handle energy spreading
def e_iso_interp_spread(r,p):
    Tlab = p['tobs']+ r/gv.cl * np.cos(p['theta'])
    theta_arr = p['theta']*180/np.pi*(len(p['theta0_arr'][0,:])-1)/180
    phi_arr = p['phi']*180/np.pi*(len(p['theta0_arr'][:,0])-1)/360
#    print('theta0_arr', p['theta0_arr'])
#    print('theta', p['theta'])
#    print('phi_arr', phi_arr)
#    print('theta_arr', theta_arr)
    
#    theta_arr = theta_arr.astype(int)
#    phi_arr = phi_arr.astype(int)
    
    theta0 = p['theta0_arr'][phi_arr, theta_arr]    

    if np.size(theta0) == 1:
        theta0 = theta0 + np.zeros((np.size(Tlab)))
        
    theta_int = (theta0+90) * np.size(p['eiso_theta_t'][0,:])/180
    time_int = (np.log10(Tlab)+2) / 12 * (np.size(p['eiso_theta_t'][:,0]) - 1)
    
    if np.size(theta_int) ==1:
        theta_int = theta_int + np.zeros((np.size(time_int))) 

    theta_int = theta_int.astype(int)
    
    time_int = time_int.astype(int)
    time_int = abs(time_int)

#   Vectorizing this procedure, use np.clip - Brian, 2/09/2017

#    if time_int < 0:
#        time_int = -1
#    if time_int >= np.size(p['eiso_theta_t'][:,0]):
#        time_int = -1
    
    time_int=np.clip(time_int,0,np.size(p['eiso_theta_t'][:,0])-1)

    eiso = p['eiso_theta_t'][time_int, theta_int]
    miso = p['miso_theta_t'][time_int, theta_int]
    
    return [eiso,miso]
    
    
#function to handle mass spreading
def m_iso_interp_spread(r, p):
    Tlab = p['tobs']+ r/gv.cl * np.cos(p['theta'])
    theta_arr = p['theta']*180/np.pi*(len(p['theta0_arr'][0,:])-1)/180
    phi_arr = p['phi']*180/np.pi*(len(p['theta0_arr'][:,0])-1)/360
    
#    theta_arr = theta_arr.astype(int)
#    phi_arr = phi_arr.astype(int)
    
    theta0 = p['theta0_arr'][phi_arr, theta_arr]
    
    theta_int = (theta0+90) * len(p['eiso_theta_t'][0,:])/180
    time_int = (np.log10(Tlab)+2) / 12 * (len(p['eiso_theta_t'][:,0]) - 1)

    if np.size(theta_int) == 1:
        theta_int = theta_int + np.zeros((np.size(time_int)))
        
    theta_int = theta_int.astype(int)
    time_int = time_int.astype(int) 
    time_int = abs(time_int)

#   Vectorizing this procedure, use np.clip - Brian, 2/09/2017

#    if time_int < 0:
#        time_int = -1
#    if time_int >= np.size(p['eiso_theta_t'][:,0]):
#        time_int = -1   
    time_int=np.clip(time_int,0,np.size(p['eiso_theta_t'][:,0])-1)
      
    miso = p['miso_theta_t'][time_int, theta_int]
    
    return miso
    
    
    
#this is dR/v  #-dR*cos(theta)
#this is the function to integrate to find tobs for a given value of bigr
#@functools.lru_cache(maxsize=1024)
def function_tobs2_corrected(r, p):    
    k = p['k']
    rho_k = p['rho_k']
    theta = p['theta']
    
    #need to make p[0] a stucture with all the variables - Brian (5/3/2016)
    
#    e0 = e_iso_interp_spread(r, p)
#    mass0 = m_iso_interp_spread(r, p)
#    e_data = e_iso_interp_spread(r, p)
#    print(r,p['tobs'],p['theta'],p['phi'])
    e_data = e_iso_interp_spread_memoized(r,p['tobs'],p['theta'],p['phi'])
    e0=e_data[0]
    mass0=e_data[1]
    
    m_k = 4 / (3 - k) * np.pi * rho_k * r **(3 - k) #+ mass0 * (1.+1.e-10)
    
    #print('e0,m_k,gv.cl,k = ',e0,m_k,gv.cl,k)
    
    c_k2 = 4 * (e0 / (m_k * gv.cl**2)) * (17 -4*k) / 4
    
    g_ad = 4/3
    
    #new stuff for Deolle formulation for gamma
    #alpha_k = 8*np.pi*(g_ad+1) / (3*(g_ad-1)*(3*g_ad-1)**2)
    alpha_k = 5.60 # For k-2   # 1.25 #0.8840750103008531   # For k=0
    
    e_iso_interp = e0
    mass_init_interp = mass0

    gamma_init = 1+(e_iso_interp/(mass_init_interp*gv.cl**2))
    beta_init = np.sqrt(1-1/gamma_init**2)
    
    x0 = 8*np.pi/ (17 - 4*k)
    y0 = (5 - k)**2/(4*alpha_k) * alpha_k**2

    f0 = e0/(m_k*gv.cl**2) * 4*np.pi/(3-k)
    
    # adjusting f0 for initial mass - Brian, 4/13/2021
            
    gamma_0_init_2 = (e0/(mass0*gv.cl**2))**2
    shock_thickness = gv.shock_thickness #10. * gv.cl #0.1 * gv.cl
            
    #print('test = ',gamma_0_init_2,shock_thickness,e0,m_k-mass0)
    
    #f1 = e0/(m_k * gv.cl**2) * 4*np.pi/(3-k) * ((m_k-mass0)**(2/(3-k))/(gamma_0_init_2*shock_thickness) * m_k/(m_k-mass0))
    #f1 = 1.42*(3*e0/(16*np.pi*rho_k*shock_thickness))**(1/2) * r**(-1) / x0 / gv.cl**(1) #(gv.cl)**(-1/2)
    #f1 = 1.42*(3*e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * r**(-1) * x0 / 2.7 / gv.cl #*(cl)**(1/2)
    f1 = 1.42*(3*e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * r**(-1) * x0 / 2.3 / gv.cl #*(cl)**(1/2)
            
    #f0 = np.minimum(f0,f1)
            
    #print('f0 = ',f0)

    # end adjust f0 - Brian, 4/13/2021


    # Further adjustments fitting to numerical simulations - Brian, 6/9/2021
            
            
    #e1 = 1/2*1/f1*e_iso_interp*1.35**2 * r #* np.sqrt(1-1/gamma_init**2)
    e1 = 0.8*4*np.pi/(3-k)*rho_k * r**(3-k) * gv.cl**2. * gamma_init**2. * 2

    m2b_k = 4/(3-k) * np.pi * rho_k * r**(3-k) + mass_init_interp / np.sqrt(gamma_init)
            
    e2 = e_iso_interp * (1+(mass_init_interp/m2b_k)**.225/gamma_init**1.1125) # between .15 and .275 works okay. .225 seems good. #* m2b_k/mass_init_interp  # Work wiht just *1 for gamma_init=10

    m0b_k = m2b_k #np.where(m2b_k<mass_init_interp/2,m2b_k*2,m0b_k)

    # *** This f0 works for ST16 ***
    f0 = e2/(m0b_k * gv.cl**2) * 4*np.pi/(3-k)

    # Gamma-Beta formulation - Brian, 7/22/2021
    exp1 = .25 #1/3 # and 1.4
    exp2 = .25 #1/3 # and 1.4
    z1_f0 = (x0*1.6*f0**exp1+y0/f0**exp2)/(1.6*f0**exp1+1/f0**exp2)
    v_s_f0 = np.sqrt(f0/(f0+z1_f0))
    gamma_s_f0 = 1./np.sqrt(1-v_s_f0**2)
    gb_f0 = gamma_s_f0**2*v_s_f0**2
    
    
    #print('HERE!!! r,m0b_k,gamma_s_f0')
    #print(r,m0b_k,gamma_s_f0)
    

    #f_init = e_iso_interp/(mass_init_interp * gv.cl**2) * 4*np.pi/(3-k) * np.sqrt(gamma_init) *2 #*(1+1/gamma_init) #*2.

    # New f_init calculation - Brian, 7/6/2021
    dens_ratio_init = (g_ad*gamma_init+1) / (g_ad-1)
    beta_test_init = ((dens_ratio_init*gamma_init) / (dens_ratio_init*gamma_init-1))**(1/3) * beta_init
    gamma_test_init = np.sqrt(1/(1-beta_test_init**2))
    #f_init = gamma_test_init**2 * beta_test_init**2 * x0 * (1+1/gamma_init**.5) #* (1+1/gamma_init**2)
    f_init = gamma_test_init**2 * beta_test_init**2 * 4*np.pi/(3-k)/2 * (1+1/gamma_init**2.) #* (1+1/gamma_init**2)

            
    #f0 = np.where(e1<e_iso_interp, f_init*(1-e1/e_iso_interp)+f0*(e1/e_iso_interp),f0)
    # New interpolation wiht f_init - Brian, 7/6/2021
    #f0 = np.where(e1<e_iso_interp, np.maximum(f_init,f0)*(1-(e1/e_iso_interp)**.1)+f0*(e1/e_iso_interp)**.1,f0)
    f_init_0 = e_iso_interp/(mass_init_interp * gv.cl**2) * 4*np.pi/(3-k) * np.sqrt(gamma_init) * (1+gamma_init**(.5*.225)/gamma_init**1.1125) #*2.

    # New version with x^x - Brian, 7/19/2021
    f0_index_a = (1-(f_init-f_init_0)/f_init)*.8+.1333
    fvar = (np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))
    ffunc = np.abs(fvar)**(.025+(1-np.abs(fvar))**4*.175)
    f0 = np.where(f0>f0_index_a*f_init_0, f_init*(1-ffunc)+f0*ffunc,f0)

    #f0_index_a=0.2
    #print('f0 diff1 = ',f_init_0-f0)
    #print('f_init_0 = ',f_init_0)
    #print('f_init = ',f_init)
    #f0 = np.where(f0>f0_index_a*f_init_0, f_init*(1-(np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))**.05)+f0*((np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))**.05),f0)
    #print('f0 = ',f0)
    
    f0 = np.minimum(f1,f0)
    
    # Gamma-Beta formulation - Brian, 7/22/2021
    

    
    gb_test_init = gamma_test_init**2*beta_test_init**2

    # Use exact solution for piston phase:
    #print('beta_init = ',beta_init)
    with np.nditer([beta_init, None]) as it:
        for beta_solve,gamma_f_init_y in it:
            #print('beta_solve = ',type(beta_solve))
            gamma_f_init_y[...] = function_gamma_f_solve_memoized(float(beta_solve),g_ad)
        gamma_f_init_solve = it.operands[1]
    #print('gamma_f_init_solve = ',gamma_f_init_solve)

    gamma_s_init_solve2 = (gamma_f_init_solve+1) * (g_ad*(gamma_f_init_solve - 1) + 1)**2/(g_ad*(2-g_ad)*(gamma_f_init_solve-1)+2)
    gamma_s_init_solve = np.sqrt(gamma_s_init_solve2)
    v_s_init_solve = np.sqrt(1-1/gamma_s_init_solve2)

    gb_init_solve = gamma_s_init_solve2*(1-1/gamma_s_init_solve2)

    
        # For wind phase:
    #gb_f1 = (1.42*(3*e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * bigr**(-1)/gv.cl/2.7)
            
    gamma_f_f1 = np.sqrt(2.7/.449*(e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * r**(-1)/gv.cl/2+1)
    gamma_s_f1_2 = (gamma_f_f1+1) * (g_ad*(gamma_f_f1 - 1) + 1)**2/(g_ad*(2-g_ad)*(gamma_f_f1-1)+2)
    gb_f1 = gamma_s_f1_2 - 1 

    
    # Updated 10/7/2021.  First, normalize to v_s_init_solve.  Then apply correction - Brian, 10/7/2021
            
    v_s_f1 = np.sqrt(gb_f1/(gb_f1+1))
    v_s_test = v_s_f1 * v_s_init_solve      #* np.sqrt(gb_f0_a/(gb_f0_a+1))**.25
    gb_f1_6 = v_s_test**2/(1-v_s_test**2)   #* (v_s_test/v_s_init_solve)**.5

    gb_f1_8 =  gb_f1_6 * ((v_s_test/v_s_init_solve)**2.*(v_s_test/v_s_init_solve)**.5 + (1-(v_s_test/v_s_init_solve)**2)*(v_s_test/v_s_init_solve)**.75) 
    #gb_f1 = gb_f1_8

    
    
    # For piston phase:
    
    exp1 = .25 #1/3 # and 1.4
    exp2 = .25 #1/3 # and 1.4
    z1_init_0 = (x0*1.6*f_init_0**exp1+y0/f_init_0**exp2)/(1.6*f_init_0**exp1+1/f_init_0**exp2)
    v_s_init_0 = np.sqrt(f_init_0/(f_init_0+z1_init_0))
    gamma_s_init_0 = 1./np.sqrt(1-v_s_init_0**2)
    gb_init_0 = gamma_s_init_0**2*v_s_init_0**2

    # Gamma-Beta formulation - Brian, 7/22/2021
    gb_init = gb_init_solve
    #gb0_index_a = 1-(gb_init-gb_init_0)/gb_init/2
    gb0_index_a = (1-(gb_init-gb_init_0)/gb_init)*.8+.1333
    gbvar = (np.abs(gb_init_0-gb_f0)/((1-gb0_index_a)*gb_init_0))
    gbfunc = np.abs(gbvar)**(.025+(1-np.abs(gbvar))**4*.175)
    #gb_f0_a = gb_f0
    
    # I think this is all wrong for wind, take it out for now - Brian, 1/20/2023
    #gb_f0 = np.where(gb_f0>gb0_index_a*gb_init_0, gb_init*(1-gbfunc)+gb_f0*gbfunc,gb_f0)
    #gb_f0 = np.minimum(gb_f1,gb_f0)

    # end of further adjustments - Brian, 6/9/2021
            
    
    v_s2 = ((f0**2 + 4*f0*x0 - 2*f0*y0 + y0**2)**0.5 - f0 - y0) / (2*x0 - 2*y0)
    v_s = np.sqrt(v_s2) + 1e-24
    #end of new stuff
    
    
    # Trying new interpolation based on numerical results - Brian, 3/19/2021
            
    v_s2_rel = f0/(f0+x0)
    v_s_rel = np.sqrt(v_s2_rel)

    v_s2_nr = f0/(f0+y0)
    v_s_nr = np.sqrt(v_s2_nr)

    v_s2 = (v_s_rel*v_s_rel**2 + v_s_nr*(1-v_s_rel**2))**2
    v_s = np.sqrt(v_s2) + 1e-24
    
    # Gamma-Beta formulation - Brian, 7/22/2021
    v_s2 = gb_f0/(gb_f0+1) #np.sqrt(f0/(f0+z1))
    v_s = np.sqrt(v_s2) + 1.e-24

    # End new interpolation - Brian, 3/19/2021
    
    # Interpolation based on fit for gamma = 4/3, constant density - Brian, 4/13/2021
    '''            
    #exp1 = .3 #1/3 # and 1.4
    #exp2 = .3 #1/3 # and 1.4
    #z1 = (x0*1.5*f0**exp1+y0/f0**exp2)/(1.5*f0**exp1+1/f0**exp2)
    exp1 = .25 #1/3 # and 1.4
    exp2 = .25 #1/3 # and 1.4
    z1 = (x0*1.6*f0**exp1+y0/f0**exp2)/(1.6*f0**exp1+1/f0**exp2)

    v_s2 = f0/(f0+z1)
    v_s = np.sqrt(v_s2) + 1e-24
    #print('1-v_s = ',1-v_s)
    '''        
    # End best fit interpolation - Brian, 4/13/2021

    print('r = ',r,'gamma*beta = ',np.sqrt(gb_f0))
    #plt.plot(r,v_s/np.sqrt(1-v_s**2.))
    #plt.plot(r,np.sqrt(gb_f0))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.show()

    ret = 1/v_s    # - 1*np.cos(theta) # moving this to function_r_tobs
    
    return ret
    
    
    
#returns tobs for a given value of bigr
def function_tobs_integrated_old(r, p):
    
    theta = p['theta']

    # Trying romberg integration instead - Brian, 2/09/2017
    #ret = integrate.quad(function_tobs2_corrected, 0, r, args=(p,), epsrel = 1e-4)[0] / gv.cl
    #ret = integrate.romberg(function_tobs2_corrected, 0, r, args=(p,), rtol = 1e-3, tol=1e-3, divmax=20,vec_func=True)[0] / gv.cl
    
    # TAking care of cos(theta) term here instead - Brian, 3/30/2018
    ret = integrate.quadrature(function_tobs2_corrected, 0, r, args=(p,), rtol = 1e-3, tol=1e-3, maxiter=50, vec_func=True)[0] / gv.cl - r/gv.cl * np.cos(theta)

#    print('romberg ret = ',ret,r,p['tobs'])
#    plt.plot([0,1])
#    plt.show()

    return ret

#used to find bigr at time tobs
def function_r_tobs(r, EXTRA):

    ret = EXTRA['tobs'] - function_tobs_integrated(r, EXTRA)
   
    #print('r_tobs ',r,'tobs_integrated ',ret-EXTRA['tobs'],'tobs ',EXTRA['tobs'])

    return ret


# this is dR/v to be integrated
# In other words, this is dTobs at R
def function_tobs2_smallr(r, p):
    k = p['k']
    rho_k = p['rho_k']
    theta = p['theta']
    
    #need to make p[0] a stucture with all the variables - Brian (5/3/2016)
    
#    e0 = e_iso_interp_spread(r, p)
#    mass0 = m_iso_interp_spread(r, p)
#    e_data = e_iso_interp_spread(r, p)
#    print(r,p['tobs'],p['theta'],p['phi'])
    e_data = e_iso_interp_spread_memoized(r,p['tobs'],p['theta'],p['phi'])
    e0=e_data[0]
    mass0=e_data[1]
    
    m_k = 4 / (3 - k) * np.pi * rho_k * r **(3 - k) #+ mass0 * (1.+1.e-10)
  
    c_k2 = 4 * (e0 / (m_k * gv.cl**2)) * (17 -4*k) / 4
    
    g_ad = 4/3
    
    #new stuff for Deolle formulation for gamma
    #alpha_k = 8*np.pi*(g_ad+1) / (3*(g_ad-1)*(3*g_ad-1)**2)
    alpha_k = 5.60 # For k-2  # 1.25 #0.8840750103008531   # For k=0

    
    e_iso_interp = e0
    mass_init_interp = mass0
    
    gamma_init = 1+(e_iso_interp/(mass_init_interp*gv.cl**2))
    beta_init = np.sqrt(1-1/gamma_init**2)
        
    
    x0 = 8*np.pi/ (17 - 4*k)
    y0 = (5 - k)**2/(4*alpha_k) * alpha_k**2

    f0 = e0/(m_k*gv.cl**2) * 4*np.pi/(3-k)
    
    # adjusting f0 for initial mass - Brian, 4/13/2021
            
    gamma_0_init_2 = (e0/(mass0*gv.cl**2))**2
    shock_thickness = gv.shock_thikcness #10. * gv.cl #0.1 * gv.cl
    
    #print('test = ',gamma_0_init_2,shock_thickness,e0,m_k-mass0)

    
    #f1 = e0/(m_k * gv.cl**2) * 4*np.pi/(3-k) * ((m_k-mass0)**(2/(3-k))/(gamma_0_init_2*shock_thickness) * m_k/(m_k-mass0))
    #f1 = 1.42*(3*e0/(16*np.pi*rho_k*shock_thickness))**(1/2) * r**(-1) / x0 / gv.cl**(1) #(gv.cl)**(-1/2)
    #f1 = 1.42*(3*e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * r**(-1) * x0 / 2.7 / gv.cl #*(cl)**(1/2)
    f1 = 1.42*(3*e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * r**(-1) * x0 / 2.3 / gv.cl #*(cl)**(1/2)
            
#    f0 = np.minimum(f0,f1)
    
    #print('f0 = ',f0)
            
    # end adjust f0 - Brian, 4/13/2021

    
    # Further adjustments fitting to numerical simulations - Brian, 6/9/2021
            
    
       
    #e1 = 1/2*1/f1*e_iso_interp*1.35**2 * r #* np.sqrt(1-1/gamma_init**2)
    e1 = 0.8*4*np.pi/(3-k)*rho_k * r**(3-k) * gv.cl**2. * gamma_init**2. * 2

    m2b_k = 4/(3-k) * np.pi * rho_k * r**(3-k) + mass_init_interp / np.sqrt(gamma_init)
            
    e2 = e_iso_interp * (1+(mass_init_interp/m2b_k)**.225/gamma_init**1.1125) # between .15 and .275 works okay. .225 seems good. #* m2b_k/mass_init_interp  # Work wiht just *1 for gamma_init=10

    m0b_k = m2b_k #np.where(m2b_k<mass_init_interp/2,m2b_k*2,m0b_k)

    # *** This f0 works for ST16 ***
    f0 = e2/(m0b_k * gv.cl**2) * 4*np.pi/(3-k)

    # Gamma-Beta formulation - Brian, 7/22/2021
    exp1 = .25 #1/3 # and 1.4
    exp2 = .25 #1/3 # and 1.4
    z1_f0 = (x0*1.6*f0**exp1+y0/f0**exp2)/(1.6*f0**exp1+1/f0**exp2)
    v_s_f0 = np.sqrt(f0/(f0+z1_f0))
    gamma_s_f0 = 1./np.sqrt(1-v_s_f0**2)
    gb_f0 = gamma_s_f0**2*v_s_f0**2

    
    #f_init = e_iso_interp/(mass_init_interp * gv.cl**2) * 4*np.pi/(3-k) * np.sqrt(gamma_init) *2 #*(1+1/gamma_init) #*2.

    # New f_init calculation - Brian, 7/6/2021
    dens_ratio_init = (g_ad*gamma_init+1) / (g_ad-1)
    beta_test_init = ((dens_ratio_init*gamma_init) / (dens_ratio_init*gamma_init-1))**(1/3) * beta_init
    gamma_test_init = np.sqrt(1/(1-beta_test_init**2))
    #f_init = gamma_test_init**2 * beta_test_init**2 * x0 * (1+1/gamma_init**.5) #* (1+1/gamma_init**2)
    f_init = gamma_test_init**2 * beta_test_init**2 * 4*np.pi/(3-k)/2 * (1+1/gamma_init**2.) #* (1+1/gamma_init**2)

            
    #f0 = np.where(e1<e_iso_interp, f_init*(1-e1/e_iso_interp)+f0*(e1/e_iso_interp),f0)
    # New interpolation wiht f_init - Brian, 7/6/2021
    #f0 = np.where(e1<e_iso_interp, np.maximum(f_init,f0)*(1-(e1/e_iso_interp)**.1)+f0*(e1/e_iso_interp)**.1,f0)
    f_init_0 = e_iso_interp/(mass_init_interp * gv.cl**2) * 4*np.pi/(3-k) * np.sqrt(gamma_init) * (1+gamma_init**(.5*.225)/gamma_init**1.1125) #*2.
    
    # New version with x^x - Brian, 7/19/2021
    f0_index_a = (1-(f_init-f_init_0)/f_init)*.8+.1333
    fvar = (np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))
    ffunc = np.abs(fvar)**(.025+(1-np.abs(fvar))**4*.175)
    f0 = np.where(f0>f0_index_a*f_init_0, f_init*(1-ffunc)+f0*ffunc,f0)

    #f0_index_a=0.2
    #print('f0 diff = ',f_init_0-f0)
    #print('f_init_0 = ',f_init_0)
    #print('f_init = ',f_init)
    #f0 = np.where(f0>f0_index_a*f_init_0, f_init*(1-(np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))**.05)+f0*((np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))**.05),f0)
    #print('f0 = ',f0)
    
    f0 = np.minimum(f1,f0)
    
    # Gamma-Beta formulation - Brian, 7/22/2021
    
    
    gb_test_init = gamma_test_init**2*beta_test_init**2
            
    # Use exact solution for piston phase:
    #print('beta_init = ',beta_init)
    with np.nditer([beta_init, None]) as it:
        for beta_solve,gamma_f_init_y in it:
            gamma_f_init_y[...] = function_gamma_f_solve_memoized(float(beta_solve),g_ad)
        gamma_f_init_solve = it.operands[1]
    #print('gamma_f_init_solve = ',gamma_f_init_solve)
    
    gamma_s_init_solve2 = (gamma_f_init_solve+1) * (g_ad*(gamma_f_init_solve - 1) + 1)**2/(g_ad*(2-g_ad)*(gamma_f_init_solve-1)+2)
    gamma_s_init_solve = np.sqrt(gamma_s_init_solve2)
    v_s_init_solve = np.sqrt(1-1/gamma_s_init_solve2)

    gb_init_solve = gamma_s_init_solve2*(1-1/gamma_s_init_solve2)


            # For wind phase:
    #gb_f1 = (1.42*(3*e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * bigr**(-1)/gv.cl/2.7)
            
    gamma_f_f1 = np.sqrt(2.7/.449*(e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * r**(-1)/gv.cl/2+1)
    gamma_s_f1_2 = (gamma_f_f1+1) * (g_ad*(gamma_f_f1 - 1) + 1)**2/(g_ad*(2-g_ad)*(gamma_f_f1-1)+2)
    gb_f1 = gamma_s_f1_2 - 1 

    
    # Updated 10/7/2021.  First, normalize to v_s_init_solve.  Then apply correction - Brian, 10/7/2021
            
    v_s_f1 = np.sqrt(gb_f1/(gb_f1+1))
    v_s_test = v_s_f1 * v_s_init_solve      #* np.sqrt(gb_f0_a/(gb_f0_a+1))**.25
    gb_f1_6 = v_s_test**2/(1-v_s_test**2)   #* (v_s_test/v_s_init_solve)**.5

    gb_f1_8 =  gb_f1_6 * ((v_s_test/v_s_init_solve)**2.*(v_s_test/v_s_init_solve)**.5 + (1-(v_s_test/v_s_init_solve)**2)*(v_s_test/v_s_init_solve)**.75) 
    #gb_f1 = gb_f1_8

    
    
    # For piston phase:    
    
    exp1 = .25 #1/3 # and 1.4
    exp2 = .25 #1/3 # and 1.4
    z1_init_0 = (x0*1.6*f_init_0**exp1+y0/f_init_0**exp2)/(1.6*f_init_0**exp1+1/f_init_0**exp2)
    v_s_init_0 = np.sqrt(f_init_0/(f_init_0+z1_init_0))
    gamma_s_init_0 = 1./np.sqrt(1-v_s_init_0**2)
    gb_init_0 = gamma_s_init_0**2*v_s_init_0**2

    # Gamma-Beta formulation - Brian, 7/22/2021
    gb_init = gb_init_solve
    #gb0_index_a = 1-(gb_init-gb_init_0)/gb_init/2
    gb0_index_a = (1-(gb_init-gb_init_0)/gb_init)*.8+.1333
    gbvar = (np.abs(gb_init_0-gb_f0)/((1-gb0_index_a)*gb_init_0))
    gbfunc = np.abs(gbvar)**(.025+(1-np.abs(gbvar))**4*.175)
    #gb_f0_a = gb_f0
    
    # I think this is all wrong for wind, take it out for now - Brian, 1/20/2023
    #gb_f0 = np.where(gb_f0>gb0_index_a*gb_init_0, gb_init*(1-gbfunc)+gb_f0*gbfunc,gb_f0)
    #gb_f0 = np.minimum(gb_f1,gb_f0)

        
    # end of further adjustments - Brian, 6/9/2021

    
    v_s2 = ((f0**2 + 4*f0*x0 - 2*f0*y0 + y0**2)**0.5 - f0 - y0) / (2*x0 - 2*y0)
    v_s = np.sqrt(v_s2) + 1e-24
    #end of new stuff
    
    
    # Trying new interpolation based on numerical results - Brian, 3/19/2021
            
    v_s2_rel = f0/(f0+x0)
    v_s_rel = np.sqrt(v_s2_rel)

    v_s2_nr = f0/(f0+y0)
    v_s_nr = np.sqrt(v_s2_nr)

    v_s2 = (v_s_rel*v_s_rel**2 + v_s_nr*(1-v_s_rel**2))**2
    v_s = np.sqrt(v_s2) + 1e-24
    
    # Gamma-Beta formulation - Brian, 7/22/2021
    v_s2 = gb_f0/(gb_f0+1) #np.sqrt(f0/(f0+z1))
    v_s = np.sqrt(v_s2) + 1.e-24

    # End new interpolation - Brian, 3/19/2021
      
    # Interpolation based on fit for gamma = 4/3, constant density - Brian, 4/13/2021
    '''        
    #exp1 = .3 #1/3 # and 1.4
    #exp2 = .3 #1/3 # and 1.4
    #z1 = (x0*1.5*f0**exp1+y0/f0**exp2)/(1.5*f0**exp1+1/f0**exp2)
    exp1 = .25 #1/3 # and 1.4
    exp2 = .25 #1/3 # and 1.4
    z1 = (x0*1.6*f0**exp1+y0/f0**exp2)/(1.6*f0**exp1+1/f0**exp2)

    v_s2 = f0/(f0+z1)
    v_s = np.sqrt(v_s2) + 1e-24
    #print('1-v_s = ',1-v_s)
    '''        
    # End best fit interpolation - Brian, 4/13/2021

    
    ret = 1/v_s
    
    return ret


#returns the intergral of dR/v for a given value of bigr
def function_tobs_integrated_smallr_old(r,p):
    
    # Trying romberg integration instead - Brian, 2/09/2017
    #ret = integrate.quad(function_tobs2_smallr,  0, r, args= (p,) , epsrel = 1e-4)[0]
    ret = integrate.romberg(function_tobs2_smallr,  0, r, args= (p,) , rtol = 1e-3, tol=1.e-3, divmax=20,vec_func=True)[0]
    
    ret = ret/gv.cl    
    
    return ret


#this is used to find bigr at time obs and smallr
def function_r_tobs_smallr(r, EXTRA):
    tobs = EXTRA['tobs']
    smallr = EXTRA['smallr']
    theta = EXTRA['p']['theta']

    p = EXTRA['p']    
    
    ret = tobs - function_tobs_integrated_smallr(r, p) + smallr/gv.cl * np.cos(theta)
    
    return ret
    

def function_s_theta(theta, EXTRA):
    
    EXTRA['p']['theta'] = theta
#    print(theta)
    #print(function_r_tobs(1,EXTRA['p']),function_r_tobs(1e25,EXTRA['p']))
    #r0 = optimize.brentq(function_r_tobs, 1, 1e25, maxiter = 1000, xtol=1e-8, rtol=1e-10, args=(EXTRA['p'],))
    r0 = function_r_integrated(EXTRA['p'])

    ret = -r0 * np.sin(theta)
    
    #print('theta = ',theta,'ret = ',ret)

    return ret    
    
    
#adding funtion to solve for big R at a specified hight (z value), front and back
def function_theta_z(theta, EXTRA):
    z = EXTRA['z']
    
    EXTRA['p']['theta'] = theta
    
    #r0 = optimize.brentq(function_r_tobs, 1, 1e25, maxiter = 1000, xtol=1e-8, rtol=1e-10, args=(EXTRA['p']))
    r0 = function_r_integrated(EXTRA['p'])

    ret = (r0*np.sin(theta)-z)/z
    
    return ret
    
# adding function to find gamma_f from gamma_s
# takes guess of gamma_f as input, along with desired gamma_s, adiabatic, index g_ad
def function_gamma_f_gamma_s(gamma_f, EXTRA):
    gamma_s = EXTRA['gamma_s']
    g_ad = EXTRA['g_ad']
    
    gamma_s_new2 = (gamma_f+1) * (g_ad*(gamma_f - 1) + 1)**2/(g_ad*(2-g_ad)*(gamma_f-1)+2)
    gamma_s_new = np.sqrt(gamma_s_new2)
    ret = gamma_s_new - gamma_s
    
    return ret




#adding funtion to find height at a specificed theta and time (Tlab).  
# This function will need a 2d minimization - Brian, 5/8/2017
def function_s_theta_tobs(data, EXTRA):
    r0=data[0]
    theta=data[1]

    tobs = EXTRA['tobs']


    EXTRA['p']['theta'] = theta

#    r0 = optimize.brentq(function_r_tobs, 1, 1e25, maxiter = 1000, xtol=1e-8, rtol=1e-10, args=(EXTRA['p'],))

    t = function_tobs_integrated(r0, EXTRA['p'])   

    #print(theta,r0,tobs-t,(tobs - t)*(tobs - t) - (r0*np.sin(theta))/gv.cl)

    ret = (tobs - t)*(tobs - t) - (r0*np.sin(theta))/gv.cl
    #ret = -r0 * np.sin(theta)
    
    return ret    




#adding funtion to solve for big R at a specified hight (z value) and time (Tlab), front and back.  
# This function will need a 2d minimization - Brian, 5/5/2017
def function_theta_z_tobs_test(data, EXTRA):
    r0=data[0]
    theta=data[1]

    z = EXTRA['z']
    tobs = EXTRA['tobs']

    EXTRA['p']['theta']=theta
    
    #EXTRA['p']['theta'] = theta
    
    #r0 = optimize.brentq(function_r_tobs, 1, 1e25, maxiter = 1000, xtol=1e-8, rtol=1e-10, args=(EXTRA['p']))
    
    #t = integrate.romberg(function_tobs2_corrected, 0, r, args=(p,), rtol = 1e-3, tol=1e-3, divmax=20,vec_func=True)[0] / gv.cl
    
    t = function_tobs_integrated(r0, EXTRA['p'])

#    print(r0, theta, (tobs - t)*(tobs - t) )

    ret = abs(tobs - t) + abs((r0*np.sin(theta)-z)/z)
    #ret = (tobs - t)*(tobs - t) + ((r0*np.sin(theta)-z)/z)*((r0*np.sin(theta)-z)/z)
    #ret = (r0*np.sin(theta)-z)/z
    
    return ret




#adding funtion to solve for big R at a specified hight (z value) and tobs, front and back
#Doing this with just an integration, not a solve. - Brian, 5/9/2017
def function_theta_z_tobs_old(theta, EXTRA):
    z = EXTRA['z']
    tobs = EXTRA['tobs']
    
    EXTRA['p']['theta'] = theta
    
    r0 = z/np.sin(theta)
    
    t = function_tobs_integrated(r0, EXTRA['p'])

    ret = tobs - t

#    ret = (r0*np.sin(theta)-z)/z
    
    return ret


def function_theta_z_tobs(theta, EXTRA):
    z = EXTRA['z']
    tobs = EXTRA['tobs']

    EXTRA['p']['theta'] = theta

    r0 = z/np.sin(theta)

    bigr = function_smallr_integrated(r0,EXTRA['p'])

    ret = bigr*(1.-1e-9)-r0

    return ret




# Adding function to return a memoized interpolation function of tobs - Brian, 3/30/2018



                #########
                # Setting up memoized tobs(R) here - Brian, 3/30/2018
                # Need to loop over all possible thetas - Brian, 3/30/2018
                ########

@functools.lru_cache()
def function_tobs_memoized(phi):

    theta_arr = np.array([-90.,90.])
    theta_arr = np.insert(theta_arr,1,gv.theta_index)
#    print(theta_arr)
    p = {'k':gv.kb, 'rho_k':gv.rho_k, 'theta':0., 'phi':phi, 'tobs':0., 'theta0_arr':gv.theta0_arr,'eiso_theta_t':gv.eiso_theta_t, 'miso_theta_t':gv.miso_theta_t, 'theta_index':gv.theta_index, 'time_index':gv.time_index}

    num_R_for_mem = 1081
    R_for_mem = 10**(np.arange(num_R_for_mem)/(num_R_for_mem-1)*27-1)
    #  R_for_mem[0] = 0
    int_tobs_arr = np.zeros((np.size(theta_arr),num_R_for_mem-1))
    int_tobs_smallr_arr = np.copy(int_tobs_arr)

#    print('R_for_mem = ',R_for_mem)

    for iii in range(np.size(theta_arr)):
        p['theta'] = (theta_arr[iii])*np.pi/180.

        #print(p['theta'],p['phi'])
        #print('R_for_mem = ',R_for_mem)

        dR_over_vs = function_tobs2_corrected(R_for_mem,p)   # how have dR/vs
        #dR_over_vs = [dR_over_vs[0],dR_over_vs]
	#print(1./dR_over_vs)
        int_tobs_smallr = integrate.cumtrapz(dR_over_vs/gv.cl,R_for_mem) 
        int_tobs = int_tobs_smallr - R_for_mem[1:]/gv.cl * np.cos(p['theta'])
        #print(int_tobs)
        int_tobs_arr[iii,:] = np.copy(int_tobs)
        int_tobs_smallr_arr[iii,:] = np.copy(int_tobs_smallr)

        #print('overall array, theta',p['theta'])

#    print('int_tobs_arr[0,:] = ',int_tobs_arr[0,:])

    ret = interpolate.RectBivariateSpline(theta_arr*np.pi/180.,R_for_mem[1:],int_tobs_arr,kx=1,ky=1)
    
    ret_smallr_interpol = interpolate.RectBivariateSpline(theta_arr*np.pi/180.,R_for_mem[1:],int_tobs_smallr_arr,kx=1,ky=1)

    int_r_arr = np.copy(int_tobs_arr)
    int_smallr_arr = np.copy(int_tobs_arr)
    #tobs_for_mem = 10**(np.arange(num_R_for_mem-1)/(num_R_for_mem-2)*27-8)
    tobs_for_mem = 10**(np.arange(num_R_for_mem-1)/(num_R_for_mem-2)*26-8)
    #print('tobs_for_mem = ',tobs_for_mem)

    for iii in range(np.size(theta_arr)):
        interp_t = interpolate.interp1d(int_tobs_arr[iii,:],R_for_mem[1:])
        #print(iii,'init_tobs_arr[iii,:] = ',int_tobs_arr[iii,:])
        int_r_arr[iii,:] = interp_t(tobs_for_mem)
        interp_smallr_t = interpolate.interp1d(int_tobs_smallr_arr[iii,:],R_for_mem[1:])
        int_smallr_arr[iii,:] = interp_smallr_t(tobs_for_mem)

        #int_r_arr[iii,:] = ret(theta_arr[iii]*np.pi/180.,tobs_for_mem)
        #int_smallr_arr[iii,:] = ret_smallr_interpol(theta_arr[iii]*np.pi/180.,tobs_for_mem)

#    print('test 4e-13 ',ret(theta_arr[0]*np.pi/180.,0.11))
#    print('int_r_arr[0,:] = ',int_r_arr[0,:])

    ret_r = interpolate.RectBivariateSpline(theta_arr*np.pi/180.,tobs_for_mem,int_r_arr,kx=1,ky=1)

    ret_smallr = interpolate.RectBivariateSpline(theta_arr*np.pi/180.,tobs_for_mem,int_smallr_arr,kx=1,ky=1)

    return ret,ret_r,ret_smallr






# A version of function_tobs_integrated using interpolation above.

def function_tobs_integrated_memoized(r,p):

    theta = p['theta']
    #tobs = p['tobs']
    phi = p['phi']
    #print('phi = ',phi)
    interp_func,interp_func_r,interp_func_smallr = function_tobs_memoized(phi)
    ret = interp_func(theta,r)
    
    #print('memoized theta ',theta,'r ',r,'ret ',ret)

    return ret


def function_r_integrated_memoized(p):

    theta = p['theta']
    tobs = p['tobs']
    phi = p['phi']
    #print('phi = ',phi)
    interp_func,interp_func_r,interp_func_smallr = function_tobs_memoized(phi)
    ret = interp_func_r(theta,tobs)
			    
    #print('memoized theta ',theta,'r ',r,'ret ',ret)
    
    #print('tobs = ',tobs,'r = ',ret)

    return ret


def function_smallr_integrated_memoized(smallr,p):

    theta = p['theta']
    tobs = p['tobs']
    phi = p['phi']
    #print('phi = ',phi)
    interp_func,interp_func_r,interp_func_smallr = function_tobs_memoized(phi)
    ret = interp_func_smallr(theta,tobs+smallr/gv.cl*np.cos(theta))

    #print('memoized theta ',theta,'r ',r,'ret ',ret)

    return ret



def function_tobs_integrated(r,p):

    #return function_tobs_integrated_old(r,p)
    return function_tobs_integrated_memoized(r,p)



def function_tobs_integrated_smallr(r,p):

    #return function_tobs_integrated_smallr_old(r,p)
    return function_tobs_integrated_memoized(r,p) + r/gv.cl*np.cos(p['theta'])



def function_r_integrated(p):
    #return optimize.brentq(function_r_tobs, 1, 1e25, args=(p), maxiter = 1000, xtol=1e-8, rtol = 1e-10)
    return function_r_integrated_memoized(p)

def function_smallr_integrated(smallr,p):
    return function_smallr_integrated_memoized(smallr,p)


# Use this function to find gamma - the Lotentz factor behind the shock in the piston phase - memoized version
@functools.lru_cache()
def function_gamma_f_solve_memoized(beta_solve,g_ad):
    return optimize.fsolve(function_gamma_f_init,2,args=(beta_solve,g_ad))
        
# Use this function to find gamma - the Lotentz factor behind the shock in the piston phase
def function_gamma_f_init(gamma,beta0,g_ad):
    return 1-(g_ad*(2 - g_ad)*(gamma-1)+2)/((gamma+1)*(g_ad*(gamma-1)+1)**2) - beta0**2*(gamma/(gamma-(g_ad-1)/(g_ad*gamma+1)))**(2/3)




