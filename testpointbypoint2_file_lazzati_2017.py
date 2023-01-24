import numpy as np
import pickle
import h5py
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import integrate
from scipy.ndimage import filters
from coord_change2 import sph_to_cart
from coord_change2 import cart_to_sph
from coord_change2 import rot3d
from global_var import sysvars as gv
from scipy import signal as sg
from matplotlib import pyplot as plt

# pointbypoint routine to create files for energy and mass

def pointbypoint_powerlaw_varedge_spread(eiso, bins, phi_bins, rot_angle, 
            powerlaw_index, savefile, opening_angle, 
            eiso_off = 'def', center_angle = 1.0, gamma_init = 200., k_index = 0., external_nodensity = 1., spread=1, datafile='0', var='gamma_inf', cutoff=0, efficiency=1, floor=1e43):


                
    # spread=0, no spreading
    # spread=1, diffusion spreading starting at t_theta
    # spread=2, instant spreading a t_theta
    # spread=3, instant spreading at t=0 (always isotropic)
 
    #opening angle is in degrees
    #rot_angle is the observer angle (off-axis) in degrees
    #eiso is eiso at 1 degree
    #powerlaw_index is the index in degree space
    #eiso_off is the (constant) eiso outside the jet angel (opening_angle)
    # center_angle is the size of the jet center, inside which eiso is constant

    # If using an input data file, eiso, gamma_init and opening_angle are used as spreading parameters.




    theta = (np.arange(0,bins) + 0.5) * (180/bins) * (np.pi/180) 
    thetamin = theta - 0.5 * (180/bins) * (np.pi/180)
    thetamax = theta + 0.5 * (180/bins) * (np.pi/180)   
    
    phi_gap = 360/phi_bins/180*np.pi   



    theta_degr = (np.arange(0, bins) + 0.5) * 180/bins - 90
    phi_degr = (np.arange(0, phi_bins)) * 360/phi_bins - 180
      
    theta_arr = np.zeros((phi_bins, bins))
    phi_arr = np.zeros((phi_bins, bins))
    r_arr = np.ones((phi_bins, bins))
        
    for i in range(phi_bins):
        theta_arr[i,:] = theta_degr
        
    for i in range(bins):
        phi_arr[:,i] = phi_degr
            
       
    sph_coord = np.zeros((phi_bins*bins,3))
        
    theta_arr = np.reshape(theta_arr,(1, bins*phi_bins))
    phi_arr = np.reshape(phi_arr, (1, bins*phi_bins))
    r_arr = np.reshape(r_arr, (1, bins*phi_bins))
         
    sph_coord[:,0] = phi_arr
    sph_coord[:,1] = theta_arr
    sph_coord[:,2] = r_arr
        
    cart_coord = sph_to_cart(sph_coord, 'd')
    cart_coord_rot = rot3d(cart_coord, r_or_d='d', yang = rot_angle)  
    sph_coord_rot = cart_to_sph(cart_coord_rot, 'd')
        
    theta_rot = np.zeros((phi_bins, bins))
    theta_rot = np.reshape(theta_rot, bins*phi_bins)
    theta_rot[:] = sph_coord_rot[:,1]
    phi_rot = np.zeros((phi_bins, bins))
    phi_rot = np.reshape(phi_rot, bins*phi_bins)
    phi_rot[:] = sph_coord_rot[:,0]


    theta_rot = np.reshape(theta_rot, (phi_bins, bins))










    # If there is a datafile given, use it to create an energy and mass profile in theta.








    if (datafile != '0'):


        f=np.loadtxt(datafile)

        theta_file1 = f[:,0]
        eiso_file1 = f[:,1]
        gamma_inf_file1 = f[:,5]  # I think this is the initial Lorentz factor that does with energy in column 1 # f[:,5]

        bins_file = len(theta_file1)

        theta_file2 = theta_file1[0:bins_file-1]
        eiso_file2 = eiso_file1[0:bins_file-1]
        gamma_inf_file2 = gamma_inf_file1[0:bins_file-1]

        theta_file = np.concatenate((theta_file1,180-theta_file2[::-1]))
        eiso_file = np.concatenate((eiso_file1,eiso_file2[::-1]))
        gamma_inf_file = np.concatenate((gamma_inf_file1,gamma_inf_file2[::-1]))


        theta_degr = (np.arange(0, bins) + 0.5) * 180/bins - 90
        phi_degr = (np.arange(0, phi_bins)) * 360/phi_bins - 180


        interf_eiso_file = interpolate.interp1d(theta_file, eiso_file, bounds_error = False, fill_value = 'extrapolate')       
        eiso_file_new = interf_eiso_file(theta_degr+90)

        interf_gamma_inf_file = interpolate.interp1d(theta_file, gamma_inf_file, bounds_error = False, fill_value = 'extrapolate')       
        gamma_inf_file_new = interf_gamma_inf_file(theta_degr+90)



        # correct for efficiency of gamma radiation

#        energy_density = energy_density * efficiency
#        gamma_inf = (gamma_inf-1)*efficiency + 1




        binsize=180./bins


        # now correct the mass so that it produces the right gamma_inf

        binned_gamma_inf = gamma_inf_file_new

        g_ad = 4./3.
        gamma_s2 = (binned_gamma_inf+1)*(g_ad*(binned_gamma_inf-1)+1)**2 / (g_ad*(2-g_ad)*(binned_gamma_inf-1)+2)
        alpha_k = 8*np.pi*(g_ad+1)/(3*(g_ad-1)*(3*g_ad-1)**2)
        x0 = 8*np.pi / (17 - 4*k_index)
        y0 = (5-k_index)**2/(4*alpha_k)
        f0 = (gamma_s2-1)*(x0*gamma_s2-x0+y0)/gamma_s2
#        binned_mass_new = binned_energy/(gv.cl**2) * 4.*np.pi/(3-k_index) / f0
#
#        binned_mass = np.copy(binned_mass_new)





        # Now, convert to E_iso, M_iso

        theta_degr = (np.arange(0, bins) + 0.5) * 180/bins - 90
        phi_degr = (np.arange(0, phi_bins)) * 360/phi_bins - 180

        theta_rad=np.arange(bins)/bins*np.pi    # array of theta values of bins in radians (complements phi_rad

#        eiso_theta=4*np.pi * binned_energy/(2.*np.pi*(np.cos(theta_rad)-np.cos(theta_rad+binsize*np.pi/180.)))*9e47
#
#        eiso_theta[np.where(eiso_theta < floor)] = floor
#        eiso_theta[np.where(np.isfinite(eiso_theta) == 0)] = floor

        eiso_theta = eiso_file_new

        #miso_theta=4*np.pi * binned_mass/(2.*np.pi*(np.cos(theta_rad)-np.cos(theta_rad+binsize*np.pi/180.)))*9e47
        miso_theta = eiso_theta/(gv.cl**2) * 4.*np.pi/(3-k_index) / f0

        if gamma_init == 'file':
            gamma_init = binned_gamma_inf
        
        print('gamma_init = ',gamma_init)
        
        plt.plot(gamma_init)
        plt.yscale('log')
        plt.show()
        
        miso_theta = eiso_theta/(gv.cl**2)/(gamma_init-1)


        print('Binning complete!')

        # Use this to set the eiso and gamma_init used for spreading to the values at the opening angle (given as input)
        index_oa_bs = int(np.rint(opening_angle/binsize))
        print(index_oa_bs)
     
        eiso = eiso_theta[index_oa_bs]
        gamma_init = binned_gamma_inf[index_oa_bs]

        g_ad = 4./3.
        gamma_s2 = (gamma_init+1)*(g_ad*(gamma_init-1)+1)**2 / (g_ad*(2-g_ad)*(gamma_init-1)+2)
        alpha_k = 8*np.pi*(g_ad+1)/(3*(g_ad-1)*(3*g_ad-1)**2)
        x0 = 8*np.pi / (17 - 4*k_index)
        y0 = (5-k_index)**2/(4*alpha_k)
        f0 = (gamma_s2-1)*(x0*gamma_s2-x0+y0)/gamma_s2
        miso = eiso/(gv.cl**2) * 4.*np.pi/(3-k_index) / f0    # we need this to get the spreading right

        miso = eiso/(gv.cl**2)/(gamma_init-1)
#        miso = eiso/(gv.cl**2)/(binned_gamma_inf-1)


        print('eiso etc. = ',eiso,miso,gamma_init,opening_angle,eiso_theta[0],eiso_theta[1],eiso_theta[-1])
        print(eiso_file)
#        plt.plot(eiso_theta)
#        plt.plot(binned_gamma_inf)
#        plt.yscale('log')
##        plt.show()  # show all


    # If no input file given, create an energy and mass profile

    else:
    
        if eiso_off == 'def':
            eiso_off = eiso * 1.6e-6
        theta = (np.arange(0,bins) + 0.5) * (180/bins) * (np.pi/180) 
        thetamin = theta - 0.5 * (180/bins) * (np.pi/180)
        thetamax = theta + 0.5 * (180/bins) * (np.pi/180)   
    
        phi_gap = 360/phi_bins/180*np.pi   
    
        test = np.zeros((phi_bins, bins))
        test_invert = np.zeros((phi_bins, bins))
    
        #Adding in variable initial mass loading - Brian 10/01/2015
        mass_test = np.zeros((phi_bins, bins))
        mass_test_invert = np.zeros((phi_bins, bins))
        
        theta_degr = (np.arange(0, bins) + 0.5) * 180/bins - 90
        phi_degr = (np.arange(0, phi_bins)) * 360/phi_bins - 180
        
        theta_arr = np.zeros((phi_bins, bins))
        phi_arr = np.zeros((phi_bins, bins))
        r_arr = np.ones((phi_bins, bins))
        
        for i in range(phi_bins):
            theta_arr[i,:] = theta_degr
            
        for i in range(bins):
            phi_arr[:,i] = phi_degr
            
       
        sph_coord = np.zeros((phi_bins*bins,3))
        
        theta_arr = np.reshape(theta_arr,(1, bins*phi_bins))
        phi_arr = np.reshape(phi_arr, (1, bins*phi_bins))
        r_arr = np.reshape(r_arr, (1, bins*phi_bins))
         
        sph_coord[:,0] = phi_arr
        sph_coord[:,1] = theta_arr
        sph_coord[:,2] = r_arr
        
        cart_coord = sph_to_cart(sph_coord, 'd')
        cart_coord_rot = rot3d(cart_coord, r_or_d='d', yang = rot_angle)  
        sph_coord_rot = cart_to_sph(cart_coord_rot, 'd')
        
        theta_rot = np.zeros((phi_bins, bins))
        theta_rot = np.reshape(theta_rot, bins*phi_bins)
        theta_rot[:] = sph_coord_rot[:,1]
        phi_rot = np.copy(test)
        phi_rot = np.reshape(phi_rot, bins*phi_bins)
        phi_rot[:] = sph_coord_rot[:,0]
    
        eiso_theta = np.copy(theta_degr)
        eiso_theta = eiso * (np.arcsin(np.sin((theta_degr + 90)*np.pi/180))*180/np.pi)**powerlaw_index
    #    w = ((theta_degr + 90) >= opening_angle).nonzero()    
        w = ((90 - abs(theta_degr)) >= opening_angle).nonzero()    
        w = np.reshape(w, np.size(w))
        if len(w) != 0:
            eiso_theta[w] = eiso_off
    #    w2 = ((theta_degr + 90) <= center_angle).nonzero()
        w2 = ((90 - abs(theta_degr)) <= center_angle).nonzero()
        w2 = np.reshape(w2, np.size(w2))
        if len(w2) != 0:
            eiso_theta[w2] = eiso*center_angle**powerlaw_index
    
       
        
        eiso_theta_phi = eiso_theta[((theta_arr + 90) / np.abs(theta_degr[1] - theta_degr[0])).astype(int)]
        theta_rot = np.reshape(theta_rot, (phi_bins, bins))
        test = eiso_theta[((theta_rot + 90)/np.abs(theta_degr[1]-theta_degr[0])).astype(int)]
    
    
# Really this should be a free variable, but assuming initial Lorentz factor is always 200 for now - Brian, 4/20/2017
#    gamma_init = 10. # 200

        mass0 = 1e52 / (gv.cl**2 * gamma_init * (gamma_init - 1))
    
# This should be the correct way to set the initial mass, but it gives quite a low shock Lorentz factor of ~17. - Brian, 4/20/2017
#    miso_theta = eiso_theta/(gv.cl**2 * (gamma_init -1))

# Use this temporarily to get a high initial Lorentz factor - Brian, 4/20/2017
#    miso_theta = eiso_theta/(gv.cl**2 * (gamma_init -1) * gamma_init)

# Corrected to get gamma_init = post shock Lorentz factor - Brian, 5/30/2017
#    miso_theta = eiso_theta/(gv.cl**2 * (gamma_init -1) * gamma_init)*np.sqrt(2.)

# Exact formula to make gamma_int - post shock Lorentz factor - Brian, 6/1/2017
        g_ad = 4./3.
        gamma_s2 = (gamma_init+1)*(g_ad*(gamma_init-1)+1)**2 / (g_ad*(2-g_ad)*(gamma_init-1)+2)
        alpha_k = 8*np.pi*(g_ad+1)/(3*(g_ad-1)*(3*g_ad-1)**2)
        x0 = 8*np.pi / (17 - 4*k_index)
        y0 = (5-k_index)**2/(4*alpha_k)
        f0 = (gamma_s2-1)*(x0*gamma_s2-x0+y0)/gamma_s2
        miso_theta = eiso_theta/(gv.cl**2) * 4.*np.pi/(3-k_index) / f0

        miso_theta = eiso_theta/(gv.cl**2)/(gamma_init-1)

        # pick eiso at edge of jet for spreading! - Brian, 7/18/2017

        binsize=180./bins
        eiso = eiso_theta[np.int(opening_angle/binsize-1)]

        miso = eiso/(gv.cl**2) * 4.*np.pi/(3-k_index) / f0
    
        miso = eiso/(gv.cl**2)/(gamma_init-1)


        mass_test = miso_theta[((theta_rot + 90) / np.abs(theta_degr[1] - theta_degr[0])).astype(int)]
    





    # Now make the eiso_theta_t and miso_theta_t arrays, with appropriate spreading


    #creating Eiso vs. theta and time array
    theta_index = theta_degr
    ntime = 100 
    time_index = 10**(np.arange(ntime+1)/ntime*12 - 2)
    
    tspread = 5e7
    w = (time_index > tspread).nonzero()
    spread_deg = time_index * 0
    spread_deg[w] = np.log(time_index[w]/tspread)/np.sqrt(20)*180/np.pi
    
    spread_deg0 = spread_deg
    tspread0 = tspread
    
    tspread = tspread0/1.5
    w = (time_index > tspread).nonzero()
    spread_deg = time_index*0
    spread_deg[w] = np.log(time_index[w]/tspread) / np.sqrt(20)*180/np.pi
    
    spread_deg = spread_deg*1.6
    w = (spread_deg > 180).nonzero()
    spread_deg[w] = 180
    nnew = 20000
    newindex = np.arange(nnew + 1)/nnew*2

    eiso_theta_t = np.zeros((time_index.size,theta_index.size))
    miso_theta_t = np.zeros((time_index.size,theta_index.size)) 


    # Adding in fill_value = 'extrapolate' to all interpolations here.  May also be needed other places? - Brian, 2/07/2017


    for i in range(np.size(time_index)):
#        nspread = (1 - np.cos(spread_deg[i]/180*np.pi)) * nnew/2  
        nspread = 0    # disable spreading
        interf_eiso_new = interpolate.interp1d(1-np.cos(thetamax), eiso_theta, bounds_error = False, fill_value = 'extrapolate')       
        eiso_new = interf_eiso_new(newindex)
        eiso_smooth_new = filters.uniform_filter(eiso_new, size=nspread, mode='mirror')
        interf_eiso_smooth_theta = interpolate.interp1d(newindex, eiso_smooth_new, bounds_error = False, fill_value = 'extrapolate')
        eiso_smooth_theta = interf_eiso_smooth_theta(1 - np.cos(thetamax))

        eiso_theta_t[i,:] = eiso_smooth_theta
        interf_miso_new = interpolate.interp1d(1 - np.cos(thetamax), miso_theta, bounds_error = False, fill_value = 'extrapolate')
        miso_new = interf_miso_new(newindex)
        miso_smooth_new = filters.uniform_filter(miso_new, size=nspread, mode='mirror')
        interf_miso_smooth_theta = interpolate.interp1d(newindex, miso_smooth_new, bounds_error = False, fill_value = 'extrapolate')
        miso_smooth_theta = interf_miso_smooth_theta(1 - np.cos(thetamax))
        miso_theta_t[i,:] = miso_smooth_theta



# Try spreading by diffusion - Brian, 7/06/2017


# First, find the sperical harmonic coefficients that add up to eiso_theta
    num_l = 501 #2001
    a = np.zeros(num_l)
    x = (90+theta_degr)*np.pi/180

    for ll in range(0,num_l,2):
       c = np.zeros(ll+1)
       c[-1] = 1
       values = 2*np.pi*np.sin(x) * np.sqrt((2*ll+1)/(4*np.pi)) * np.polynomial.legendre.legval(np.cos(x),c) * eiso_theta
#       a[ll] = np.sum(values)
#       a[ll] = integrate.trapz(values,x)
       a[ll] = integrate.trapz(np.append(values,values[0]),np.append(x,np.pi))
       print(ll,a[ll])


# Find the deceleration radius? - Brian, 7/07/2017
# For the center of the jet

    mass = eiso/gv.cl**2 - miso
    radius = (mass/(external_nodensity*gv.mp * 4*np.pi/(3-k_index)))**(1/(3-k_index))
#    gamma = eiso/gv.cl**2/mass + 1
    gamma = np.sqrt(eiso/gv.cl**2/mass) + 1
    v_spread = gv.cl/np.sqrt(3)/gamma
    t_spread = np.pi/2*radius/v_spread
    t_decel = radius/gv.cl/(1-1/(16.*gamma**2.))
    t_decel_2 = radius/gv.cl/(1-1/(1.*gamma**2.))
    print('mass = ',mass,gamma,radius,t_spread,t_decel,t_decel_2)

    gamma_theta = 1/np.sin(opening_angle*np.pi/180)
#    mass_theta = eiso/gv.cl**2/(gamma_theta - 1)
#    mass_theta = eiso/gv.cl**2/(gamma_theta - 1)**2
    mass_theta = eiso/gv.cl**2/(gamma_theta)**2
    radius_theta = (mass_theta/(external_nodensity*gv.mp * 4*np.pi/(3-k_index)))**(1/(3-k_index))
    v_spread_theta = gv.cl/np.sqrt(3)/gamma_theta
    t_spread_theta = np.pi/2*radius_theta/v_spread_theta
    t_theta = radius_theta/gv.cl/(1-1/(16.*gamma_theta**2.))
    print('mass_theta = ',mass_theta,gamma_theta,radius_theta,t_spread_theta,t_theta)

    gamma_theta_all = 1/(np.sin((theta_degr+90)*np.pi/180))
#    mass_theta = eiso/gv.cl**2/(gamma_theta - 1)
#    mass_theta_all = eiso_theta/gv.cl**2/(gamma_theta_all - 1)**2
    mass_theta_all = eiso_theta/gv.cl**2/(gamma_theta_all)**2
    radius_theta_all = (mass_theta_all/(external_nodensity*gv.mp * 4*np.pi/(3-k_index)))**(1/(3-k_index))
    v_spread_theta_all = gv.cl/np.sqrt(3)/gamma_theta_all
    t_spread_theta_all = np.pi/2*radius_theta_all/v_spread_theta_all
    t_theta_all = radius_theta_all/gv.cl/(1-1/(16.*gamma_theta_all**2))

#    plt.plot(theta_degr+90,radius_theta_all)
#    plt.yscale('log')
#    plt.show()

    #radii=10.**(np.arange(31)/30*5)*radius_theta/1e1
    radii=10.**(np.arange(31)/30*6)*radius_theta/1e2 + radius_theta
#    radii=np.concatenate((radii,[1e30]),axis=0)
    mass_r = external_nodensity*gv.mp * 4.*np.pi/(3.-k_index) * radii**(3.-k_index) + miso
#    gamma_r = eiso/gv.cl**2./mass_r + 1
    gamma_r = np.sqrt(eiso/gv.cl**2./mass_r) + 1
#    t_r = radii/gv.cl/(1.-1./(16.*gamma_r**2.))
    t_r = radii/gv.cl*(1.+1./(16.*(gamma_r-1)**2.))
#    v_spread_r = gv.cl/np.sqrt(3)/gamma_r
    v_spread_r = gv.cl/np.sqrt(3)/gamma_r * np.sqrt(1-1/gamma_r**2)
    t_spread_r = np.pi/2*radii/v_spread_r
    print('t_r = ',t_r)
    print('radii = ',radii)
    print('gamma_r = ',gamma_r)

#    np.set_printoptions(threshold=np.nan)
#    print('theta_rot = ',theta_rot[0,:])
#    print('theta_rot shape = ',theta_rot.shape)
#    plt.plot(theta_rot[:,np.arange(30)*60])
#    plt.show()


    print('sum = ',np.sum(eiso_theta[0:900]*(np.sin(x[1:901])-np.sin(x[0:900]))**2),eiso_theta[0]*(1-np.cos(opening_angle*np.pi/180)))

    l = np.arange(num_l)
    c = np.zeros(num_l)
    c[:] = 1
    alpha=1
    t = .001 #1e-5
    eiso_theta_test = np.zeros(len(eiso_theta))

    #for ang in range(len(theta_degr)):
    #    eiso_theta_test[ang] = np.sum(np.exp(-l*(l+1)*alpha*t) * a * np.sqrt((2*l+1)/(4*np.pi)) * np.polynomial.legendre.legval(np.cos(x[ang]),c))

# Now, diffuse by the right amount for time t - Brian, 7/06/2017
#    for ll in range(0,num_l,1):
#       c = np.zeros(ll+1)
#       c[-1] = 1
##       print(np.polynomial.legendre.legval(np.cos(x),c))
#       values = np.exp(-ll*(ll+1)*alpha*t) * a[ll] * np.sqrt((2*ll+1)/(4*np.pi)) * np.polynomial.legendre.legval(np.cos(x),c)
#       eiso_theta_test = eiso_theta_test + values

    #alpha = 0.01
    #t=t_decel/t_spread
    #t=t_theta/t_spread_theta

#    alpha = 0.005 #0.02 #.01  ; seems okay?
    alpha = .5 #.1
    ang95=np.zeros(len(t_spread_r))
    ang90=np.zeros(len(t_spread_r))
    ang50=np.zeros(len(t_spread_r))
    eiso_theta_diffuse=np.zeros((t_r.size,theta_index.size))
    for tt in range(len(t_spread_r)):
#        t=np.max([0,t_r[tt]/t_theta-.5])  # -1.])
#        t=np.max([0,t_r[tt]/t_theta])  # -1.])  # seems okay?
        t=np.max([0,(t_r[tt]-t_theta)/t_spread_r[tt]])  # -1.])
#        t=np.max([0,(t_r[tt])/t_spread_r[tt]])  # -1.])
        eiso_theta_test = np.zeros(len(eiso_theta))
        for ll in range(0,num_l,2):
            c = np.zeros(ll+1)
            c[-1] = 1
#       print(np.polynomial.legendre.legval(np.cos(x),c))
            values = np.exp(-ll*(ll+1)*alpha*t) * a[ll] * np.sqrt((2*ll+1)/(4*np.pi)) * np.polynomial.legendre.legval(np.cos(x),c)
            eiso_theta_test = eiso_theta_test + values
        eiso_theta_diffuse[tt,:]=eiso_theta_test[:]
#        plt.plot(theta_degr[0:900]+90,eiso_theta_test[0:900])
        energy=np.cumsum(eiso_theta_test[0:900]*(-np.cos(x[1:901])+np.cos(x[0:900])))
        ener_i = (np.abs((energy-energy[-1]*.95))).argmin()
        ang95[tt]=x[ener_i]
        ener_i = (np.abs((energy-energy[-1]*.90))).argmin()
        ang90[tt]=x[ener_i]
        ener_i = (np.abs((energy-energy[-1]*.50))).argmin()
        ang50[tt]=x[ener_i]
#        print(t,radii[tt],t_r[tt]/t_theta,gamma_r[tt], x[ener_i],t_r[tt]-radii[tt]/gv.cl)

#    print(t,theta[0:10],theta[-10:-1])
#    plt.show()

    print('radii = ',radii)

    spread_diffuse = 1
    if spread_diffuse == 1:
        time_index2=np.zeros(len(t_r)+5)
        time_index2[0]=0
        time_index2[1]=t_theta/4.
        time_index2[2]=t_theta/2.
        time_index2[3]=t_theta
        time_index2[4:-1]=t_r[:]
        time_index2[-1]=1e30
        radii2=np.zeros(len(t_r)+5)
        radii2[0]=0
        radii2[1]=radius_theta/4.
        radii2[2]=radius_theta/2.
        radii2[3]=radius_theta
        radii2[4:-1]=radii[:]
        radii2[-1]=1e30
        print('time_index = ',time_index)
        eiso_theta_t2 = np.zeros((time_index2.size,theta_index.size))
        miso_theta_t2 = np.zeros((time_index2.size,theta_index.size)) 

        eiso_theta_t2[0,:]=eiso_theta[:]
        eiso_theta_t2[1,:]=eiso_theta[:]
        eiso_theta_t2[2,:]=eiso_theta[:]
        eiso_theta_t2[3,:]=eiso_theta[:]
        eiso_theta_t2[-1,:]=eiso_theta_diffuse[-1,:]
        miso_theta_t2[0,:]=miso_theta[:]
        miso_theta_t2[1,:]=miso_theta[:]
        miso_theta_t2[2,:]=miso_theta[:]
        miso_theta_t2[3,:]=miso_theta[:]
        mscale = miso_theta/eiso_theta
        miso_theta_t2[-1,:]=eiso_theta_diffuse[-1,:]*mscale
        for ii in range(len(t_r)):
            eiso_theta_t2[ii+4,:]=eiso_theta_diffuse[ii,:]
            miso_theta_t2[ii+4,:]=eiso_theta_diffuse[ii,:]*mscale
  
    print('radii2 = ',radii2)
#    plt.plot(eiso_theta_t2[0,:])
#    plt.plot(theta_degr[0:900]+90,eiso_theta[0:900])
#    plt.plot(theta_degr[0:900]+90,eiso_theta_test[0:900])
#    plt.xscale('log')
#    plt.show()

#    plt.plot(t_r/3600/24,ang95)
#    plt.plot(t_r/3600/24,ang90)
#    plt.plot(t_r/3600/24,ang50)
#    plt.xscale('log')
#    plt.show()

    t_obs=t_r-radii/gv.cl
#    plt.plot(t_obs/3600/24,ang95)
#    plt.plot(t_obs/3600/24,ang90)
#    plt.plot(t_obs/3600/24,ang50)
#    plt.xscale('log')
#    plt.show()

    print(time_index)
    print(eiso_theta_t[:,0])
    print(time_index2)
    print(eiso_theta_t2[:,0])
#    plt.plot(theta_index,eiso_theta_t[0,:])
#    plt.plot(theta_index,eiso_theta_t2[0,:])
#    plt.yscale('log')
#    plt.show()
    

    time_index = np.copy(time_index2)
    radii = np.copy(radii2)
    eiso_theta_t = np.copy(eiso_theta_t2)
    miso_theta_t = np.copy(miso_theta_t2)

 
    if (spread == 0):
        time_index=np.array([0.,1e30])
        radii=np.array([0.,1e30])
        eiso_theta_t = np.array([eiso_theta,eiso_theta])
        miso_theta_t = np.array([miso_theta,miso_theta])

    if (spread == 2):
        time_index=np.array([0.,t_theta,t_theta*1.00001,1e30])
        radii=np.array([0.,radius_theta,radius_theta*1.00001,1e30])
        eiso_theta_t = np.array([eiso_theta,eiso_theta,eiso_theta_t2[-1,:],eiso_theta_t2[-1,:]])
        miso_theta_t = np.array([miso_theta,miso_theta,miso_theta_t2[-1,:],miso_theta_t2[-1,:]])

    if (spread == 3):
        time_index=np.array([0.,1e30])
        radii=np.array([0.,1e30])
        eiso_theta_t = np.array([eiso_theta_t2[-1,:],eiso_theta_t2[-1,:]])
        miso_theta_t = np.array([miso_theta_t2[-1,:],miso_theta_t2[-1,:]])



#       plt.plot(values)
#       plt.show()

#    plt.plot(eiso_theta_t)
#    plt.yscale('log')
##    plt.show()    # show all


        
#    for j in range(phi_bins):
#        test_invert[j,:]=1/(4*np.pi) * test[j,:] * (phi_gap*(np.cos(thetamin)-np.cos(thetamax)))/9e47
#        mass_test_invert[j,:]= 1/(4*np.pi) * mass_test[j,:]*(phi_gap*(np.cos(thetamin)-np.cos(thetamax)))/9e47
        
#    binned_energy = test_invert
#    binned_mass = mass_test_invert

    binned_energy = np.zeros((phi_bins, bins))
    binned_mass = np.zeros((phi_bins, bins))
    
    theta0_arr = theta_rot
    
    output = open(savefile, 'wb')

    print('eiso_theta_t = ',eiso_theta_t)
    print('miso_theta_t = ',miso_theta_t)
    print('gamma_theta_t = ',eiso_theta_t/(miso_theta_t * gv.cl**2)+1)
    
    plt.plot(miso_theta_t[0,:])
    plt.yscale('log')
    plt.show()
    
    plt.plot((eiso_theta_t/(miso_theta_t * gv.cl**2)+1)[0,:])
    plt.yscale('log')
    plt.show()
    
    
    pickle.dump((binned_energy, binned_mass, eiso_theta_t, miso_theta_t, time_index, theta_index, theta0_arr, radii), output)
    output.close()

    
