import numpy as np
import pickle
from scipy import optimize
from scipy import interpolate
from matplotlib import pyplot as plt

from global_var import sysvars as gv
#import function_v_decolle_spread
from function_v_decolle_spread_wind import *
import functools
import gc

def blandford_mckee_full_brian(savefile, time0 = 1e4, s_points = 10, z_points = 10, phi_slices = 10, 
                         k_index = 0.0, m_index = 3.0, external_nodensity = 1.0, shock_thickness = 0.1):
                         
#            This function is meant to read in a savefile of binned energies produced by the point by point function
#            From this information, it will impliment the Blandford-McKee solution to determine the conditions
#            and behind the shock at a given time (time0 dedefaults to 10**4 seconds), over a grid in 
#            cylindrical coordinates (s,z,phi) with dimensions determined by the s_points, z_points, and phi_slices
#            (default to 10 each).  The grids are sized such that within a given phi_slice, layers are chosen
#            such that they are evenly spread in s, andwithin each indiviual s-layer, points are evenly spaced
#            in z such thatthey fall primarily within the radius of the shock.
#            
#            Unless phi_slices is set to 360, significant resolution will be lost.  (But if it's equal to 360,
#            then running the afterglow code will require 360 loops, which would take ages.
#            Using the shell keyword will find the radius of the shock and relavant parameters at full resolution
#            ....just in case that information is interesting
            
            
            #gc.collect()
            ##print('gc.get_objects() = ',gc.get_objects())
            #wrappers = [
            #    a for a in gc.get_objects() 
            #    if isinstance(a, functools._lru_cache_wrapper)]
            #
            #for wrapper in wrappers:
            #    wrapper.cache_clear()
        
            data_in = open(savefile, 'rb')
            binned_energy, binned_mass, eiso_theta_t, miso_theta_t, time_index, theta_index, theta0_arr, radii = pickle.load(data_in)
            data_in.close()             
            
            gv.theta0_arr = theta0_arr
            gv.eiso_theta_t = eiso_theta_t
            gv.miso_theta_t = miso_theta_t
            gv.radii_spread = radii
            gv.time_index = time_index
            gv.ntheta = len(gv.theta0_arr[0,:])
            gv.nphi = len(gv.theta0_arr[:,0])
            gv.neiso_t = np.size(gv.eiso_theta_t[:,0])
            gv.neiso_theta = np.size(gv.eiso_theta_t[0,:])
            gv.theta_index = theta_index
            gv.shock_thickness = shock_thickness * gv.cl

            print('gv.theta0_arr = ',gv.theta0_arr)
            
            k = k_index
            m = m_index
            
            gamma_init = 200
            mass0 = 1e52/(gv.cl**2 * gamma_init * (gamma_init-1))
            
            rho_k = external_nodensity * gv.mp  #mass density, rather than number density
            
            gv.rho_k = rho_k
            
            gv.kb = k_index   # Added to correct problem with rerunning - Brian, 7/25/2022  

            
            # convert form full-resolution to the coarse-resolution pi-slices
            #phi-slices should be a divisor of 360 for this to work
           
            #this seems to be the part that takes the longest, so I'm reducing the number of (theta) bins we calculate over
            #I should really figure out a more efficient way to find whatever it is we need this information for - Brian (8/20/2015)
            
            bins = len(binned_energy[0,:])
            phi_bins = len(binned_energy[:,0])
            
            e_slice = np.zeros((phi_slices, bins))
            mass_init_slice = np.zeros((phi_slices, bins))
            
            # change the spacing/size of the phi slices here. - Brian, (8/21/2015)
            #this assumes the highest eregy is around phi = 0 - Brian (8/21/2015)
            #new way for phi2, wiht forward and reverse jet. Just cover half of phi, with high resolution near 0 and
            #near 180 degrees - Brian (5/6/2016)
            
            phi = (np.arange(0,phi_slices/2)/(phi_slices/2))**2 * 90
            temp_phi = (180-phi[::-1])
            phi = np.append(phi, 90)
            phi = np.append(phi, (temp_phi))
            
            for i in range(phi_slices):
                if i < phi_slices-1:
                    phi_min = phi[i]*(phi_bins/360)
                    phi_max = phi[i+1]*(phi_bins/360)
                else:
                   phi_min = phi[i]*(phi_bins/360)
                   phi_max = phi[-1]*(phi_bins/360) 
                
#  old, from Jeremiah
#                e_slice[i,:] = 1e0  

#  new, this is done assuming thete and phi (bins and phi_bins) are flipped from my code - Brian, 2/06/2017
#  might really need axis=0?

                phi_min = int(np.rint(phi_min))
                phi_max = int(np.rint(phi_max))

                e_temp = binned_energy[phi_min:phi_max,:]
                e_slice[i,:] = np.sum(e_temp,axis=0)   #binned_energy[phi_min:phi_max,:].sum(axis=0)
                
                #adding in variable initial mass loading - Brian (10/1/2015)
                gamma_init = 200
                mass0 = 1e52/(gv.cl**2 * gamma_init * (gamma_init - 1))
#  old, from Jeremiah
#                mass_init_slice[i,:] = 1

#  new, this is done assuming thete and phi (bins and phi_bins) are flipped from my code - Brian, 2/06/2017
#  might really need axis=0?
                m_temp = binned_mass[phi_min:phi_max,:]
                mass_init_slice[i,:] = np.sum(m_temp,axis=0)   #binned_mass[phi_min:phi_max,:].sum(axis=0)
                
            #nowfind the radius of the shock front at full resolution in theta but coarse resolution in phi.
            
            r0 = np.zeros((phi_slices, bins))
            e_iso = np.copy(r0)
            theta = np.copy(r0)
            
            #adding in variable initial mass loading - Brian (10/1/2015)
            mass_init = np.copy(r0)
            
            #updating for variable phi bins sizes - Brian (8/21/2015)
            phi_gap = (np.floor(phi[1:] * (phi_bins/360)) - np.floor(phi[0:-1] * (phi_bins/360))) * (360/phi_bins) * np.pi / 180
            
            
            for i in range(bins):
                thetamin = i/bins * np.pi
                thetamax = (i+1)/bins * np.pi
                thetamid = (i+0.5)/bins * np.pi
                theta[:,i] = thetamid
                
                for j in range(phi_slices):
                    e_iso[j,i] = 4*np.pi*e_slice[j,i]/(phi_gap[j]*(np.cos(thetamin)-np.cos(thetamax)))*9e47
                    
                    # adding in variable initial mass loading - Brian (10/1/2015)
                    gamma_init = 200
                    mass0 = 1e52/(gv.cl**2 * gamma_init * (gamma_init-1))
                    
                    mass_init[j,i] = 4*np.pi * mass_init_slice[j,i]/(phi_gap[j]*(np.cos(thetamin) - np.cos(thetamax)))*9e47  # mass3 - from mass_initIslices above
                    
#                    p = np.concatenate(e_iso[j,i], k, rho_k, mass_init[j,i], thetamid)
                    
                    
#            smallr = np.zeros((s_points, z_points, phi_slices))
            smallr = np.zeros((z_points, s_points, phi_slices))
            bigr = np.copy(smallr)
            theta_grid = np.copy(smallr)
            e_iso_interp = np.copy(smallr)
            z_grid = np.copy(smallr)
            s_grid = np.copy(smallr)
            phi_grid = np.copy(smallr)
            r0_interp = np.copy(smallr)
            r0_test1 = np.copy(smallr)
            r0_test2 = np.copy(smallr)
            r0_test3 = np.copy(smallr)
            
            #adding in vaiable initial mass loading - Brian (10/1/2016)
            mass_init_interp = np.copy(smallr)
            
            
            # Here's the meaty bit: set up grid to be balndford-mckee'd later
            # Not entirely sure if this is the best way to go about this, but we'll cross that bridge eventually
            
            # Interpolation in the following section is QUITE BROKEN. Linear interpolation WOULD work fine if we had higher
            # resolution on-axis; as it stande, the minimal resolution highly distorts the spread of pointswith linear interolation
            # However, quadratic/spline/least-squares interpolation all either fail to execute or return a vast quantity of NaN.
            # THis must be fixed; recommendation is to make pointbypoint return higher resolution on-axis - Dominic (7/31/13)
            
            # I really need to implement a solver that will find the maximum s value for each slice - Brian (8/27/2015)
            

	    # This is the main loop over phi_slices - Brian, 3/30/2018
            
            for i in range(phi_slices):
                phi_i = (phi[i] + phi[i+1])/2*np.pi/180  #the mid bin lication of this phi, to be used for energy/mass0 interpolation
                
                theta_by_slice = theta[i,:]
                e_iso_by_slice = e_iso[i,:]
                r0_by_slice = r0[i,:]
                
                # adding in variable initial mass loading - Brian (10/1/2015)
                mass_init_by_slice = mass_init[i,:]
                
                theta_temp = np.copy(theta_by_slice)
                e_iso_temp = np.copy(e_iso_by_slice)
                r0_temp = np.copy(r0_by_slice)
                first_index = len(r0_by_slice)
                #adding in variable initial mass loading - Brian (10/1/2015)
                mass_init_temp = np.copy(mass_init_by_slice)
                
                
                #####################################-added - Brian (8/27/2015)
                
#                p = np.concatenate(k, rho_k, mass0)
                
                #adding in variable inital mass loading - Brian (10/1/2015)
                
#                EXTRA = {'p':p, 'tobs':time0, 'e_iso_temp':e_iso_temp, 'theta_temp':theta_temp, 'mass_init_temp':mass_init_temp}
                PARINFO = np.repeat({'values':0, 'fixed':0, 'limited':[0,0], 'limits':[0,0], 'step':1e-6},1)
                PARINFO[0]['limited'] = [1,1]
                PARINFO[0]['limits'] = [0,np.pi/2]
                
#                angles = np.arange(61)/60 * np.pi
#                angles = np.arange(61)/60 * np.pi
                angles = (np.arange(31)/31)**1.7 * np.pi/2.
                angles = np.concatenate((angles,np.array([np.pi/2.]),np.pi-angles[::-1]),axis=0)

                height = np.zeros((1,np.size(angles)))
                
                p = {'k':k, 'rho_k':rho_k, 'theta':thetamid, 'phi':phi_i, 'tobs':time0, 'theta0_arr':theta0_arr,
                     'eiso_theta_t':eiso_theta_t, 'miso_theta_t':miso_theta_t, 'theta_index':theta_index, 'time_index':time_index}
                
                EXTRA = {'p':p, 'tobs':time0}
                
                
		#########
		# Setting up memoized tobs(R) here - Brian, 3/30/2018
                # Need to loop over all possible thetas - Brian, 3/30/2018
		########

                #p['theta'] = 0.

                print('HERE TEST1',phi_i,p['theta'])
                tobs2 = function_tobs_integrated_memoized(1e18,p)
                tobs1 = function_tobs_integrated_old(1e18,p)
                print('HERE1',tobs1)
                print('HERE2',tobs2)

                tobs3 = function_tobs_integrated_smallr(1e18,p)
                print('HERE3',tobs3-1e18/gv.cl*np.cos(p['theta']))

                print('HERE4',function_r_tobs(1e5,p))

                #EXTRA['p']['tobs']=1e5
                print('HERE5',function_r_integrated(p))
                print('optimize ',optimize.brentq(function_r_tobs, 1, 1e25, args=(EXTRA['p']), maxiter = 1000, xtol=1e-8, rtol = 1e-10))
                print('HERE6',function_smallr_integrated(1e4,p),'theta = ',p['theta'])

                #plt.plot(tobs1)
                #plt.show()


# Before here, I need to set up a memoized function to find tobs(R) by interpolation. - Brian, 3/30/2018


                #calculated the hight of S (y) at a few values of theta - Brian (7/8/2016)
                angles = angles
                for iii in range(np.size(angles)):
#                    if iii == 31:
                    height[0,iii] = function_s_theta(angles[iii], EXTRA)
                height = np.reshape(height, (np.size(height)))
                #height = np.where(np.isfinite(height),height,0.)
                    
                print('height = ',height)
#                
#                
#                #Using the above, bracket where the maximum height will be
#                imax = ((height == min(height))).nonzero()
#                #print('imax = ',imax)
#                imax = imax[0]
#                #imax = imax[0][0]
#                #print('angles = ',angles)
#                #print('height = ',height)
#                print('imax = ',imax)
#
#                test_array = np.zeros(3)
#                print(test_array[2.5])
                
                
                imax = np.argmin(height)
                print('imax = ',imax)
                
                #a = angles[imax-1][0]
                #b = angles[imax][0]
                #c = angles[imax+1][0]
                #fa = height[imax-1][0]
                #fb = height[imax][0]
                #fc = height[imax+1][0]

#                imax = imax[0]
                a = angles[imax-1]
                b = angles[imax]
                c = angles[imax+1]
                fa = height[imax-1]
                fb = height[imax]
                fc = height[imax+1]

                #print('a = ',a)
                
#                raise ValueError('A very specific bad thing happened.')

#                a = angles[imax[0]-1][0]
#                b = angles[imax[0]][0]
#                c = angles[imax[-1]+1][0]
#                fa = height[imax[0]-1][0]
#                fb = height[imax[0]][0]
#                fc = height[imax[-1]+1][0]

                #print('height = ',height,imax)
                #print(a,b,c)
                #print(fa,fb,fc)
                #a=0
                #b=0
                #c=0
            
           
                #now iteritively find the value of theta that corresponds to the maximum height
                theta_max, s_max, iterations, funcalls = optimize.brent(function_s_theta,
                                                                        args=(EXTRA,), brack=(a,b,c),
                                                                        tol = 1e-10,
                                                                        full_output=True)

                # New way to find front edge using 2D minimization - Brian, 5/8/2017
                #theta_max_minimize = optimize.minimize(function_s_theta_tobs, [time0*gv.cl, np.pi/2.], args=(EXTRA), bounds=((1, 1e25), (0, np.pi)), 
                 #                                        method='L-BFGS-B', tol=1.e-18  )

                #theta_max=theta_max_minimize.x[1]
                #s_max=theta_max_minimize.x[0]

                                                                            
                s_max = abs(s_max.flatten())      # Brian, 4/13/2018
                theta_max = theta_max.flatten()   # Brian, 4/13/2018
                theta_max = theta_max % np.pi
                
                print('theta_max =', theta_max, 's_max =',s_max,'a,b,c = ',a,b,c,'fa,fb,fc = ',fa,fb,fc)
                    
                    
                #####################################-end added - Brian (8/27/2015)   
                    
                #even distrubution of points between s_max and zero, including neither
                #(half-integer values rather than integer values in the generating array)
                # should maybe really doa different distribution - Brian (7/8/2016)
                
#                s_funct = (np.arange(0,s_points)+0.5)/s_points
#                s_funct = 1-(s_points-1-np.arange(0,s_points)+0.5)**2/s_points**2
#                s_funct = 1-(s_points-1-np.arange(0,s_points)+.99)**2/s_points**2
                s_funct = (1.-(((np.arange(s_points))/(s_points))**2))*.99
                s_funct = np.copy(s_funct[::-1])
                s_raw = s_max.flatten() * s_funct
                
                
                interp_r_before = np.zeros((1,len(s_raw)))
                interp_r_after = np.zeros((1,len(s_raw)))
                interp_theta_before = np.zeros((1,len(s_raw)))
                interp_theta_after = np.zeros((1,len(s_raw)))
#                print(interp_r_before)                
                
                p = {'k':k, 'rho_k':rho_k, 'theta':thetamid, 'phi':phi_i, 
                     'tobs':time0, 'theta0_arr':theta0_arr, 'eiso_theta_t':eiso_theta_t, 
                     'miso_theta_t':miso_theta_t, 'theta_index':theta_index, 'time_index':time_index}
                     
                EXTRA = {'p':p, 'tobs':time0}
                EXTRAz = {'p':p,'tobs':time0, 'z':s_raw[0]}
                
                #loop over s slices - Brian (12/18/2013)
                for ii in range(len(s_raw)):
                    p = [k, rho_k, mass0]
                    
                    EXTRAz['z'] = s_raw[ii]
                    
                    #front edgoe of shock at this value of S (y)  Brian (7/8/2016)
#                    print(function_theta_z_tobs(1e-12,EXTRAz),function_theta_z_tobs(theta_max,EXTRAz),function_theta_z_tobs(theta_max*1.1,EXTRAz),theta_max,s_raw[ii],s_max)
#                    EXTRAz['z'] = s_max
#                    print(function_theta_z_tobs(theta_max,EXTRAz))
#                    print('s_raw[ii] = ',s_raw,len(s_raw))
#                    EXTRAz['z'] = s_raw[ii]

                    if ( np.sign(function_theta_z_tobs(1e-12,EXTRAz)) == np.sign(function_theta_z_tobs(theta_max,EXTRAz)) ):
                        print('THETA_Z_TOBS FAIL!!! ',theta_max)
                        theta_test0=theta_max
                    else:
                        theta_test0 = optimize.brentq(function_theta_z_tobs, 1e-12, theta_max, 
                                                        args=(EXTRAz), maxiter = 1000, xtol=1e-10, rtol = 1e-10)
                                                    

#                    print(s_raw[ii], theta_max/2.,function_theta_z_tobs([s_raw[ii], theta_max/2.],EXTRAz) )
#                    EXTRA['p']['theta']=theta_max/2.
#                    t_test=function_tobs_integrated(s_raw[ii],EXTRA['p'])
#                    print(t_test,abs(time0 - t_test), abs((s_raw[ii]*np.sin(theta_max/2.)-s_raw[ii])/s_raw[ii]))
#                    mulfac=1.+1.49e-8
#                    print(s_raw[ii]*mulfac, theta_max/2.,function_theta_z_tobs([s_raw[ii]*mulfac, theta_max/2.],EXTRAz))
#                    t_test=function_tobs_integrated(s_raw[ii]*mulfac,EXTRA['p'])
#                    print(t_test,abs(time0 - t_test), abs((s_raw[ii]*mulfac*np.sin(theta_max/2.)-s_raw[ii])/s_raw[ii]))
#
#                    print(s_raw[ii], theta_max/2.*mulfac,function_theta_z_tobs([s_raw[ii], theta_max/2.*mulfac],EXTRAz))
#                    EXTRA['p']['theta']=theta_max/2.*mulfac
#                    t_test=function_tobs_integrated(s_raw[ii],EXTRA['p'])
#                    print(t_test,abs(time0 - t_test), abs((s_raw[ii]*np.sin(theta_max/2.*mulfac)-s_raw[ii])/s_raw[ii]))

                    # New way to find front edge using 2D minimization - Brian, 5/8/2017
#                    theta_test0_minimize = optimize.minimize(function_theta_z_tobs, [s_raw[ii]/2., theta_max/2.], args=(EXTRAz), bounds=((1, 1e25), (0, theta_max)), 
#                                                             method='COBYLA', options={'tol':1e-18} )
#                                                             method='TNC', options={'ftol':1e-18,'eps':1e-2,'gtol':1e-18,'xtol':1e-18,'scale':(1e12,1)} )
#                                                             method='SLSQP', options={'ftol':1e-18,'eps':1e-6} )
#                                                             method='L-BFGS-B', options={'gtol':1e-24,'ftol':1e-18,'eps':1e-1}  )
#                                                             method='Nelder-Mead', options={'maxiter':2000} )
#                                                             method='Powell', options={'maxiter':2000} )


#                    bounds=[(1, 1e25), (0, theta_max)]
#                    print(len(bounds))
#                    theta_test0_minimize = optimize.differential_evolution(function_theta_z_tobs, bounds, args=(EXTRAz), 
#                                                                            tol=0.01)


#                    theta_test0_minimize = optimize.root(function_theta_z_tobs, [s_raw[ii],theta_max/2.], args=(EXTRAz) )


#                    theta_test0=theta_test0_minimize.x[1]

#                    print(theta_test0_minimize)


                    #theta_test0 = theta_max+(theta_test0 - theta_max) * (1 - 1e-4)  #safety factor

                    # New safety factor test, to get rid of strange shape a nose - Brian, 2/07/201
                    sfactor=1e-4

                    theta_test0 = np.arctan(1.e0/((1.e0/np.tan(theta_test0)-1.e0/np.tan(theta_max))*(1.e0-sfactor) + 1.e0/np.tan(theta_max)))


#                    interf_e_iso_test0 = interpolate.interp1d(np.concatenate(([-theta_temp[0]], theta_temp)), np.concatenate(([e_iso_temp[0]], e_iso_temp)), bounds_error = False)
#                    e_iso_test0 = interf_e_iso_test0(theta_test0)
                    
                    #adding in varibale intial mass loading - Brian (10/01/2015)
#                    interf_mass_init_test0 = interpolate.interp1d(np.concatenate(([-theta_temp[0]], theta_temp)), np.concatenate(([mass_init_temp[0]],mass_init_temp)), bounds_error = False)
#                    mass_init_test0 = interf_mass_init_test0(theta_test0)
#                    mass0 = mass_init_test0
                    
                    EXTRA['p']['theta'] = theta_test0
                    
                    #r0_front_test0 = optimize.brentq(function_r_tobs, 1, 1e25, args=(EXTRA['p']), maxiter = 1000, xtol=1e-8, rtol = 1e-10)
                    r0_front_test0 = function_r_integrated(EXTRA['p'])

#                    print(r0_front_test0, theta_test0)
                    
#                    p = np.concatenate(k, rho_k, mass0)
                    
                    #find back edge of the shock at this value of S(y) - Brian (7/8/16)

                    if ( np.sign(function_theta_z_tobs(np.pi-1e-12,EXTRAz)) == np.sign(function_theta_z_tobs(theta_max,EXTRAz)) ):
                        print('THETA_Z_TOBS FAIL!!! ',theta_max)
                        theta_test0=theta_max
                        theta_test1=theta_max
                    else:
                        theta_test1 = optimize.brentq(function_theta_z_tobs, theta_max, np.pi-1e-12, 
                                                        args=(EXTRAz), maxiter = 1000, xtol=1e-10, rtol = 1e-10)
                                                    
                    # New way to find front edge using 2D minimization - Brian, 5/8/2017
#                    theta_test1_minimize = optimize.minimize(function_theta_z_tobs, [s_raw[ii], theta_max], args=(EXTRAz), bounds=((1, 1e25), (theta_max, np.pi)), 
#                                                             method='L-BFGS-B', tol=1.e-8  )
#                                                             method='Nelder-Mead', options={'maxiter':2000} )
#                                                             method='Powell', options={'maxiter':2000,'ftol':1e-7} )

#                    theta_test1=theta_test1_minimize.x[1]

#                    print(theta_test1_minimize)

                    #theta_test1 = theta_max + (theta_test1-theta_max) * ( 1 - 1e-4)  #safety factor
                    
                    # New safety factor test, to get rid of strange shape a nose - Brian, 2/07/2017

                    theta_test1 = np.arctan(1.e0/((1.e0/np.tan(theta_test1)-1.e0/np.tan(theta_max))*(1.e0-sfactor) + 1.e0/np.tan(theta_max)))


#                    interf_e_iso_test1 = interpolate.interp1d(np.concatenate(([-theta_temp[0]], theta_temp)), np.concatenate(([e_iso_temp[0]], e_iso_temp)), bounds_error = False)
#                    e_iso_test1 = interf_e_iso_test1(theta_test0)
                    
                    #adding in varibale intial mass loading - Brian (10/01/2015)
#                    interf_mass_init_test1 = interpolate.interp1d(np.concatenate(([-theta_temp[0]], theta_temp)), np.concatenate(([mass_init_temp[0]],mass_init_temp)), bounds_error = False)
#                    mass_init_test0 = interf_mass_init_test0(theta_test0)
#                    mass0 = mass_init_test0
                    
                    EXTRA['p']['theta'] = theta_test1

                #    print('theta = ',theta_test1)
                    #r0_back_test1 = optimize.brentq(function_r_tobs, 1, 1e25, args=(EXTRA['p']), maxiter = 1000, xtol=1e-8, rtol = 1e-10)
                    r0_back_test1 = function_r_integrated(EXTRA['p'])
#                    print(r0_back_test1, theta_test1)
                    
                    interp_r_before[0,ii] = r0_front_test0
                    interp_r_after[0,ii] = r0_back_test1
                    
                    interp_theta_before[0,ii] = theta_test0
                    interp_theta_after[0,ii] = theta_test1
                    
                
                # This finds the front (z_max) and back (z_min) of each s-layer, and converts z-min to be negative if necessary.
                
                #Just do it by angle - Brian (8/27/2013)
                
                z_max = s_raw * 1/np.tan(interp_theta_before)
                z_min = s_raw * 1/np.tan(interp_theta_after)
                
                z_max = np.reshape(z_max, np.size(z_max))                
                z_min = np.reshape(z_min, np.size(z_min))  
                
                # add in middle point here - Brian (8/26/2015)
                z_mid = s_raw * 1/np.tan(theta_max)
                theta_mid = np.arctan2(s_raw, z_mid)
#                print(z_mid)
                for iii in range(len(z_mid)):
                    if z_mid[iii] <= z_min[iii]:
                        print('stop')
                        
                theta_max = interp_theta_before
                theta_min = interp_theta_after
                
                theta_max = np.reshape(theta_max, np.size(theta_max))                
                theta_min = np.reshape(theta_min, np.size(theta_min))                 
                
                if max(theta_min) >= np.pi/2:
                    highest_theta_min = max(theta_min)
                    highest_index = (theta_min == highest_theta_min).nonzero()
                    #added if statement Brian 12/16/2013
                    print('highest_index ',highest_index)
                    highest_index = highest_index[0].flatten()
                    highest_index = highest_index[0]
                    if highest_index >= 1:
                        z_min[0:highest_index] = -1
                        
                        
                # make z alittle smaller, z_min a little larger so ensure r<R - Brian 12/23/2013
                z_range = (z_max - z_min)/2
                z_mean = (z_max+z_min)/2
                
                # This is a new version to space points between z_max and z_mid and z_min - Brian 8/26/2015
                mult_ones = np.ones((z_points, 1))
                mult_ones0 = np.ones((int(z_points/2), 1))   # adding int() here - Brian, 9/13/2018
                
                z_min = np.reshape(z_min, (1, np.size(z_min)))
                z_mid = np.reshape(z_mid, (1, np.size(z_mid)))
                z_max = np.reshape(z_max, (1, np.size(z_max)))
                
                z_span0 = z_mid - z_min
                z_min_grid0 = np.matmul(mult_ones0, z_min)
                #z_funct0 = (np.arange((0,z_points/2))/z_points/2 -1)  # unifrom spacing
#                z_funct0 = (np.arange((z_points/2))/(z_points/2 -1))**3  #non-uniform spacing
                z_funct0 = (np.arange((z_points/2-9))/(z_points/2-9 -1))**3  #non-uniform spacing
                z_funct0 = np.append(z_funct0[0]+(np.arange((10))/(11 - 1))**3*(z_funct0[1]-z_funct0[0]),z_funct0[1:])  # add 10 extra points between first 2

                z_funct0 = np.reshape(z_funct0, (np.size(z_funct0), 1))                
                
                z_offset0 = np.matmul(z_funct0,z_span0)
                z_raw0 = z_offset0 + z_min_grid0
                
                z_span1 = z_max - z_mid
                z_min_grid1 = np.matmul(mult_ones0, z_mid)
#                z_funct1 = np.arange((z_points/2))/(z_points/2 - 1) # uniform spacing
#                z_funct1 = 1 - (np.arange((z_points/2))/(z_points/2 - 1))**2  #non-uniform spacing, 2 seems okay  - Brian, 6/23/2017
                z_funct1 = 1 - (np.arange((z_points/2-9))/(z_points/2-9 - 1))**2  #non-uniform spacing, 2 seems okay  - Brian, 6/23/2017
                z_funct2 = np.append(z_funct1[0]+(np.arange((10))/(11 - 1))**2*(z_funct1[1]-z_funct1[0]),z_funct1[1:])  # add 10 extra points between first 2
                z_funct1 = np.copy(z_funct2[::-1])  #non-uniform spacing  - Brian, 6/23/2017
                
                z_funct1 = np.reshape(z_funct1, (np.size(z_funct1), 1))
                
                z_offset1 = np.matmul(z_funct1, z_span1)
                z_raw1 = z_offset1+z_min_grid1
                
                s_raw = np.reshape(s_raw, (1, np.size(s_raw)))                
                
                z_raw = np.concatenate((z_raw0, z_raw1))
                s_layers = np.matmul(mult_ones, s_raw)
                

#                print(z_raw)

                ## And finally, fill in the empty arrays we made earlier
                theta_grid[:,:,i] = np.arctan2(s_layers, z_raw)
                smallr[:,:,i] = np.sqrt(s_layers**2 + z_raw**2)

                #using the function to find e_iso_interp - Brian 5/3/2016
                
                # I'm not sure what the eiso_miso_index part and loop over j is doing.  Can I just skip this? - Brian, 2/07/2017
                # seemed to break it, going back to old way - Brian, 2/07/2017

                eiso_miso_index = (smallr[:,:,i] == smallr[:,:,i]).nonzero()
                
                eiso_miso_index0 = eiso_miso_index[0]   
                eiso_miso_index1 = eiso_miso_index[1]
                
#                print ('eiso_miso_index0 =',eiso_miso_index0)
#                print ('eiso_miso_index1 =',eiso_miso_index1)

                for j in range(len(eiso_miso_index0)):
                
                    p = {'k':k, 'rho_k':rho_k, 'theta':theta_grid[eiso_miso_index0[j],eiso_miso_index1[j],i], 'phi':phi_i, 'tobs':time0, 'theta0_arr':theta0_arr, 'eiso_theta_t':eiso_theta_t, 'miso_theta_t':miso_theta_t, 'theta_index':theta_index, 'time_index':time_index}

                    # Fixed how e_iso_interp is being filled - Brian, 2/07/2017
                    #e_iso_interp[:,:,i] = e_iso_interp_spread(smallr[eiso_miso_index0[j],eiso_miso_index1[j],i],p)
#                    e_iso_interp[eiso_miso_index0[j],eiso_miso_index1[j],i] = e_iso_interp_spread(smallr[eiso_miso_index0[j],eiso_miso_index1[j],i],p)
                
                    #adding in variable initial mass loading - Brian 10/01/2015
                    #mass_init_interp[:,:,i] = m_iso_interp_spread(smallr[eiso_miso_index0[j],eiso_miso_index1[j],i],p)
#                    mass_init_interp[eiso_miso_index0[j],eiso_miso_index1[j],i] = m_iso_interp_spread(smallr[eiso_miso_index0[j],eiso_miso_index1[j],i],p)

# Using interp_spread functions instead - Brian, 4/13/2017

#                    e_iso_interp[eiso_miso_index0[j],eiso_miso_index1[j],i] = e_iso_interp_spread(smallr[eiso_miso_index0[j],eiso_miso_index1[j],i], p)
#                    mass_init_interp[eiso_miso_index0[j],eiso_miso_index1[j],i] = m_iso_interp_spread(smallr[eiso_miso_index0[j],eiso_miso_index1[j],i], p)
#                    e_data = e_iso_interp_spread(smallr[eiso_miso_index0[j],eiso_miso_index1[j],i], p)
                    e_data = e_iso_interp_spread_memoized(smallr[eiso_miso_index0[j],eiso_miso_index1[j],i], p['tobs'], p['theta'], p['phi'])
                    e_iso_interp[eiso_miso_index0[j],eiso_miso_index1[j],i] = e_data[0]
                    mass_init_interp[eiso_miso_index0[j],eiso_miso_index1[j],i] = e_data[1]



#                p = {'k':k, 'rho_k':rho_k, 'theta':theta_grid[:,:,i], 'phi':phi_i, 'tobs':time0, 'theta0_arr':theta0_arr, 'eiso_theta_t':eiso_theta_t, 'miso_theta_t':miso_theta_t, 'theta_index':theta_index, 'time_index':time_index}
#
#                e_iso_interp[:,:,i] = e_iso_interp_spread(smallr[:,:,i],p)
#                
#                #adding in variable initial mass loading - Brian 10/01/2015
#                mass_init_interp[:,:,i] = m_iso_interp_spread(smallr[:,:,i],p)

                # end modified - Brian, 2/07/2017

                
                # don't need this? - Brian, 2/08/2017
                #interf_r0_interp = interpolate.interp1d(theta_temp, r0_temp)
                #r0_interp[:,:,i] = interf_r0_interp(theta_grid[:,:,i])
                
                z_grid[:,:,i] = z_raw
                s_grid[:,:,i] = s_layers
                phi_grid[:,:,i] = phi_i
                
                ## end phi loop
                
            #and now we just B-McK base don the gridded points.
                
            length = np.size(e_iso_interp)
            
            r_index_arr = (smallr == smallr).nonzero()
            
            r_index_arr0 = r_index_arr[0]
            r_index_arr1 = r_index_arr[1]
            r_index_arr2 = r_index_arr[2]
            
            p = {'k':k, 'rho_k':rho_k, 'theta':theta_grid[r_index_arr0[0], r_index_arr1[0], r_index_arr2[0]], 'phi':phi_grid[r_index_arr0[0], r_index_arr1[0], r_index_arr2[0]], 'tobs':time0, 'theta0_arr':theta0_arr, 'eiso_theta_t':eiso_theta_t, 'miso_theta_t':miso_theta_t, 'theta_index':theta_index, 'time_index':time_index}
            EXTRA = {'p':p, 'tobs':time0, 'smallr':smallr[r_index_arr0[0], r_index_arr1[0], r_index_arr2[0]]}
 
            for i in range(len(r_index_arr0)):
                EXTRA['p']['theta']=theta_grid[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]]
                EXTRA['p']['phi']=phi_grid[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]]
                EXTRA['smallr']=smallr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]]
               
                #error handeling in case no value for smallr is found.  Seems to be a problem far off axis - Brian 8/25/2015
                
                
#                print(function_r_tobs_smallr(EXTRA['smallr']*.1,EXTRA),function_r_tobs_smallr(1e25,EXTRA),EXTRA['smallr'])
                if((smallr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]] > 0) and np.isfinite(smallr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]]) and (function_r_tobs_smallr(EXTRA['smallr']*.1,EXTRA) > 0)):
#                    EXTRA = {'p':p, 'tobs':time0, 'smallr':smallr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]]}
 
#                    bigr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]] = optimize.brentq(function_r_tobs_smallr, 1, 1e25, args = (EXTRA), maxiter = 1000, xtol=1e-8, rtol=1e-8)
                    #bigr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]] = optimize.brentq(function_r_tobs_smallr, EXTRA['smallr']*.1, 1e25, args = (EXTRA), maxiter = 1000, xtol=1e-8, rtol=1e-8, disp=False)
                    bigr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]] = function_smallr_integrated(EXTRA['smallr'],EXTRA['p'])

                else:
                    bigr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]] = 1
#                print ('bigr', bigr[r_index_arr0[i], r_index_arr1[i], r_index_arr2[i]])
                
            
#            for i in range(length):
#                
##                p= [e_iso_interp[i], k, rho_k, mass_init_interp[i], theta_grid[i]]
#                
#                p = {'k':k, 'rho_k':rho_k, 'theta':theta_grid[i], 'phi':phi_grid[i], 'tobs':time0, 'theta0_arr':theta0_arr, 'eiso_theta_t':eiso_theta_t, 'miso_theta_t':miso_theta_t, 'theta_index':theta_index, 'time_index':time_index}
#                
#                #error handeling in case no value for smallr is found.  Seems to be a problem far off axis - Brian 8/25/2015
#                
#                if(smallr[i] <= 0):
#                    bigr[i] = 1
#                else:
#                    EXTRA = {'p':p, 'tobs':time0, 'smallr':smallr[i]}
#                    bigr[i] = optimize.brentq(function_r_tobs_smallr, 1, 1e25, args = (EXTRA))
#                print (bigr[i])
#                
            rxi0 = smallr/bigr
            
            
            m0b_k = 4/(3-k)*np.pi*rho_k*bigr**(3-k) + mass_init_interp
            g0b = (1+(1+(17-4*k)/4*4*e_iso_interp/m0b_k/gv.cl**2)**(0.5))/2
            c0b_k = 4 * e_iso_interp/m0b_k/gv.cl**2 * (17-4*k)/4
            v0b = ((-2+2*(1+c0b_k)**(0.5)+c0b_k)/(2+2*(1+c0b_k)**(0.5)+c0b_k))**0.5
            gs0b = np.sqrt((g0b+1)*(4/3 * (g0b-1)+1)**2/(4/3 *(2-4/3)*(g0b-1)+2))
            vs0b = np.sqrt(1-1/gs0b**2)
            
            # DeColle formulation below.  Part above this not strictly necessary - Brian, 5/22/2017

            m0b_k = 4/(3-k) * np.pi * rho_k * bigr**(3-k) #+ mass_init_interp * (1.+1.e-10)
            c0b_k = 4 * e_iso_interp/m0b_k/gv.cl**2 * (17-4*k)/4
            
            g_ad = 4/3
            
            #alpha_k = 8*np.pi*(g_ad+1)/(3*(g_ad-1)*(3*g_ad-1)**2)   # This is the wrong formula
            alpha_k = 5.60 # For k=2  # 1.25 #0.8840750103008531   # For k=0
            
            
            # For any k, hopefully - Brian, 2/17/2021
            gammat = g_ad
            nn=2  # spherical coordinates
            ntaye = ((2-nn)*gammat**2. + (3.*nn+1)*gammat - 1 - k*gammat*(gammat+1)) / (gammat**2.-1.)
            pte = (2.*(gammat + 2.*nn + 1 - k*(gammat+1.))) / (3.*nn + 1. - (nn-1.)*gammat - k*(gammat+1.))
            qte = (2.*gammat**2. + (3.*nn+1.)*gammat) - (nn+1.) - (k*gammat*(gammat+1.)) / (3.*nn + 1. - (nn-1.)*gammat - k*(gammat+1.))
            ste = (gammat+1.) / (3.*nn + 1. - (nn-1.)*gammat - k*(gammat+1.))

            ntay=ntaye
            pt=pte #-1.
            qt=pte 
            st=ste

            rint = (np.arange(1000000)+.5)/1000000
            
            # Use this commented part to calculate alpha_k for a givem adiabatic index and k - Brian, 1/23/2022
            # Based on Taylor 1950.
            '''
            rho_taylor = rint**((3.-k*gammat)/(gammat-1.))*((gammat+1.)/gammat - rint**(ntay-1.)/gammat)**(-pt)
            u_taylor = (gammat+1.)/2. * (rint/gammat + (gammat-1.)/(gammat+1.)*rint**ntay/gammat)
            pres_taylor = ((gammat+1.)/gammat - rint**(ntay-1.)/gammat)**(-qt)

            I_k = np.sum(rho_taylor*u_taylor**2*rint**2*(rint[1]-rint[0])) #*(9/4*gammat**2-1)
            I_t = np.sum(pres_taylor*rint**2*(rint[1]-rint[0])) #* gammat**2

            #beta_A = 4*(3-k)/(gammat**2-1.)*(I_k+I_t)
            #alpha_k_taylor = (2/(5-k))**2*(2*np.pi/(3-k)) * beta_A
            #alpha_k_taylor = 8*np.pi/((5-k)**2*(3-k)) * beta_A
            alpha_k = np.sqrt(2) * 8*(4*np.pi)/(gammat**2-1)* 1/(5-k)**2*(I_k+I_t)
            
            print('HERE7 alpha_k = ',alpha_k)
            
            ###'''
            
            # alpha_k should be set to a good value now.
            
            
            f0 = e_iso_interp/(m0b_k * gv.cl**2) * 4*np.pi/(3-k)
            print('e_iso_interp, mass_init_interp, m0b_k, bigr = ',e_iso_interp[0,0,0])
            print(mass_init_interp[-1,0,0])
            print(m0b_k[-1,0,0])
            print(bigr[-1,0,0])
            print('f0 = ',f0[-1,0,0])
            # adjusting f0 for initial mass - Brian, 4/13/2021
            
            #print('e_iso_interp = ',e_iso_interp)
            gamma_0_init_2 = (e_iso_interp/(mass_init_interp*gv.cl**2))**2
            
            shock_thickness = gv.shock_thickness #10. * gv.cl #0.1 * gv.cl
            
            gamma_init = 1+(e_iso_interp/(mass_init_interp*gv.cl**2))
            beta_init = np.sqrt(1-1/gamma_init**2)

            x0 = 8*np.pi / (17 - 4*k)
            y0 = (5-k)**2/(4*alpha_k) * alpha_k**2
            

            #f1 = e_iso_interp/(m0b_k * gv.cl**2) * 4*np.pi/(3-k) * ((m0b_k-mass_init_interp)**(2/(3-k)) / (gamma_0_init_2*shock_thickness) * m0b_k/(m0b_k-mass_init_interp))
            #f1 = 1.42*(3*e_iso_interp/(16*np.pi*rho_k*shock_thickness))**(1/2) * bigr**(-1) / x0 / gv.cl #(gv.cl)**(-1/2)
            #f1 = 1.42*(3*e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * bigr**(-1) * x0 / 2.7 / gv.cl
            f1 = 1.42*(3*e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * bigr**(-1) * x0 / 2.3 / gv.cl
            print('f1 = ',f1[-1,0,0])
            


            
            #f0 = np.minimum(f0,f1)
            
            # end adjust f0 - Brian, 4/13/2021
            
            # Further adjustments fitting to numerical simulations - Brian, 6/9/2021
            

            e1 = 1/2*1/f1*e_iso_interp*1.35**2 * bigr #* np.sqrt(1-1/gamma_init**2)
            e3 = 0.8*4*np.pi/(3-k)*rho_k * bigr**(3-k) * gv.cl**2. * gamma_init**2. * 2

            print(e1.shape)
            print('e1 = ',e1[-1,0,0])
            print('e3 = ',e3[-1,0,0])
            #print('e3 = ',e3)
            e1=e3
            
            m2b_k = 4/(3-k) * np.pi * rho_k * bigr**(3-k) + mass_init_interp / np.sqrt(gamma_init)
            
            e2 = e_iso_interp * (1+(mass_init_interp/m2b_k)**.225/gamma_init**1.1125) # between .15 and .275 works okay. .225 seems good. #* m2b_k/mass_init_interp  # Work wiht just *1 for gamma_init=10

            m0b_k = m2b_k #np.where(m2b_k<mass_init_interp/2,m2b_k*2,m0b_k)

            # *** This f0 works for ST16 ***
            f0 = e2/(m0b_k * gv.cl**2) * 4*np.pi/(3-k)
            print('HERE TEST f0 = ',f0[-1,0,0])

            # Gamma-Beta formulation - Brian, 7/22/2021
            exp1 = .25 #1/3 # and 1.4
            exp2 = .25 #1/3 # and 1.4
            z1_f0 = (x0*1.6*f0**exp1+y0/f0**exp2)/(1.6*f0**exp1+1/f0**exp2)
            v_s_f0 = np.sqrt(f0/(f0+z1_f0))
            gamma_s_f0 = 1./np.sqrt(1-v_s_f0**2)
            gb_f0 = gamma_s_f0**2*v_s_f0**2

            
            print('HERE!!! bigr,m0b_k,gamma_s_f0')
            print(bigr,m0b_k,gamma_s_f0)
    

            #f_init = e_iso_interp/(mass_init_interp * gv.cl**2) * 4*np.pi/(3-k) * np.sqrt(gamma_init) *2 #*(1+1/gamma_init) #*2.

            # New f_init calculation - Brian, 7/6/2021
            dens_ratio_init = (g_ad*gamma_init+1) / (g_ad-1)
            beta_test_init = ((dens_ratio_init*gamma_init) / (dens_ratio_init*gamma_init-1))**(1/3) * beta_init
            gamma_test_init = np.sqrt(1/(1-beta_test_init**2))

            print('gamma_init = ',gamma_init)
            print('gamma_test_init = ',gamma_test_init)
            
            dens_ratio_init = (g_ad*gamma_init+1) / (g_ad-1)
            beta_test_init = ((dens_ratio_init*gamma_init) / (dens_ratio_init*gamma_init-1))**(1/3) * beta_init
            gamma_test_init = np.sqrt(1/(1-beta_test_init**2))

            #f_init = gamma_test_init**2 * beta_test_init**2 * 4*np.pi/(3-k) * 1/2 * (1+1/gamma_init**.5) #* (1+1/gamma_init**2)
            f_init = gamma_test_init**2 * beta_test_init**2 * 4*np.pi/(3-k)/2 * (1+1/gamma_init**2.) #* (1+1/gamma_init**2)
            print('f_init = ',f_init[-1,0,0])


            
            #f0 = np.where(e1<e_iso_interp, f_init*(1-e1/e_iso_interp)+f0*(e1/e_iso_interp),f0)
            # New interpolation wiht f_init - Brian, 7/6/2021
            #f0 = np.where(e1<e_iso_interp, f_init*(1-(e1/e_iso_interp)**.1)+f0*(e1/e_iso_interp)**.1,f0)
            ##f0 = np.where(e1<e_iso_interp, np.maximum(f_init,f0)*(1-(e1/e_iso_interp)**.1)+f0*(e1/e_iso_interp)**.1,f0)
            f_init_0 = e_iso_interp/(mass_init_interp * gv.cl**2) * 4*np.pi/(3-k) * np.sqrt(gamma_init) * (1+gamma_init**(.5*.225)/gamma_init**1.1125) #*2.
            
            

            
            # New version with x^x - Brian, 7/19/2021
            f0_index_a = (1-(f_init-f_init_0)/f_init)*.8+.1333
            fvar = (np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))
            ffunc = np.abs(fvar)**(.025+(1-np.abs(fvar))**4*.175)
            f0 = np.where(f0>f0_index_a*f_init_0, f_init*(1-ffunc)+f0*ffunc,f0)

            #f0_index_a=0.2
            #f0 = np.where(f0>f0_index_a*f_init_0, f_init*(1-(np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))**.05)+f0*((np.abs(f_init_0-f0)/((1-f0_index_a)*f_init_0))**.05),f0)
            
            f0 = np.minimum(f1,f0)

            print('f_init = ',f_init)
            print('f_init_0 = ',f_init_0)
            print('f0 BM = ',f0)

            
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
            
            gamma_f_f1 = np.sqrt(2.7/.449*(e_iso_interp*beta_init/(16*np.pi*rho_k*shock_thickness))**(1/2) * bigr**(-1)/gv.cl/2+1)
            gamma_s_f1_2 = (gamma_f_f1+1) * (g_ad*(gamma_f_f1 - 1) + 1)**2/(g_ad*(2-g_ad)*(gamma_f_f1-1)+2)
            #gamma_s_f1 = np.sqrt(gamma_s_f1_2)
            #v_s_f1 = np.sqrt(1-1/gamma_s_f1_2)
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
            
            
            v_s2 = ((f0**2 + 4*f0*x0-2*f0*y0+y0**2)**0.5-f0-y0)/(2*x0-2*y0)
            v_s = np.sqrt(v_s2)
            
            
            # Trying new interpolation based on numerical results - Brian, 3/19/2021
            
            v_s2_rel = f0/(f0+x0)
            v_s_rel = np.sqrt(v_s2_rel)

            v_s2_nr = f0/(f0+y0)
            v_s_nr = np.sqrt(v_s2_nr)

            v_s2 = (v_s_rel*v_s_rel**2 + v_s_nr*(1-v_s_rel**2))**2
            v_s = np.sqrt(v_s2)

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
            '''
            # Gamma-Beta formulation - Brian, 7/22/2021
            v_s2 = gb_f0/(gb_f0+1) #np.sqrt(f0/(f0+z1))
            v_s = np.sqrt(v_s2) + 1.e-24
              
            
            # End best fit interpolation - Brian, 4/13/2021

            
            
            gamma_s2 = 1/(1-v_s2)
            
            
            vs0b = v_s
            g_s0b = np.sqrt(gamma_s2)
            
            g = gamma_s2
            a = g_ad
            g0b = gamma_s2
        
        # Calculate g0b by calling optimize.brentq
    #        for ijk in range(len(r_index_arr0)):
    #            EXTRA = {'gamma_s':np.sqrt(gamma_s2[r_index_arr0[ijk], r_index_arr1[ijk], r_index_arr2[ijk]]), 'g_ad':g_ad}
    #            low = np.sqrt(2*gamma_s2[r_index_arr0[ijk], r_index_arr1[ijk], r_index_arr2[ijk]])
    #            g0b_ijk = optimize.brentq(function_gamma_f_gamma_s, low , 1, args = (EXTRA), maxiter = 1000, xtol=1e-8, rtol=1e-10)
    #            g0b[r_index_arr0[ijk], r_index_arr1[ijk], r_index_arr2[ijk]] = g0b_ijk
     
        
            # Added new way of calculating g0b, without calling solver.  Analytic solution instead. - Brian, 2/17/2021
            
            a = gamma_s2+0j
            #a = gamma_test_values**2.+0j #gamma_test_values**2.
            g = g_ad
            g0b = gamma_s2

            # Need to solve for gamma_f given gamma_s2 above.
            # gamma_s_new2 = (gamma_f+1) * (g_ad*(gamma_f - 1) + 1)**2/(g_ad*(2-g_ad)*(gamma_f-1)+2)

            gamma_f_test = 0.0
            gamma_f_test_i  = (18*a*g**6 - 18*a*g**5 + 18*a*g**4 + np.sqrt(4*(3*a*g**4 - 6*a*g**3 - 4*g**4 + 4*g**3 - g**2)**3 + (18*a*g**6 - 18*a*g**5 + 18*a*g**4 - 16*g**6 + 24*g**5 - 12*g**4 + 2*g**3)**2) - 16*g**6 + 24*g**5 - 12*g**4 + 2*g**3)**(1/3)/(3*2**(1/3)*g**2) - (2**(1/3) * (3*a*g**4 - 6*a*g**3 - 4*g**4 + 4*g**3 - g**2))/(3*g**2 * (18*a*g**6 - 18*a*g**5 + 18*a*g**4 + np.sqrt(4 * (3*a*g**4 - 6*a*g**3 - 4*g**4 + 4*g**3 - g**2)**3 + (18*a*g**6 - 18*a*g**5 + 18*a*g**4 - 16*g**6 + 24*g**5 - 12*g**4 + 2*g**3)**2) - 16*g**6 + 24*g**5 - 12*g**4 + 2*g**3)**(1/3)) - (2*g - g**2)/(3*g**2)
            
            g0b = gamma_f_test_i.real

            g = gamma_s2
            a = g_ad
            
            
            g0b2 = g0b**2
            v0b2 = 1 - 1/g0b2
            v0b = np.sqrt(v0b2)
            
            ## Now we have all the values at the shock front
            
            ## Calculate the internal structure
            ##BM solutions
            
            
            xi = 1 + 4*(m+1) * (1-rxi0) *g0b**2
            #xi[np.where(xi <= 1.)]=1.

            
            print('xi = ',xi[98:,:,0])

            densprime_test2 = bigr**(-k) * rho_k * 3 * (4/3 * g0b+1) * g0b*xi**(-(7-2*k)/(4-k))
            pres_test2 = bigr**(-k)*rho_k* (g0b-1)*(4/3 *g0b + 1) * xi**(-(17-4*k)/(12-3*k))
#            gamma_test2 = np.sqrt((g0b-1)**2/xi)+1
#            gamma_test2 = np.sqrt((g0b*g0b-1)/xi+1)
#            gamma_test2 = np.sqrt((g0b**2.-1.)*(1./xi-1./(1.+4.*(m+1.)*g0b**2.))/(1.-1./(1.+4.*(m+1.)*g0b**2.)) +1.)
            gamma_test2 = np.sqrt(g0b*g0b/xi)

            w=np.where(gamma_test2 <= 1.)
            gamma_test2[w]=1.
            #densprime_test2[w]=0.
            dens_test2 = densprime_test2/gamma_test2

            beta_test2 = np.sqrt(1-gamma_test2**(-2))
            
#            beta_test2 = np.sqrt(1-gamma_test2**(-2))*rxi0
#            gamma_test2 = 1./(1.-beta_test2*beta_test2)
            





            #*********************
            # Taylor approximation to Sedov-Taylor structure
            # Using some approximation to sedov solution dehind shock, rather than blandford-mckee ?? - Brian, 3/18/2016
            gammat=4./3.
            ntay=(7.*gammat-1.)/(gammat**2.-1.)                                                                                      
            pt=2.*(gammat+5.)/(7.-gammat)                                                                                         
            qt=(2.*gammat**2.+7.*gammat-3.)/(7.-gammat) 
            st=(gammat+1.)/(7.-gammat)                                                                                            

          # Using extended Taylor approximations, for any external density.  From Petruk 2000 A&A 357, 686.  His m is our k - Brian, 8/01/2016

            nn=2  # spherical coordinates
            ntaye = ((2-nn)*gammat**2. + (3.*nn+1)*gammat - 1 - k*gammat*(gammat+1)) / (gammat**2.-1.)
            pte = (2.*(gammat + 2.*nn + 1 - k*(gammat+1.))) / (3.*nn + 1. - (nn-1.)*gammat - k*(gammat+1.))
            qte = (2.*gammat**2. + (3.*nn+1.)*gammat) - (nn+1.) - (k*gammat*(gammat+1.)) / (3.*nn + 1. - (nn-1.)*gammat - k*(gammat+1.))
            ste = (gammat+1.) / (3.*nn + 1. - (nn-1.)*gammat - k*(gammat+1.))

            ntay=ntaye
            pt=pte
            qt=pte
            st=ste


            # With new coefficients, works for any external density. - Brian, 8/01/2016
  
            #rxi0[np.where(rxi0 > 1.)]=1.
            
            rxi = g0b**2.*(1./g0b**2.+rxi0-1.)  # similarity variable to take gamma into account.  Not valid beyond 1./g0b**2. from front edge
#            rxi = rxi0
            rxi_ww0 = np.where(rxi <= 0.)
            rxi[rxi_ww0]=0.
            rxi_ww1 = np.where(rxi > 1.)
            rxi[rxi_ww1] = 1.
            
    
            print('NUMBERS = ',gammat,ntay,pt,qt,st)
            print('rxi0.shape = ',rxi0.shape)
            print('g0b**2. = ',g0b[98:,:,0]**2.)
            print('rxi0 = ',rxi0[98:,:,0])
            print('rxi = ',rxi[98:,:,0])


            densprime_test3 = bigr**(-k)*rho_k * 3.*(4./3.*g0b+1.)*g0b * rxi**(3./(gammat-1.))*((gammat+1.)/gammat - rxi**(ntay-1.)/gammat)**(-pt) * (rxi/rxi0)**2.  # for area change? --- - Brian, 5/4/2016
            #densprime_test3 = bigr**(-k)*rho_k * 3.*(4./3.*g0b+1.)*g0b #* rxi**((3.-k*gammat)/(gammat-1.))*((gammat+1.)/gammat - rxi**(ntay-1.)/gammat)**(-pt) #* (rxi/rxi0)**2.  # for area change? --- - Brian, 5/4/2016
        
            #densprime_test3 = bigr**(-k)*rho_k * 3.*(4./3.*g0b+1.)*g0b * rxi**((3.-k*gammat)/(gammat-1.))*((gammat+1.)/gammat - rxi**(ntay-1.)/gammat)**(-pt) * (rxi/rxi0)**2.  # for area change? --- - Brian, 5/4/2016
            #densprime_test3 = densprime_test2

            pres_test3=bigr**(-k)*rho_k * (g0b-1.e0)*(4./3.*g0b+1.e0) * ((gammat+1.)/gammat - rxi**(ntay-1.)/gammat)**(-qt) * (rxi/rxi0)**2.  # for area change? --- - Brian, 5/4/2016
            vel_taylor = v0b * (gammat+1.)/2. * (rxi/gammat + (gammat-1.)/(gammat+1.)*rxi**ntay/gammat)
            gamma_test3=np.sqrt(1.e0/(1.e0-vel_taylor**2.e0))  # sqrt((g0b-1.d0)^2.d0/xi)+1.d0
            dens_test3=densprime_test3/gamma_test3

            #pres_test3[np.where(rxi <= 0.)]=0.
#            dens_test3[np.where(rxi <= 0.)]=1.e-40
            #gamma_test3[np.where(rxi <= 0.)]=1.
            
            #pres_test2[np.where(pres_test3 < 0.)]=0.
            #gamma_test2[np.where(pres_test3 < 0.)]=0.

            
            # need a new value for beta also - Brian, 12/18/2013
            beta_test3=np.sqrt(1e0-gamma_test3**(-2e0))

            dens_test3[np.where(dens_test3 <= 0.)]=1.e-45
            pres_test3[np.where(pres_test3 <= 0.)]=1.e-40

            dens_test3[rxi_ww0]=1.e-45
            pres_test3[rxi_ww0]=1.e-40
            dens_test3[rxi_ww1]=1.e-45
            pres_test3[rxi_ww1]=1.e-40
            
            
            #print('vel_taylor = ',vel_taylor)
            #print('v0b = ',v0b)
            #print('p factor = ',((gammat+1.)/gammat - rxi**(ntay-1.)/gammat)**(-qt))
            
            dens_factor = rxi**((3.-k*gammat)/(gammat-1.))*((gammat+1.)/gammat - rxi**(ntay-1.)/gammat)**(-pt)
            #print('dens factor = ',rxi**(3./(gammat-1.))*((gammat+1.)/gammat - rxi**(ntay-1.)/gammat)**(-pt))
            print('dens factor = ',dens_factor[98:,:,0])
            print('gamma_test2 = ',gamma_test2[98:,:,0])
            print('gamma_test3 = ',gamma_test3[[0,1,98,99],:,0])
            print('beta_test3 = ',beta_test3[[0,1,98,99],:,0])
            print('dens_test3 = ',dens_test3[[0,1,98,99],:,0])
            print('pres_test3 = ',pres_test3[[0,1,98,99],:,0])
            print('dens_test2 = ',dens_test2[[0,1,98,99],:,0])
            print('pres_test2 = ',pres_test2[[0,1,98,99],:,0])
            


  #*********************
  # Interpolate between BM and ST solutions.  I do not think the interpolation is really correct.
  # Using thus makes the afterglow too bright at late times, with a shallower decay - Brian, 11/3/2019

            densprime_test4 = densprime_test2*v0b**2. + densprime_test3*(1.-v0b**2.)
            pres_test4 = pres_test2*v0b**2. + pres_test3*(1.-v0b**2.)
            beta_test4 = np.sqrt(beta_test2**2.*v0b**2. + beta_test3**2.*(1.-v0b**2.))
            gamma_test4 = np.sqrt(1.e0/(1.e0-beta_test4**2.e0))
            dens_test4 = densprime_test4/gamma_test4
  

  #*********************
  # Pick either BM and ST solutions.  - Brian, 9/29/2019
  # This uses a switch to use either the BM or ST structure behind the shock.  
  # It is set to switch when the velocity just behind the sock is 0.5 c - Brian, 11/3/2019
            

            #pres_test2[:]=0
    
            v0b_switch = 0.999999999
            densprime_test5 = np.where((v0b > v0b_switch),densprime_test2,densprime_test3)
            pres_test5 = np.where((v0b > v0b_switch),pres_test2,pres_test3)
            beta_test5 = np.where((v0b > v0b_switch),beta_test2,beta_test3)
            gamma_test5 = np.where((v0b > v0b_switch),gamma_test2,gamma_test3)
            dens_test5 = np.where((v0b > v0b_switch),dens_test2,dens_test3)



  #*********************
  # Interpolate between BM and ST solutions.  I do not think the interpolation is really correct.
  # Using thus makes the afterglow too bright at late times, with a shallower decay - Brian, 11/3/2019
  # For gamma, do a different interpolation. - Brian, 3/20/2021

            densprime_test6 = densprime_test2*v0b**2. + densprime_test3*(1.-v0b**2.)
            pres_test6 = pres_test2*v0b**2. + pres_test3*(1.-v0b**2.)
            gamma_test6 = np.sqrt((g0b-1)**2/xi)+1  # This is better, but still needs to drop off at large xi - Brian, 3/16/2021
            beta_test6 = np.sqrt(1-1/gamma_test6**2)
            dens_test6 = densprime_test6/gamma_test6

            

  #*********************
  # Start with 6, but do powerlaws for densprime, pres, gamma
  # This doesn't work.  If I want to do a powerlaw, I need to calculate densprime, pres, and gamma 
  # at xi=1 and xi=1.01 and fine the powerlaw in between. - Brian, 3/20/2021

    
            xi_0 = 1.
            xi_1 = 1.01
            rxi0_0 = 1.
            rxi0_1 = 1 + (1-xi_1)/(4*(m+1)*g0b**2)
            rxi_0 = g0b**2.*(1./g0b**2.+rxi0_0-1.)  
            rxi_1 = g0b**2.*(1./g0b**2.+rxi0_1-1.)  
            
            
            densprime_test2_0 = bigr**(-k) * rho_k * 3 * (4/3 * g0b+1) * g0b*xi_0**(-(7-2*k)/(4-k))
            pres_test2_0 = bigr**(-k)*rho_k* (g0b-1)*(4/3 *g0b + 1) * xi_0**(-(17-4*k)/(12-3*k))
            gamma_test2_0 = np.sqrt(g0b*g0b/xi_0)

            densprime_test2_1 = bigr**(-k) * rho_k * 3 * (4/3 * g0b+1) * g0b*xi_1**(-(7-2*k)/(4-k))
            pres_test2_1 = bigr**(-k)*rho_k* (g0b-1)*(4/3 *g0b + 1) * xi_1**(-(17-4*k)/(12-3*k))
            gamma_test2_1 = np.sqrt(g0b*g0b/xi_1)

            
            densprime_test3_0 = bigr**(-k)*rho_k * 3.*(4./3.*g0b+1.)*g0b * rxi_0**(3./(gammat-1.))*((gammat+1.)/gammat - rxi_0**(ntay-1.)/gammat)**(-pt) * (rxi_0/rxi0_0)**2.  # for area change? --- - Brian, 5/4/2016
            pres_test3_0=bigr**(-k)*rho_k * (g0b-1.e0)*(4./3.*g0b+1.e0) * ((gammat+1.)/gammat - rxi_0**(ntay-1.)/gammat)**(-qt) * (rxi_0/rxi0_0)**2.  # for area change? --- - Brian, 5/4/2016
            vel_taylor_0 = v0b * (gammat+1.)/2. * (rxi_0/gammat + (gammat-1.)/(gammat+1.)*rxi_0**ntay/gammat)
            gamma_test3_0=np.sqrt(1.e0/(1.e0-vel_taylor_0**2.e0))  # sqrt((g0b-1.d0)^2.d0/xi)+1.d0

            densprime_test3_1 = bigr**(-k)*rho_k * 3.*(4./3.*g0b+1.)*g0b * rxi_1**(3./(gammat-1.))*((gammat+1.)/gammat - rxi_0**(ntay-1.)/gammat)**(-pt) * (rxi_1/rxi0_1)**2.  # for area change? --- - Brian, 5/4/2016
            pres_test3_1=bigr**(-k)*rho_k * (g0b-1.e0)*(4./3.*g0b+1.e0) * ((gammat+1.)/gammat - rxi_1**(ntay-1.)/gammat)**(-qt) * (rxi_1/rxi0_1)**2.  # for area change? --- - Brian, 5/4/2016
            vel_taylor_1 = v0b * (gammat+1.)/2. * (rxi_1/gammat + (gammat-1.)/(gammat+1.)*rxi_1**ntay/gammat)
            gamma_test3_1=np.sqrt(1.e0/(1.e0-vel_taylor_1**2.e0))  # sqrt((g0b-1.d0)^2.d0/xi)+1.d0

            
            densprime_test6_0 = densprime_test2_0*v0b**2. + densprime_test3_0*(1.-v0b**2.)
            pres_test6_0 = pres_test2_0*v0b**2. + pres_test3_0*(1.-v0b**2.)
            gamma_test6_0 = np.sqrt((g0b-1)**2/xi_0)+1  # This is better, but still needs to drop off at large xi - Brian, 3/16/2021

            densprime_test6_1 = densprime_test2_1*v0b**2. + densprime_test3_1*(1.-v0b**2.)
            pres_test6_1 = pres_test2_1*v0b**2. + pres_test3_1*(1.-v0b**2.)
            gamma_test6_1 = np.sqrt((g0b-1)**2/xi_1)+1  # This is better, but still needs to drop off at large xi - Brian, 
            
            
            densprime_test6_plaw = (np.log(densprime_test6_0/densprime_test6_1)/np.log(xi_0/xi_1))
            pres_test6_plaw = (np.log(pres_test6_0/pres_test6_1)/np.log(xi_0/xi_1))
            gamma_test6_plaw = (np.log(gamma_test6_0/gamma_test6_1)/np.log(xi_0/xi_1))
            
            densprime_test8 = densprime_test6_0*xi**densprime_test6_plaw
            pres_test8 = pres_test6_0*xi**pres_test6_plaw
            gamma_test8 = gamma_test6_0*xi**gamma_test6_plaw
            
            w=np.where(gamma_test8 <= 1.)
            gamma_test8[w]=1.
            densprime_test8[w]=0.

            beta_test8 = np.sqrt(1-1/gamma_test8**2)
            dens_test8 = densprime_test8/gamma_test8

            #print('SHAPES = ',densprime_test6.shape,densprime_test6_plaw.shape )
            #print('PLAW densprime = ',densprime_test6_plaw)
            #print('PLAW pres = ',pres_test6_plaw)
            #print('PLAW gamma = ',gamma_test6_plaw)

            
            
            
            xi_max = 1+4*(m+1)*g0b**2.
            
            # Old way of doing this. - Brian, 6/29/2021
            #gamma_test9 = (gamma_test6-1) * (np.exp(-(xi-1)**2/xi_max) ) + 1
            #beta_test9 = np.sqrt(1-1/gamma_test9**2)
            #dens_test9 = densprime_test6/gamma_test9
            
            # New way, works better - Brian, 6/29/2021
            beta_test9 = beta_test4 * (np.exp(-(xi-1)**2/xi_max))**.75
            gamma_test9 = np.sqrt(1.e0/(1.e0-beta_test9**2.e0))
            dens_test9 = densprime_test4/gamma_test9


            
            # Thin shell approximation
            
            M_sweep = bigr**(3-k) * rho_k * 4*np.pi/3
            rho_f = bigr**(-k) * rho_k * 3 * (4/3 * g0b+1)
            R_contact = (bigr**3 - M_sweep/(4*np.pi/3)/rho_f)**(1/3)

            #xi_max = 2
            #densprime_test10 = np.where(xi > xi_max, bigr**(-k) * rho_k * 3 * (4/3 * g0b+1) * g0b, 1.e-45)
            #pres_test10 = np.where(xi > xi_max, bigr**(-k)*rho_k* (g0b-1)*(4/3 *g0b + 1), 1.e-40)
            #gamma_test10 = np.where(xi > xi_max, g0b, 1.)

            densprime_test10 = np.where(smallr > R_contact, bigr**(-k) * rho_k * 3 * (4/3 * g0b+1) * g0b, 1.e-45)
            pres_test10 = np.where(smallr > R_contact, bigr**(-k)*rho_k* (g0b-1)*(4/3 *g0b + 1), 1.e-40)
            gamma_test10 = np.where(smallr > R_contact, g0b, 1.)

            w=np.where(gamma_test10 <= 1.)
            gamma_test2[w]=1.
            dens_test10 = densprime_test10/gamma_test10

            beta_test10 = np.sqrt(1-gamma_test10**(-2))
            
            

  #*********************
  # Here, we select which version of the internal strucure to use.  The default is densprime_test2 etc.
  # Don't use sedov for this one - Brian, 5/9/2016
  # For now, easiest to just stick with the BM structure at all times - Brian, 11/3/2019

            densprime_test2=densprime_test4
            pres_test2=pres_test4
            gamma_test2=gamma_test9
            dens_test2=dens_test9
            beta_test2=beta_test9

            #densprime_test2=densprime_test4
            #pres_test2=pres_test4
            #gamma_test2=gamma_test4
            #dens_test2=dens_test4
            #beta_test2=beta_test4

            # Thin shell
            #densprime_test2=densprime_test10
            #pres_test2=pres_test10
            #gamma_test2=gamma_test10
            #dens_test2=dens_test10
            #beta_test2=beta_test10




            ##build a glow array for use in afterglow code
            
#            glow = np.zeros((12, s_points, z_points, phi_slices))
            glow = np.zeros((12, z_points, s_points, phi_slices))            

            glow[0,:,:,:] = z_grid
            glow[1,:,:,:] = s_grid
            glow[2,:,:,:] = smallr
            glow[3,:,:,:] = theta_grid
            glow[4,:,:,:] = np.arccos((np.cos(theta_grid) - beta_test2)/(1-beta_test2*np.cos(theta_grid)))
            glow[5,:,:,:] = dens_test2/gv.mp
            glow[6,:,:,:] = pres_test2/(4/3-1)*gv.cl**2
            glow[7,:,:,:] = gamma_test2
            glow[8,:,:,:] = bigr
            glow[9,:,:,:] = xi
            glow[10,:,:,:] = beta_test2
            glow[11,:,:,:] = e_iso_interp
            
            return glow
