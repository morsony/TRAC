import numpy as np
import pickle
from scipy import optimize
from scipy import integrate
from scipy import interpolate
from matplotlib import pyplot as plt
from numba import jit

from global_var import sysvars as gv
from blandford_mckee_full_brian_wind import *
from jitter_python import *

########################################

#This files contains:
#    afterglow_lineofsight, which integrates radiation along one dimension
#    afterglow_total_power, which calls jitter, to get emission and absorption, and afterglow_linedofsight to integrate them
#    afterglow_aniso_bmk, which loops aver time, creating the hydro parameters (calls blandford_mckee_full) and calculates the resulting radiation ( calls afterglow_total power)

#######################################    

def afterglow_lineofsight( alpha, j, omegas, init, pos, thetaprime, shockbeta, 
                          shockgamma, n, glow, eps, 
                          eqparb, k_index, pindex, side=-1):
                 
            # Funciton to calculate equation 1.3 from Rybicki and Lightman to
            # calculate the specific intesity due to a constant soursce funciton
            # with application to prompt bursts in GRBs
            # this for a single angle
                              
            #PARAMETERS
            # alpha -- is the com-moving absorption cosfficient
            # j -- is the co-moving emmision coefficient
            # omegas -- the array of co-moving frequenies used
            # init -- is the initial intensity, 0 is fine
            # pos -- position of points in te shock
            # thetaprime -- is the angle in the co-moving frame - map this properly
            # shockbeta -- is the co-moving beta
            # shockgamma -- self explanatory
            
            # for afterglow_lineofsight version
            # alpha is a 2D array over frequency and position
            # j is a 2D array over frequency and position
            # ds, thetaprime, shockbeta, shock gamma are all 1D arrays over position
            
            # find out how many frequencies
            num_freq = len(omegas)
            num_slice = len(shockgamma)
            
            #find the number of positional points
            nx = len(thetaprime)
            
            # Create an array to hold the optical depth
#            tau = np.zeros((num_freq, nx))
#            dtau = np.zeros((num_freq, nx))
#            x = np.zeros((num_freq, nx))
            
            # create the array for the co-moving intensity
            coint = np.zeros((num_freq))
            
            # create the array for the observed intensity
            obint = np.zeros((num_freq))
            
            # Boost factor for the co-moving to observed frame
            boost = shockgamma * (1 + shockbeta*np.cos(thetaprime))
            
#                        boost = glow[7,i,j]*(1 + glow[10,i,j]*np.cos(glow[4,i,j]))

#            omegasobs = omegas * boost[0]
            omegasobs = omegas
            
            sigma_t = 6.6524e-25
            alpha_t = n*sigma_t*0
            
            # Lorentz factor just behind the shock
            g0b = np.sqrt((glow[7,:] - 1)**2 * glow[9,:])+1
            
            #Lorentz factor of the shock
            g_ad=4./3.
            gs0b = np.sqrt((g0b+1)*(g_ad * (g0b-1) + 1)**2 / (g_ad * (2-g_ad)*(g0b-1)+2))
            
            #Beta (velocity) of the shock
            vs0b = np.sqrt(1 - 1/gs0b**2)
            
            #Using velocity difference
            time_lab = (glow[8,:] - glow[2,:]) / (vs0b-glow[10,:])/gv.cl   # time to travel from shock to current location in lab frame
            time = time_lab/glow[7,:]                         # time to travel from shock in current comoving frame

            time0 = glow[8,:]/(1.-1./(2.*(4.-k_index)*gs0b*gs0b))/gv.cl   # rest frame time of shock to reach bigR (approximate, BM eqn. 26)

            t_shock = time0*(glow[2,:]/glow[8,:])  # time when particles were initially shocked (approximate)
#            r_shock = gv.cl*(t_shock)*(1.-1./(2.*(4.-k_index)*gs0b*gs0b))  # radius when particles were initially shocked (approximate)  Just gives back smallr :(
            r_shock = glow[2,:]*glow[9,:]**(-1/(4-k_index)) # radius when particles were initially shocked (approximate)  Granot et al. 2002 eqn. A8
            time_shock2 = (glow[2,:]-r_shock)/(glow[10,:]*gv.cl)/glow[7,:]

            time_shock = (time0-t_shock)/glow[7,:]   # time since particles were shocked (approximate)
            gamma_shock = gs0b*(glow[8,:]/r_shock)**((3.-k_index)/2.)   # Lorentz factor of shock when particles were shocked (approximate) (compatable with Granot et al. 2002)
#            gamma_shock = gs0b*(glow[9,:])**((3.-k_index)/(2*(4.-k_index)))   # Lorentz factor of shock when particles were shocked, using Granot et al. 2002
            
            ener_b = glow[6,:] * eqparb
            b = np.sqrt(ener_b*8*np.pi)
            
#            ener_b0 = glow[6,:] * eqparb/glow[9,:]**(-(17-4*k_index)/(12-3*k_index))   # magnetic field energy when particles where shocked (approximate)
            ener_b0 = glow[6,:] * eqparb/glow[9,:]**(-2*(13-4*k_index)/(12-3*k_index))   # magnetic field energy when particles where shocked, using Granot et al. 2002 eqn. A9
            b0 = np.sqrt(ener_b0*8*np.pi)
            
            ener_b_shock = ener_b0*(gamma_shock/gs0b)**2.    # magnetic field energy when particles were shocked (approximate)
            b_shock = np.sqrt(ener_b_shock*8*np.pi)


            #Using B field at shock front
            gammamaxbeta2_old = 3/4 * gv.me*gv.cl/ sigma_t/ener_b0/ time #* (1.-gv.cl*time/glow[8,:])**(3.*(3.-k_index)/2.)
            gammamax_old = 0.5 * (gammamaxbeta2_old + np.sqrt(gammamaxbeta2_old**2 + 4))
            omega_max_old = 3/2 * gv.q/gv.me/gv.cl*b*gammamax_old**2  # b or b0 ?? - Brian, 5/22/2017

            gammamaxbeta2 = 3/4 * gv.me*gv.cl/ sigma_t/ener_b0/ time * (1.-gv.cl*vs0b*time/glow[8,:]*glow[7,:])**(3.*(3.-k_index)/2.)
            gammamax = 0.5 * (gammamaxbeta2 + np.sqrt(gammamaxbeta2**2 + 4))
            omega_max = 3/2 * gv.q/gv.me/gv.cl*b*gammamax**2  # b or b0 ?? - Brian, 5/22/2017
            
#            gammamaxbeta2_shock = 3/4 * gv.me*gv.cl/ sigma_t/ener_b_shock/ time_shock   #  Time isn't exact, especially at large xi, but should be ok? - Brian, 5/23/2017
            gammamaxbeta2_shock = 3/4 * gv.me*gv.cl/ sigma_t/ener_b_shock/ time_shock2 #time #time_shock  # compromise.  Time isn't exact, especially at large xi, but should be ok? - Brian, 5/23/2017
            gammamax_shock = 0.5 * (gammamaxbeta2_shock + np.sqrt(gammamaxbeta2_shock**2 + 4))
            omega_max_shock = 3/2 * gv.q/gv.me/gv.cl*b*gammamax_shock**2  # b or b0 ?? - Brian, 5/22/2017

#            gammamax_sari = 6*np.pi*gv.me*gv.cl/(sigma_t*b**2*glow[7,:]*1000)   # at t_obs = 1000 seconds - Brian, 6/22/2017
            xi = np.copy(glow[9,:])
            kk=k_index
            time_sari = time0 #glow[2,:]/(glow[10,:]*gv.cl)
            time0_sari = time_sari * xi**(-1/(4-kk)) #glow[2,:]/gv.cl #np.copy(time_sari) * xi**(1/(4-kk))
            g0_sari = glow[7,:] * xi**((7-2*kk)/(2*(4-kk)))
            gammamax_sari = 2*(19-2*kk)*np.pi*gv.me*gv.cl/(sigma_t) * g0_sari/(b0**2*time0_sari) * xi**((25-2*kk)/(6*(4-kk))) / (xi**((19-2*kk)/(3*(4-kk)))-1)   # Granot et al 2002 eqn. A12 - Brian, 6/22/2017
            omega_max_sari =  3/2 * gv.q/gv.me/gv.cl*b*gammamax_sari**2  # b or b0 ?? - Brian, 5/22/2017

            gammamax_sari2 = 2*(19-2*kk)*np.pi*gv.me*gv.cl/(sigma_t) * glow[7,:]/(b**2*time0_sari) * xi**((7-2*kk)/(2*(4-kk))) * xi**(-2*(13-kk)/(3*(4-kk))) * xi**((25-2*kk)/(6*(4-kk))) / (xi**((19-2*kk)/(3*(4-kk)))-1)   # Granot et al 2002 eqn. A12 - Brian, 6/22/2017
            omega_max_sari2 =  3/2 * gv.q/gv.me/gv.cl*b*gammamax_sari2**2  # b or b0 ?? - Brian, 5/22/2017


            gammamaxbeta2=gammamaxbeta2_shock
            gammamax=gammamax_shock
            omega_max=omega_max_shock

            gammamax=gammamax_sari
            omega_max=omega_max_sari #/boost

#            print('omega_max',gammamax_old,gammamax_shock,glow[9,:],glow[8,:]-glow[2,:],vs0b-glow[10,:],time,time_shock,time0,t_shock,gammamax_sari,time_sari,time0_sari)
          #!#  print('omega_max',omega_max_sari,omega_max_shock,glow[9,:])

#            plt.plot(glow[9,:]-1,omega_max_sari*boost)
#            plt.plot(glow[9,:]-1,omega_max_shock*boost)
#            plt.plot(glow[9,:]-1,omega_max_sari2*boost)
#            plt.plot(glow[9,:]-1,omega_max[-1]*((glow[9,:]-1)/(glow[9,-1]-1))**(-2))
#            plt.plot(glow[9,:]-1,time0_sari/glow[7,:])
#            plt.plot(glow[9,:]-1,time_sari/glow[7,:])
#            plt.plot(glow[9,:]-1,time)
#            plt.plot(glow[9,:]-1,time_shock)
#            plt.plot(glow[9,:]-1,time_shock2)
#            plt.plot(glow[9,:]-1,glow[2,:]/vs0b/gv.cl)
#            plt.plot(omega_max_old)
#            plt.yscale('log')
#            plt.xscale('log')
#            plt.show()
            
            # Have now calculate lower limit for gammamax and omega_max where cooling is important - Brian 7/8/2016
            
            # also need gammain, opmega at gammain for fast cooling - Brian 9/30/2015
            #It would be better to call the same gammain and omega_min procedure as used for calculating the synchrotron radiation - Brain 7/8/2016
            
            ethermal = glow[6,:]
            nodensity_e = glow[5,:]
#            gammamin = (((pindex-2)/(pindex-1)) * eps * ethermal/ (nodensity_e*gv.me*gv.cl**2)) + 1
#            omega_min = 3/2 * gv.q/gv.me/gv.cl * b * gammamin**2
#            constant = (pindex - 1)*nodensity_e*gammamin**(pindex-1)
 
            # Now calling the same procedure used for calculating synchrotron radiation.  Already have nodensity_e, so just pass gammaint=0.25 - Brian, 5/18/2017

            values = elecdist(eps, ethermal,0.25,nodensity_e)   # get the minimum gamma and distribution constant
            gammamin = values['gammamin']                #lower limit for integration
            constant = values['constant']                      # constant in electron distribution
            omega_min = 3/2 * gv.q/gv.me/gv.cl * b * gammamin**2

#            gammamax=gammamin*10   # don't forget to remove this - Brian, 6/23/2017
#            omega_max = 3/2 * gv.q/gv.me/gv.cl*b*gammamax**2
            

            #Use this to turn off cooling
            #omega_max[:] = 1e50
            
            #print('omage_max=',omega_max,time)
            
            #now loop overfrequencies - Brian 7/8/2016

#            print(omega_min)
#            print(omega_max)
#            plt.plot(omega_min)
#            plt.show()
            
            jc=np.copy(j)
            alphac=np.copy(alpha)

#            plt.plot(omega_max*boost)
#            plt.yscale('log')
#            plt.show()


# Can I vectorize this loop - Brian, 7/9/2020

            boost_tile = np.repeat(boost,num_freq)
            boost_tile = np.reshape(boost_tile,(num_slice,num_freq))
            #alpha_t_tile = np.broadcast_to(alpha_t,(alpha_t.size,num_freq))
            #alpha_t_tile = np.tile(alpha_t,(1,num_freq))
            alpha_t_tile = np.repeat(alpha_t,num_freq)
            alpha_t_tile = np.reshape(alpha_t_tile,(num_slice,num_freq))
            
            omega_max_tile = np.repeat(omega_max,num_freq)
            omega_max_tile = np.reshape(omega_max_tile,(num_slice,num_freq))
            
            omega_min_tile = np.repeat(omega_min,num_freq)
            omega_min_tile = np.reshape(omega_min_tile,(num_slice,num_freq))
            
            
            #print('boost shapes = ',boost.shape,boost_tile.shape,j.shape)
            #print(boost)
            #print(boost_tile[:,0])

            j = np.copy(j) * boost_tile**2
            #alpha[:,i] = np.copy(alpha[:,i])/boost + np.copy(alpha_t[:])/boost  # /9?
            alpha = np.copy(alpha)/boost_tile + np.copy(alpha_t_tile)/boost_tile  # /9?

            # error handeling
            temp = (0 == np.isfinite(j)).nonzero()
            j[temp] = 1e-90 #0
            alpha[temp] = 1e-60 #90
                    
            temp = (0 == np.isfinite(alpha)).nonzero()
            j[temp] = 1e-90 #0
            alpha[temp] = 1e-60 #90    
            
            temp = (0 == np.isfinite(boost_tile)).nonzero()
            j[temp] = 1e-90 #0
            alpha[temp] = 1e-60 #90
#           boost[temp] = 1 #0
            
                #this is correct for the Lorentz transformation to the observer's frame
                
                # adding synchrotron cooling, somewhat crude version
                # this does not take into account the case where gammamax < gammamin (strong cooling) - Brian 9/30/15

            omegasobs_tile = np.tile(omegasobs,num_slice)
            omegasobs_tile = np.reshape(omegasobs_tile,(num_slice,num_freq))
            #print('omegasobs = ',omegasobs.shape,omegasobs_tile.shape)
            #print(omegasobs_tile[0,:])
            
                
            temp_lt = ((omega_max_tile*boost_tile) < omegasobs_tile) #.nonzero()  # omegasobs, not omegas, to include boost - Brian, 6/29/2017
            
            #print('temp_lt.shape = ',temp_lt.shape)
            
            temp_lt_roll = np.roll(temp_lt,2,axis=0)
            temp_lt_roll[0:2,:] = 0
            temp_lt_roll[-1,:] = 0
            
            #print('temp_lt = ')
            #print(temp_lt[:,100])
            #print(temp_lt_roll[:,100])
            
            #temp_lt2 = (temp_lt and 
            
            #print('temp_lt.shape = ',temp_lt.shape)
            
            temp = temp_lt.nonzero()
            
            #print('temp[0].shape = ',temp[0].shape)
            
#                if temp[0] != -1:

#                print(temp)
#                plt.plot([0,1])
#                plt.show()

            jc=np.copy(j)   # need this to get the right boost factors - Brian, 6/29/2017
            alphac=np.copy(alpha)

            # Inside jj loop
        
            temp_jc_values = np.zeros(temp[0].size)
            
            temp_jc_values = np.where(omegasobs_tile[temp[0],temp[1]] < omega_min_tile[temp[0],temp[1]] 
                                      * boost_tile[temp[0],temp[1]],
                                   jc[temp[0],temp[1]] * (omegasobs_tile[temp[0],temp[1]] 
                                           / (omega_max_tile[temp[0],temp[1]] * boost_tile[temp[0],temp[1]]))**(-5/6),
                                   jc[temp[0],temp[1]] * (omegasobs_tile[temp[0],temp[1]] 
                                           / (omega_min_tile[temp[0],temp[1]]*boost_tile[temp[0],temp[1]]))**-0.5 * 
                                           (omega_min_tile[temp[0],temp[1]] / omega_max_tile[temp[0],temp[1]])**(-5/6) )
                                   
            temp_jc_values = np.where(omega_min_tile[temp[0],temp[1]] < omega_max_tile[temp[0],temp[1]],
                                   jc[temp[0],temp[1]] * (omegasobs_tile[temp[0],temp[1]] 
                                           / (omega_max_tile[temp[0],temp[1]] * boost_tile[temp[0],temp[1]]))**-0.5,
                                   temp_jc_values )
            
            
            temp_alphac_values = np.zeros(temp[0].size)
            
            temp_alphac_values = np.where(omegasobs_tile[temp[0],temp[1]] < omega_min_tile[temp[0],temp[1]] 
                                         * boost_tile[temp[0],temp[1]],
                                     alphac[temp[0],temp[1]] * (omegasobs_tile[temp[0],temp[1]] 
                                        / (omega_max_tile[temp[0],temp[1]] * boost_tile[temp[0],temp[1]]))**(-5/6),
                                     alphac[temp[0],temp[1]] * (omegasobs_tile[temp[0],temp[1]] 
                                        / (omega_min_tile[temp[0],temp[1]] * boost_tile[temp[0],temp[1]]))**-0.5 
                                        * (omega_min_tile[temp[0],temp[1]] / omega_max_tile[temp[0],temp[1]])**(-5/6) )
                                   
            temp_alphac_values = np.where(omega_min_tile[temp[0],temp[1]] < omega_max_tile[temp[0],temp[1]],
                                   alphac[temp[0],temp[1]] * (omegasobs_tile[temp[0],temp[1]] 
                                        / (omega_max_tile[temp[0],temp[1]] * boost_tile[temp[0],temp[1]]))**-0.5,
                                          temp_alphac_values )
            
            jc[temp[0],temp[1]] = temp_jc_values
            alphac[temp[0],temp[1]] = temp_alphac_values
            
            cutoff=1

            if (cutoff == 1):
            #    
            #    # Now what do I do? 
            #    for jj in range(2,len(temp[0])-1): #2  #3 
                
                temp = temp_lt_roll.nonzero()
                
                temp_jc_values = np.zeros(temp[0].size)
                
                #print( (omega_max_tile[temp[0]-1,temp[1]]*boost_tile[temp[0]-1,temp[1]] 
                #                          < omegasobs_tile[temp[0],temp[1]]) )
                #print( (glow[9,temp[0]-1] < glow[9,temp[0]]) ) 
                
                temp_jc_values = np.where( np.logical_and((omega_max_tile[temp[0]-1,temp[1]]*boost_tile[temp[0]-1,temp[1]] 
                                          < omegasobs_tile[temp[0],temp[1]]) , (glow[9,temp[0]-1] < glow[9,temp[0]])) ,
                                       jc[temp[0],temp[1]] * (omegasobs_tile[temp[0],temp[1]] 
                                          / (omega_max_tile[temp[0]-1,temp[1]] * boost_tile[temp[0]-1,temp[1]]))**-1. , 
                                       jc[temp[0],temp[1]] )
                
                temp_alphac_values = np.zeros(temp[0].size)
                
                temp_alphac_values = np.where( np.logical_and((omega_max_tile[temp[0]-1,temp[1]]*boost_tile[temp[0]-1,temp[1]] 
                                          < omegasobs_tile[temp[0],temp[1]]) , (glow[9,temp[0]-1] < glow[9,temp[0]])) ,
                                       alphac[temp[0],temp[1]] * (omegasobs_tile[temp[0],temp[1]] 
                                          / (omega_max_tile[temp[0]-1,temp[1]] * boost_tile[temp[0]-1,temp[1]]))**-1. , 
                                       alphac[temp[0],temp[1]] )
                
                jc[temp[0],temp[1]] = temp_jc_values
                alphac[temp[0],temp[1]] = temp_alphac_values
                

                # Trying to take fast cooling into account.  If omega_max < omega_min in the box ahead, set j and alppha to zero - Brian, 1/23/2023
                
                temp_lt = ((omega_max_tile*boost_tile) > 0) #.nonzero()  # omegasobs, not omegas, to include boost - Brian, 6/29/2017
            
                #print('temp_lt.shape = ',temp_lt.shape)
            
                temp_lt_roll = np.roll(temp_lt,2,axis=0)
                temp_lt_roll[0:2,:] = 0
                temp_lt_roll[-1,:] = 0
            
                temp = temp_lt_roll.nonzero()
                
                temp_jc_values = np.zeros(temp[0].size)

                temp_jc_values = np.where( np.logical_and((omega_max_tile[temp[0]-1,temp[1]]*boost_tile[temp[0]-1,temp[1]] 
                                          < omega_min_tile[temp[0],temp[1]]*boost_tile[temp[0],temp[1]]) , (glow[9,temp[0]-1] < glow[9,temp[0]])) ,
                                          np.zeros(temp_jc_values.shape),
                                          jc[temp[0],temp[1]] )
                
                
                temp_alphac_values = np.zeros(temp[0].size)
                
                temp_alphac_values = np.where( np.logical_and((omega_max_tile[temp[0]-1,temp[1]]*boost_tile[temp[0]-1,temp[1]] 
                                          < omega_min_tile[temp[0],temp[1]]*boost_tile[temp[0],temp[1]]) , (glow[9,temp[0]-1] < glow[9,temp[0]])) ,
                                          np.zeros(temp_jc_values.shape),
                                          alphac[temp[0],temp[1]] )
                
                
                
                jc[temp[0],temp[1]] = temp_jc_values
                alphac[temp[0],temp[1]] = temp_alphac_values
                
                
             
                
                               
    
    
# Can I vectorize this loop - Brian, 7/9/2020
            #for i in range(num_freq):
                


            #    for jj in range(len(temp[0])):
                    
            #        if(omega_min[temp[0][jj]] < omega_max[temp[0][jj]]):
            #            jc[temp[0][jj],i] = jc[temp[0][jj],i] * (omegasobs[i]/(omega_max[temp[0][jj]]*boost[temp[0][jj]]))**-0.5
            #            alphac[temp[0][jj],i] = alphac[temp[0][jj],i] * (omegasobs[i]/(omega_max[temp[0][jj]]*boost[temp[0][jj]]))**-0.5
                        
            #        elif(omegasobs[i] < omega_min[temp[0][jj]]*boost[temp[0][jj]]):
            #            jc[temp[0][jj],i] = jc[temp[0][jj],i] * (omegasobs[i]/(omega_max[temp[0][jj]]*boost[temp[0][jj]]))**(-5/6)
            #            alphac[temp[0][jj],i] = alphac[temp[0][jj],i] * (omegasobs[i]/(omega_max[temp[0][jj]]*boost[temp[0][jj]]))**(-5/6)
                        
            #        else:
            #            jc[temp[0][jj],i] = jc[temp[0][jj],i] * (omegasobs[i]/(omega_min[temp[0][jj]]*boost[temp[0][jj]]))**-0.5 * (omega_min[temp[0][jj]]/omega_max[temp[0][jj]])**(-5/6)                       
            #            alphac[temp[0][jj],i] = alphac[temp[0][jj],i]*(omegasobs[i]/(omega_min[temp[0][jj]]*boost[temp[0][jj]]))**-0.5 * (omega_min[temp[0][jj]]/omega_max[temp[0][jj]])**(-5/6)


            #    cutoff=1


            #    if (cutoff == 1):

            #        for jj in range(2,len(temp[0])-1): #2  #3 

            #            if((omega_max[temp[0][jj-1]]*boost[temp[0][jj-1]] < omegasobs[i]) and (glow[9,temp[0][jj-1]] < glow[9,temp[0][jj]])):

            #                jc[temp[0][jj],i] = jc[temp[0][jj],i] * (omegasobs[i]/(omega_max[temp[0][jj-1]]*boost[temp[0][jj-1]]))**-1. #0.5
            #                alphac[temp[0][jj],i] = alphac[temp[0][jj],i] * (omegasobs[i]/(omega_max[temp[0][jj-1]]*boost[temp[0][jj-1]]))**-1. #0.5

                            
                            
                            
            omega_max = omega_max*boost
            omega_min = omega_min*boost

            omega_max_tile = omega_max_tile*boost_tile
            omega_min_tile = omega_min_tile*boost_tile



            j=np.copy(jc)
            alpha=np.copy(alphac)

            #good to here? - Brian, 7/10/2020
            

#            temp = (0 == np.isfinite(alpha)).nonzero()
#            j[temp] = 1e-90 #0
#            alpha[temp] = 1e-60 #90

#            temp = (0 == np.isfinite(boost)).nonzero()							
#            j[temp] = 1e-90 #0
#            alpha[temp] = 1e-60 #90
#            boost[temp] = 1

#            temp = (0 == np.isfinite(j)).nonzero()
#            j[temp] = 1e-90 #0
#            alpha[temp] = 1e-60 #90

                        
            
#            for iji in range(50):
#                plt.plot(omegasobs,j[iji,:])
#            plt.xscale('log')
#            plt.yscale('log')
#            plt.show()            


            # Loop over frequency

            # Front side
            if (side == 0):
                j=np.concatenate((j,[j[-1,:]]))
                alpha=np.concatenate((alpha,[alpha[-1,:]]))
#                pos=np.append(pos[:],-glow[8,-1]*np.cos(glow[3,-1]))
                pos=np.append(pos[:],pos[-1]-(glow[8,-1]-glow[2,-1]))
#                pos=np.append(pos[:],pos[-1]*1.0001)
                omega_max=np.append(omega_max,omega_max[-1])
                omega_min=np.append(omega_min,omega_min[-1])
            
            # Back side
            if (side == 1):
                j=np.concatenate(([j[0,:]],j))
                alpha=np.concatenate(([alpha[0,:]],alpha))
#                pos=np.append(-glow[8,0]*np.cos(glow[3,0]),pos[:])
                pos=np.append(pos[0]+(glow[8,0]-glow[2,0]),pos[:])
                omega_max=np.append(omega_max[0],omega_max)
                omega_min=np.append(omega_min[0],omega_min)

            nx=len(pos)
            # Create an array to hold the optical depth
            tau = np.zeros((num_freq, nx))
            dtau = np.zeros((num_freq, nx))
            x = np.zeros((num_freq, nx))


            source = j/alpha             # R&L eqn. 6.54
#            source = j/alpha/4/np.pi

#            plt.plot(source)
#            plt.show()



# Can I vectorize this loop - Brian, 7/9/2020
           
            mul = 1
            n_interpol = (nx-1)*mul+1
                
            tau2 = np.zeros((n_interpol,num_freq))
            dtau2 = np.zeros((n_interpol,num_freq))
            tau4 = np.zeros((n_interpol,num_freq))
            dtau4 = np.zeros((n_interpol,num_freq))
                
            x_interp = np.linspace(0, len(pos)-1, num=n_interpol, endpoint=True)
            x_interp2 = np.linspace(0, len(pos)-1, num=len(pos), endpoint=True)
            
            
            x_interp_tile = np.repeat(x_interp,num_freq)
            x_interp_tile = np.reshape(x_interp_tile,(x_interp.size,num_freq))
            
            
            x_interp2_tile = np.repeat(x_interp2,num_freq)
            x_interp2_tile = np.reshape(x_interp2_tile,(x_interp2.size,num_freq))
           
            
            b = pos
            omgmin = omega_min
            omgmax = omega_max

            b_tile = np.repeat(b,num_freq)
            b_tile = np.reshape(b_tile,(b.size,num_freq))
                                
            a = alpha
            jj = j
            omgmin_tile = omega_min_tile
            omgmax_tile = omega_max_tile
            
            temp = (0 == np.isfinite(jj)).nonzero()
#                if temp[0] != -1:
            jj[temp] = 1e-90 #0
                    
            temp = (0 == np.isfinite(b)).nonzero()
#                if temp[0] != -1:
            b[temp] = 0
                    
                    
            ss = jj/a     # R&L eqn. 6.54
#                ss = jj/a/4/np.pi   


# Fixing intervals to account for stupid pythion goign to second index-1, not second index - Brian, 4/11/2017                

#                x = 1-b[0:n_interpol-1]
#                y = a[0:n_interpol-1]
#                y1 = y[0:len(y)-1]
#                y2 = y[1:]
#                x1 = x[0:len(x)-1]
#                x2 = x[1:]
            x = 1-b_tile[0:n_interpol,:]
            y = a[0:n_interpol,:]
            y1 = y[0:len(y)-1,:]
            y2 = y[1:,:]
            x1 = x[0:len(x)-1,:]
            x2 = x[1:,:]

            local_integrals = (x2-x1)*(y1-y2)/(np.log(y1) - np.log(y2))
                
                #adding finite value checking
            temp = (0 == np.isfinite(local_integrals)).nonzero()
#                if temp[0] != -1:
            local_integrals[temp] = 0
                    
            dtau3 = np.cumsum(local_integrals,axis=0)
#                dtau2[1:n_interpol-1] = dtau3
        
            #print('shapes tau = ',local_integrals.shape,tau2.shape,dtau2.shape,dtau3.shape)
            
            dtau2[1:n_interpol,:] = dtau3
    
#                tau2[0:n_interpol-2] = (np.cumsum(local_intergrals[::-1]))[::-1]
            tau2[0:n_interpol-1,:] = (np.cumsum(local_integrals[::-1,:],axis=0))[::-1,:]
            
                
                
            #for k in range(nx-1):
            #        tau[i,k] = tau2[k*mul]
            
            tau[:,:] = tau2[np.arange(nx)*mul,:].T
            
            
            x_arr = -tau2
                
            #print(tau2.shape)
                                
            # calculate dtau
                
            ds = np.zeros(nx)
            ds[0] = 0
            #for k in range(nx):
            #    ds[k] = pos[k-1]-pos[k]
            ds[1:] = pos[0:-1]-pos[1:]
            
            ds_tile = np.repeat(ds,num_freq)
            ds_tile = np.reshape(ds_tile,(ds.size,num_freq))
            
            
            #print('ds shape = ',ds.shape,ds_tile.shape,alpha.shape,dtau.shape,ss.shape)
            
            dtau[:,:] = (alpha[:,:]*ds_tile).T   # Should this be alphac ? - Brian, 7/10/2020
                
            # Optically thick
            coint_thick = ss[n_interpol-1,:]
            
            # Optically thin
            x = -b_tile
            y = jj*np.exp(x_arr)
            aaa = np.exp(np.log(x)+np.log(y))
            bbb = np.diff(np.log(y),axis=0)+np.diff(np.log(x),axis=0)
            sum_elem = np.diff(np.log(x),axis=0)*aaa[0:-1,:]
            sum_elem = sum_elem * (np.exp(bbb)-1)/bbb
            w=np.where(np.isfinite(sum_elem) == 0)
            sum_elem[w]=0
            #coint[i] = init[i]*np.exp(-tau2[0]) + np.sum(sum_elem)
            coint_thin = init*np.exp(-tau2[0,:]) + np.sum(sum_elem,axis=0)
            
            # Radiative transfer:
            x = dtau2
            y = np.exp(x_arr)*ss
            aaa = np.exp(np.log(x)+np.log(y))
            bbb=np.diff(np.log(y),axis=0)+np.diff(np.log(x),axis=0)
            sum_elem = np.diff(np.log(x),axis=0)*aaa[0:-1,:]*(np.exp(bbb)-1)/bbb
            w=np.where(np.isfinite(sum_elem) == 0)
            sum_elem[w]=0
            coint_rad = init*np.exp(-tau2[0,:]) + np.sum(sum_elem,axis=0)
            
            #coint[i] = init[i]*np.exp(-tau2[0]) + np.sum(sum_elem)
            
#                    coint[i] = init[i]*np.exp(-tau2[0]) + np.sum(np.diff(np.log(x))*aaa[0:-1]*(np.exp(bbb)-1)/bbb)
#                    c2 = init[i]*np.exp(-tau2[0]) + np.trapz(jj*np.exp(x_arr), x=-b)
                    # Use log-space integration
            x = -b_tile
            y = jj*np.exp(x_arr)
            aaa2 = np.exp(np.log(x)+np.log(y)) 
            bbb2 = np.diff(np.log(y),axis=0)+np.diff(np.log(x),axis=0)
            sum_elem = np.diff(np.log(x),axis=0)*aaa2[0:-1,:]*(np.exp(bbb2)-1)/bbb2
            w=np.where(np.isfinite(sum_elem) == 0)
            sum_elem[w]=0
            #c2 = init[i]*np.exp(-tau2[0]) + np.sum(sum_elem)
            c2_rad = init*np.exp(-tau2[0,:]) + np.sum(sum_elem,axis=0)
            
            
#                    c2 = init[i]*np.exp(-tau2[0]) + np.sum(np.diff(np.log(x))*aaa2[0:-1]*(np.exp(bbb2)-1)/bbb2)
#                    print('intermediate',coint[i],init[i],-tau2[0],np.trapz(np.exp(x_arr[37:41])*ss[37:41], x=dtau2[37:41]),x,y)
                  #!#  print('intermediate',coint[i],c2)
            
            coint = np.where(coint_rad > c2_rad, c2_rad, coint_rad)
        
            #if (coint[i] > c2):
            #    coint[i] = c2
                    
            coint = np.where(tau2[n_interpol-2-mul,:] >= 1, coint_thick, coint)
            
            coint = np.where(tau2[0,:] < 1.e-3, coint_thin, coint)

            coint = np.where(coint < 0, np.zeros(coint.shape), coint)
            
            

# Can I vectorize this loop - Brian, 7/9/2020
            #for i in range(num_freq):
            #    mul = 1
            #    n_interpol = (nx-1)*mul+1
            #    
            #    tau2 = np.zeros(n_interpol)
            #    dtau2 = np.zeros(n_interpol)
            #    tau4 = np.zeros(n_interpol)
            #    dtau4 = np.zeros(n_interpol)
            #    
            #    
            #    x_interp = np.linspace(0, len(pos)-1, num=n_interpol, endpoint=True)
            #    x_interp2 = np.linspace(0, len(pos)-1, num=len(pos), endpoint=True)

            #    b = pos
            #    a = alpha[:,i]
            #    jj = j[:,i]
            #    omgmin = omega_min
            #    omgmax = omega_max
                                
                
            #    temp = (0 == np.isfinite(jj)).nonzero()
            #    jj[temp] = 1e-90 #0
            #        
            #    temp = (0 == np.isfinite(b)).nonzero()
            #    b[temp] = 0
                    
                    
            #    ss = jj/a     # R&L eqn. 6.54
                
# Fixing intervals to account for stupid pythion goign to second index-1, not second index - Brian, 4/11/2017                

#                x = 1-b[0:n_interpol-1]
#                y = a[0:n_interpol-1]
#                y1 = y[0:len(y)-1]
#                y2 = y[1:]
#                x1 = x[0:len(x)-1]
#                x2 = x[1:]
            #    x = 1-b[0:n_interpol]
            #    y = a[0:n_interpol]
            #    y1 = y[0:len(y)-1]
            #    y2 = y[1:]
            #    x1 = x[0:len(x)-1]
            #    x2 = x[1:]
                
            #    local_intergrals = (x2-x1)*(y1-y2)/(np.log(y1) - np.log(y2))
            #    
            #    #adding finite value checking
            #    temp = (0 == np.isfinite(local_intergrals)).nonzero()
            #    local_intergrals[temp] = 0
            #        
            #    dtau3 = np.cumsum(local_intergrals)
            #    dtau2[1:n_interpol] = dtau3
            #    
            #    tau2[0:n_interpol-1] = (np.cumsum(local_intergrals[::-1]))[::-1]
                
            #    for k in range(nx-1):
            #        tau[i,k] = tau2[k*mul]
            #        
            #    x_arr = -tau2
                
                # calculate dtau
                
            #    ds = np.zeros(nx)
            #    ds[0] = 0
            #    for k in range(nx):
            #        ds[k] = pos[k-1]-pos[k]
            #        
            #    dtau[i,:] = alpha[:,i]*ds
                
                #added if statement to handle transition to fully optically thick regime
            #    if tau2[n_interpol-2-mul] >= 1:  #If optically thick, just use the source function, might need to be higher (1e2 instead of 1e0) Brian - 7/8/2016
            #        coint[i] = ss[n_interpol-1]
                    
            #    elif tau2[0] <= 1e-3:   # if optically thin then don't use any absorption
            #        # Use log-space integration
            #        x = -b
            #        y = jj*np.exp(x_arr)
            #        aaa = np.exp(np.log(x)+np.log(y))
            #        bbb = np.diff(np.log(y))+np.diff(np.log(x))
            #        sum_elem = np.diff(np.log(x))*aaa[0:-1]*(np.exp(bbb)-1)/bbb
            #        w=np.where(np.isfinite(sum_elem) == 0)
            #        sum_elem[w]=0
            #        coint[i] = init[i]*np.exp(-tau2[0]) + np.sum(sum_elem)
                    
            #    else:   #otherwise use radiative transfer
            #        # Use log-space integration
            #        x = dtau2
            #        y = np.exp(x_arr)*ss
            #        aaa = np.exp(np.log(x)+np.log(y))
            #        bbb=np.diff(np.log(y))+np.diff(np.log(x))
            #        sum_elem = np.diff(np.log(x))*aaa[0:-1]*(np.exp(bbb)-1)/bbb
            #        w=np.where(np.isfinite(sum_elem) == 0)
            #        sum_elem[w]=0
            #        coint[i] = init[i]*np.exp(-tau2[0]) + np.sum(sum_elem)
            #        # Use log-space integration
            #        x = -b
            #        y = jj*np.exp(x_arr)
            #        aaa2 = np.exp(np.log(x)+np.log(y))
            #        bbb2 = np.diff(np.log(y))+np.diff(np.log(x))
            #        sum_elem = np.diff(np.log(x))*aaa2[0:-1]*(np.exp(bbb2)-1)/bbb2
            #        w=np.where(np.isfinite(sum_elem) == 0)
            #        sum_elem[w]=0
            #        c2 = init[i]*np.exp(-tau2[0]) + np.sum(sum_elem)
            #        
            #        if (coint[i] > c2):
            #            coint[i] = c2
            #        
            #    if not(coint[i] >= 0):
            #        coint[i]=0

                    
            # Done with for loop here - Brian, 7/10/2020
                    
            obint = coint
            
            tau = np.swapaxes(tau, 0, 1)


#            for iji in range(50):
#                plt.plot(omegasobs,j[iji,:])
#            plt.xscale('log')
#            plt.yscale('log')
#            plt.show()            

#            print(obint)
#            print(omegasobs)
#            print(omegas)
#            print(boost,shockgamma,shockbeta,np.cos(thetaprime))
#            plt.plot(omegasobs,obint)
#            plt.plot(omegasobs,obint[99]*(omegasobs/omegasobs[99])**(-2.5/2))
#            plt.xscale('log')
#            plt.yscale('log')
#            plt.show()            



            if (side==0):            
                return omegasobs, obint, coint, tau[:-1,:], j[:-1,:]  # j is temporary - Brian, 6/26/2017
            if (side==1):            
                return omegasobs, obint, coint, tau[1:,:], j[1:,:]  # j is temporary - Brian, 6/26/2017


            return omegasobs, obint, coint, tau, j  # j is temporary - Brian, 6/26/2017
            
            
            
            
def afterglow_total_power(n_elem, t, wmax0, wmin0, nx, ny, glow, epsilon, eqparb, 
                          external_nodensity, k_index, pindex = 2.5):
            
            
            #afterglow_total_power either calls afterglow (which produces shock geometry -- does this in most implmentations but not this one)
            #or reads in pre-generated geometery paramters (from blandford_mckee_full in this implementation).
            # it then calls jitter to determine the emissivity and absorption, etc., and finally calls afterglow_lineofsight to produce the spectrum
            
            
            to_ev = 4.1356691e-15 / 2/ np.pi
            
            n = ny - 1
            nl = nx-1
            
            n_elem_single = 2001
            

 # switching nx and ny
            angle_powerarray = np.zeros((nx, ny, n_elem))
            angle_alpha = np.zeros((nx, ny, n_elem))  
            angle_single_electron = np.zeros((nx, ny, n_elem_single))
            angle_omegas = np.zeros((nx, ny, n_elem))
            
            angle_omegasobs = np.zeros((n_elem, ny))
            angle_int = np.zeros((n_elem, ny))
            angle_int_f = np.zeros((n_elem, ny))
            angle_int_b = np.zeros((n_elem, ny))
            
            glow_names = ['x', 'y', 'r', 'theta', 'theta_pime', 'n', 'e', 'gamma', 'Robs', 'xi', 'beta', 'Eiso']

            if 1 == 1:  #always do this, change to 0 to skip this block of code
                
                #loop over x and y (s and Z)
                for j in range(ny):     # changed from j to i, need to change back? - Brian, 4/6/2017
                    
                    for i in range(nl+1):   # changed from j to i, need to change back? - Brian, 4/6/2017
                        
# switching j and i - Brian, 4/7/2017

                        theta = glow[4,i,j]
                        nodensity = glow[5,i,j]
                        thermal = glow[6,i,j]
                        bsq = thermal * eqparb * 8 * np.pi
                        
                        gammaint = np.copy(nodensity/(external_nodensity/glow[2,i,j]**k_index)/4)  # wind or ISM
                        
                        boost = glow[7,i,j]*(1 + glow[10,i,j]*np.cos(glow[4,i,j]))
                        
                        wmin = np.copy(wmin0)/boost
                        wmax = np.copy(wmax0)/boost
                        
                        #call jitter to get the j and alpha values
                        #This is the emissivity and self-absorption factors for synchrotron radiation
                        
                        powerarray, alpha, omegas, single_electron, single_omegas, wjm = jitter(theta, n_elem, wmax, wmin, 
                                                                                                nodensity = np.copy(external_nodensity/glow[2,i,j]**k_index), 
                                                                                                delta = 0, bsq = bsq, thermal = thermal, gammaint = gammaint, 
                                                                                                pindex = pindex, epsilon = epsilon)  # for wind
                        
                        #store data in arrays
                        

# switching j and i - Brian, 4/7/2017

                        angle_powerarray[i,j,:] = powerarray[:]
                        angle_alpha[i,j,:] = alpha[:]
                        angle_single_electron[i,j,:] = single_electron[:]
                        angle_omegas[i,j,:] = omegas[:]
                        

                temp = (0 == np.isfinite(angle_powerarray)).nonzero()
#                if temp[0] != -1:
                angle_powerarray[temp[0],temp[1]] = 1e-80
            
                temp = (0 == np.isfinite(angle_alpha)).nonzero()
#                if temp[0] != -1:
                angle_alpha[temp[0],temp[1]] = 1e-50


                    
# switching j and i - Brian, 4/7/2017

                for j in range(ny):
                    temp = (glow[7,:,j] < 1).nonzero()

# switching j and i - Brian, 4/7/2017
                    
#                    if temp[0] != -1:
                    angle_powerarray[temp,j,:] = 1e-80
                    angle_alpha[temp,j,:] = 1e-50
                    glow[7,temp] = 1
                    
                    temp = (0 == np.isfinite(glow[4,:,j])).nonzero()
#                    if temp[0] != -1:
                    glow[4,temp,j] = 0

                    temp = (0 == np.isfinite(glow[10,:,j])).nonzero()
#                    if temp[0] != -1:
                    glow[10,temp,j] = 0

                #at this point all the j and alpha data is stored in arrays
                # now we integrate to get the emission reaching an observer
 
# chanig to nx,ny - Brian, 4/6/2017               
#                tau_arr = np.zeros((ny,nx,n_elem))
                tau_arr = np.zeros((nx,ny,n_elem))



                
                # Loop aver s
                
                for j in range(ny):

# switching j,: to :,j - Brian, 4/6/2017
                    theta = glow[4,:,j]
                    nodensity = glow[5,:,j]
                    thermal = glow[6,:,j]
                    bsq = thermal * eqparb * 8 * np.pi
                    
                    gammaint = nodensity/(external_nodensity/glow[2,:,j]**k_index)/4

                    bulkgamma = glow[7,:,j]
                    bulkbeta = glow[10,:,j]

                    ds = np.zeros(nx)
                    ds[0] = 0
                    for k in range(1, nl+1):
# switching j,k to k,j - Brian, 4/6/2017
                        ds[k] = glow[0,k,j] - glow[0,k-1,j]

                    pos = -glow[0,:,j]
#                    pos = glow[0,:,j]
                    n= glow[5,:,j]
                    #print(pos)
                    
                    if pos[1] > pos[0]:
                        print(pos)
                        print('stop pos')
                        

                        
# switching j to 2nd elments, not first
# making copies of arrays, rather than passing arrays - Brian, 4/10/2017

                    alpha = np.copy(angle_alpha[:,j,:])
                    jj = np.copy(angle_powerarray[:,j,:])

                    omegas = np.copy(angle_omegas[int(nx/2),j,:])   # adding int() here - Brian, 9/13/2018

                    #integrate along line of sight (one line of x (Z)) 3 situations
                    # This will give the light from just the back half of the outflow, the front half of the outflow; and the front and back combined (including absorption)
                    
                    # integrate the back half of shock
                    

#  For some reason, in python array[0:4] returns the first 4 elements of the array (index 0,1,2,3), not the first 5 (like idl).  Need to use 0:5 instead.  How stupid. - Brian, 4/6/2017


# changing j to last index in glow, and array length to be python comaptible (nx/2-1 -> nx/2, nx-1 -> nx

# Making copies of array rather than passing arrays. - Brian, 4/11/2017

                    nx_2 = int(nx/2)
                    
                    alpha1=np.copy(alpha[0:nx_2,:])     # adding int() here - Brian, 9/13/2018
                    jj1=np.copy(jj[0:nx_2,:])           # adding int() here - Brian, 9/13/2018
#                    omegas1=np.copy(omegas)
                    omegas1=np.copy(wmax0)
                    pos1=np.copy(pos[0:nx_2])  
                    theta1=np.copy(theta[0:nx_2])    
                    bulkbeta1=np.copy(bulkbeta[0:nx_2])
                    bulkgamma1=np.copy(bulkgamma[0:nx_2])
                    n1=np.copy(n[0:nx_2])
                    glow1=np.copy(glow[:,0:nx_2,j])

#                    omegasobs, obint, coint, tau = afterglow_lineofsight(alpha[0:nx/2,:], jj[0:nx/2,:], omegas, np.zeros(n_elem), pos[0:nx/2], 
#                                          theta[0:nx/2], bulkbeta[0:nx/2], bulkgamma[0:nx/2],  
#                                          n[0:nx/2],glow[:,0:nx/2,j], epsilon, eqparb, k_index, pindex)
                    omegasobs, obint, coint, tau, jboost1 = afterglow_lineofsight(alpha1, jj1, omegas1, np.zeros(n_elem), pos1, 
                                          theta1, bulkbeta1, bulkgamma1,  
                                          n1,glow1, epsilon, eqparb, k_index, pindex, side=1)
                    

                    # error handeling
                    temp = (0 == np.isfinite(obint)).nonzero()
                    #if temp[0] != -1:
                    obint[temp] = 0

                    # Integrate front half of shock with result of back half as initial I_nu
                    nx_2 = int(nx/2)
                    
                    alpha2=np.copy(alpha[nx_2:nx,:])
                    jj2=np.copy(jj[nx_2:nx,:])
#                    omegas2=np.copy(omegas)
                    omegas2=np.copy(wmax0)
                    pos2=np.copy(pos[nx_2:nx])
                    theta2=np.copy(theta[nx_2:nx])
                    bulkbeta2=np.copy(bulkbeta[nx_2:nx])
                    bulkgamma2=np.copy(bulkgamma[nx_2:nx])
                    n2=np.copy(n[nx_2:nx])
                    glow2=np.copy(glow[:,nx_2:nx,j])

#                    omegasobs2, int2, coint2, tau2 = afterglow_lineofsight(alpha[nx/2:nx,:], jj[nx/2:nx,:], omegas, obint, pos[nx/2:nx], theta[nx/2:nx], 
#                                          bulkbeta[nx/2:nx], bulkgamma[nx/2:nx], 
#                                          n[nx/2:nx],glow[:,nx/2:nx,j], epsilon, eqparb, k_index, pindex)
                    omegasobs2, int2, coint2, tau2, jboost2 = afterglow_lineofsight(alpha2, jj2, omegas2, obint, pos2, 
                                          theta2, bulkbeta2, bulkgamma2,  
                                          n2,glow2, epsilon, eqparb, k_index, pindex, side=0)
                    
                    # Integrate front half of shock only
                    nx_2 = int(nx/2)
            
                    alpha3=np.copy(alpha[nx_2:nx,:])
                    jj3=np.copy(jj[nx_2:nx,:])
#                    omegas3=np.copy(omegas)
                    omegas3=np.copy(wmax0)
                    pos3=np.copy(pos[nx_2:nx])
                    theta3=np.copy(theta[nx_2:nx])
                    bulkbeta3=np.copy(bulkbeta[nx_2:nx])
                    bulkgamma3=np.copy(bulkgamma[nx_2:nx])
                    n3=np.copy(n[nx_2:nx])
                    glow3=np.copy(glow[:,nx_2:nx,j])

#                    omegasobs3, int3, coint3, tau3 = afterglow_lineofsight(alpha[nx/2:nx,:], jj[nx/2:nx,:], omegas, np.zeros(n_elem), pos[nx/2:nx], theta[nx/2:nx], 
#                                          bulkbeta[nx/2:nx], bulkgamma[nx/2:nx], 
#                                          n[nx/2:nx],glow[:,nx/2:nx,j], epsilon, eqparb, k_index, pindex)
                    omegasobs3, int3, coint3, tau3, jboost3 = afterglow_lineofsight(alpha3, jj3, omegas3, np.zeros(n_elem), pos3, 
                                          theta3, bulkbeta3, bulkgamma3,  
                                          n3,glow3, epsilon, eqparb, k_index, pindex, side=0)
                    
                    # store the data in arrays
                    
                    angle_omegasobs[:,j] = np.copy(omegasobs[:])
                    angle_int[:,j] = np.copy(int2[:])
                    angle_int_f[:,j] = np.copy(int3[:])
                    angle_int_b[:,j] = np.copy(obint[:])

# changed j order and limits - Brian, 4/6/2017
                    
                    nx_2 = int(nx/2)

                    tau_arr[0:nx_2,j,:] = np.copy(tau)
                    tau_arr[nx_2:nx,j,:] = np.copy(tau2)
                    
                    angle_powerarray[0:nx_2,j,:] = jboost1
                    angle_powerarray[nx_2:nx,j,:] = jboost2
                    
                #error handling
                temp = (0 == np.isfinite(angle_int)).nonzero()
                if temp[0] != -1:
                    angle_int[temp] = 0


                temp = (0 == np.isfinite(angle_int_f)).nonzero()
                if temp[0] != -1:
                    angle_int_f[temp] = 0
                 
                temp = (0 == np.isfinite(angle_int_b)).nonzero()
                if temp[0] != -1:
                    angle_int_b[temp] = 0


                # create arrays to store data
                total_power = np.zeros(n_elem)
                total_power_f = np.zeros(n_elem)
                total_power_b = np.zeros(n_elem)
                
                # Integrate in the y(S) direction - Brian 7/8/2016
                # This is a simple intergra, we just add up the light along each line in x (Z)
                

# change index order to in glow 1,1,: from 1,:,1 - Brian, 4/6/2017

                for i in range(n_elem):
                    ai=np.copy(angle_int[i,:])
                    ai=np.append(ai[0],ai[:])
                    ai=np.append(ai[:],0.)
                    gl=np.copy(glow[1,1,:])
                    gl=np.append(0,gl[:])
#                    gl=np.append(gl[:],gl[len(gl)-1]/(1-.5**2/ny**2))  # for integer half values only - Brian, 6/29/2017
                    gl=np.append(gl[:],gl[len(gl)-1]/.99)   # for 0.99 is max y only - Brian, 6/29/2017
#                    total_power[i] = integrate.trapz(angle_int[i,:] * glow[1,1,:], x=glow[1,1,:])
#                    total_power[i] = integrate.trapz(ai * gl, x=gl)
                    total_power[i] = integrate.simps(ai * gl, x=gl, even='last')
                    
                for i in range(n_elem):
                    total_power_f[i] = integrate.simps(angle_int_f[i,:] * glow[1,1,:], x=glow[1,1,:]) 
                    
                for i in range(n_elem):
                    total_power_b[i] = integrate.simps(angle_int_b[i,:] * glow[1,1,:], x=glow[1,1,:])



                    
                
                return total_power, angle_omegasobs, angle_int, angle_powerarray, angle_alpha, angle_int_f, angle_int_b, total_power_f, total_power_b, tau_arr
                    
                    
def afterglow_aniso_bmk(savefile, s_points = 10, z_points = 10, phi_slices = 10, outputfile = 'afterglow_sych_spectrum.p', 
                        t_arr = np.array([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]), wmax = 1e6, wmin = 1e-11, 
                        n_elem = 451, epsilon = 1e-2, eqparb = 1e-5, external_nodensity = 'default', k_index = False, 
                        m_index = False, eiso_floor = 0, pindex=2.5, rot_angle=0.0, shock_thickness=0.1):
    
                # This is the procedure called by run_code_test
                # It will do a loop over time and in each loop calculate the hydro parameters
                # call (blandford_mckee_full) and calculate the resulting radiaiton (call afterglow_total_power)
                
                to_ev = 4.1356691e-15 /2/np.pi  #convert omega(frequency) to eV
                
                wmax0 = wmax/to_ev  #maximum value of range of omega in eV
                wmin0 = wmin/to_ev  #minimum value of range of omega in eV
                
                freq = wmax0
                
                # Translating the cylindrical coordinates form the BMK stuff into x, y 
                # coordinate we have here, then solving for each slice in phi.
                # Phi-slicing is not fully implmented; need to loop over each slice if
                # we want to do anything viewed off axis and therefore with an asymmetry
                
                nx = z_points
                ny = s_points
                
                nt = len(t_arr)
                
                if external_nodensity == 'default':
                    v_wind = 1000
                    m_dot_wind = 1e-6
                    external_nodensity = (m_dot_wind*1.98892*1e33/(365.25*24*3600))/(v_wind*1e5)/4/np.pi/gv.mp  #for wind situation

                #set up output arrays

# switching ny,nx to nx,ny - Brian, 4/6/2017
                
                total_power_arr = np.zeros((nt, n_elem, phi_slices))
                total_power_f_arr = np.zeros((nt, n_elem, phi_slices))
                total_power_b_arr = np.zeros((nt, n_elem, phi_slices))
               # glow_arr = np.zeros((12, nx, ny, nt, phi_slices))
                glow_arr = np.zeros((12, 1, ny, nt, phi_slices))   # Keep compatible for making images
                angle_int_arr = np.zeros((n_elem, ny, nt, phi_slices))
                #angle_int_f_arr = np.zeros((n_elem, ny, nt, phi_slices))
                #angle_int_b_arr = np.zeros((n_elem, ny, nt, phi_slices))
                freq_arr = np.zeros((nt, n_elem, phi_slices))

                #angle_powerarray_arr = np.zeros((nx, ny, n_elem, nt, phi_slices))
                #angle_alpha_arr = np.zeros((nx, ny, n_elem, nt, phi_slices))

                # Set up smaller _j arrays so we can unload the big arrays for less ram usage - Brian, 6/9/2020
                
                total_power_arr_j = np.zeros((n_elem, phi_slices))
                total_power_f_arr_j = np.zeros((n_elem, phi_slices))
                total_power_b_arr_j = np.zeros((n_elem, phi_slices))
                glow_arr_j = np.zeros((12, nx, ny, phi_slices))
                angle_int_arr_j = np.zeros((n_elem, ny, phi_slices))
                angle_int_f_arr_j = np.zeros((n_elem, ny, phi_slices))
                angle_int_b_arr_j = np.zeros((n_elem, ny, phi_slices))
                freq_arr_j = np.zeros((n_elem, phi_slices))

                #angle_powerarray_arr_j = np.zeros((nx, ny, n_elem, phi_slices))
                #angle_alpha_arr_j = np.zeros((nx, ny, n_elem, phi_slices))

                
                # Create an list with all the input parameters.  Will now be saved in the file too. - Brian, 6/9/2020
                
                input_params = [s_points, z_points, phi_slices, outputfile, 
                                t_arr, wmax, wmin, 
                                n_elem, epsilon, eqparb, external_nodensity, k_index, 
                                m_index, eiso_floor, pindex, rot_angle, shock_thickness]
                
                input_params_names = ['s_points', 'z_points', 'phi_slices', 'outputfile', 
                                      't_arr', 'wmax', 'wmin', 
                                      'n_elem', 'epsilon', 'eqparb', 'external_nodensity', 'k_index', 
                                      'm_index', 'eiso_floor', 'pindex', 'rot_angle', 'shock_thickness']
                
                
                
                # Write the empty big arrays to a file - Brian, 6/9/2020
                
                angle_omegasobs = np.zeros((2))   # dummy becuase this hasn't been created yet.
                
                afterglow_data_array = {'total_power_arr': total_power_arr, 'glow_arr': glow_arr, 'angle_int_arr': angle_int_arr, 
                                        'angle_omegasobs': angle_omegasobs, 't_arr': t_arr, 
                                        'total_power_f_arr': total_power_f_arr, 
                                        'total_power_b_arr': total_power_b_arr, 
                                        #'angle_int_f_arr': angle_int_f_arr, 'angle_int_b_arr': angle_int_b_arr,
                                        'freq_arr': freq_arr, 
                                        'input_params': input_params, 'input_params_names': input_params_names} #,
                                        #'angle_powerarray_arr': angle_powerarray_arr, 'angle_alpha_arr': angle_alpha_arr}

                output = open(outputfile, 'wb')
                pickle.dump(afterglow_data_array, output)
                output.close()
                
                
                # remove big arrays - Brian, 6/9/2020
                
                total_power_arr = 0.
                total_power_f_arr = 0.
                total_power_b_arr = 0.
                glow_arr = 0.
                angle_int_arr = 0.
                angle_int_f_arr = 0.
                angle_int_b_arr = 0.
                freq_arr = 0.
                    
                angle_powerarray_arr = 0.
                angle_alpha_arr = 0.
                
                afterglow_data_array = 0.
                    
                
                # Loop over different times.  Calculations at each time are independent of each other
                
                for i in range(nt):
                    
                    #print(i)
                    time0 = t_arr[i]
                    

                    # Generate BMK data on shock geometry, density, etc and put
                    # it in the glow_net array.
                    #This is where all the hydrodynamic variables are created
                    
                    
                    
                    glow_net = blandford_mckee_full_brian(savefile, time0, s_points, z_points, phi_slices, 
                                                          k_index, m_index, external_nodensity, shock_thickness)
                    
                    output = open('glow_test_full_res_1e1.p', 'wb')
                    pickle.dump(glow_net, output)
                    output.close()
#                    
#                    #  Can also load glow array from file to cut computation time down
#                    
#                    data_in = open('glow_test_full_res_1e1.p', 'rb')
#                    glow_net = pickle.load(data_in)
#                    data_in.close()
                    
                    #loop over phi starting here.  Phi slices are independent of each other
                    # once glow_net has been calculated
                    
                    for j in range(phi_slices):
                        glow = glow_net[:,:,:,j]  # Selects single phi slice from the generated array
                        
                        #  where xi is less when one or eiso is at the energy floor and should be ignored
                        w = (glow[9,:,:] < 1).nonzero()
                        
                        #if w[0] != -1:
                            
                        glow[5,:,:][w[0],w[1]] = 1e-30        # set density to zero
                            
                        glow[6,:,:][w[0],w[1]] = 1e-30        # set energy to zero
                            
                        glow[7,:,:][w[0],w[1]] = np.nan         # set gamma to NaN
                            
                            
                        w = (glow[11,:,:] < eiso_floor).nonzero()
                        
#                        if w[0] != -1:
                            
                        glow[5,:,:][w[0],w[1]] = 1e-30        # set density to zero
                            
                        glow[6,:,:][w[0],w[1]]= 1e-30        # set energy to zero
                            
                        glow[7,:,:][w[0],w[1]] = np.nan          # set gamma to NaN
                            

                        # Now calculate the emission  THis is the call to the radiative transfer part of the code
                        
                        total_power, angle_omegasobs, angle_int, angle_powerarray, angle_alpha, angle_int_f, angle_int_b, total_power_f, total_power_b, tau_arr = afterglow_total_power(n_elem, t_arr[i], wmax0, wmin0, nx, ny, glow, epsilon, eqparb, external_nodensity, 
                                              k_index, pindex)
                        
                        # store data in arrays
                        
                        # Store just hte data for this time in _j arrays - Brian, 6/9/2020


                        total_power_arr_j[:,j] = total_power
                        total_power_f_arr_j[:,j] = total_power_f
                        total_power_b_arr_j[:,j] = total_power_b
                        glow_arr_j[:,:,:,j] = glow[:,:,:]
                        angle_int_arr_j[:,:,j] = angle_int[:,:]
                        angle_int_f_arr_j[:,:,j] = angle_int_f[:,:]
                        angle_int_b_arr_j[:,:,j] = angle_int_b[:,:]
                        freq_arr_j[:,j] = freq

                        #angle_powerarray_arr_j[:,:,:,j] = angle_powerarray[:,:,:]
                        #angle_alpha_arr_j[:,:,:,j] = angle_alpha[:,:,:]

                    # load and save data in big arrays, then remove those arrays from memory - Brian, 6/9/2020
                    
                    # Load data
                    # Do not load angle_powerarray_arr and angle_alpha_arr for now.  Not saving them anyway.
                    
                    data_in = open(outputfile, 'rb')
                    data = pickle.load(data_in)
                    data_in.close()
                    
                    total_power_arr=data['total_power_arr']
                    total_power_f_arr=data['total_power_f_arr']
                    total_power_b_arr=data['total_power_f_arr']
                    glow_arr=data['glow_arr']
                    angle_int_arr=data['angle_int_arr']
                    #angle_int_f_arr=data['angle_int_f_arr']
                    #angle_int_b_arr=data['angle_int_b_arr']
                    freq_arr=data['freq_arr']
                    
                    #angle_powerarray_arr=data['angle_powerarray_arr']
                    #angle_alpha_arr=data['angle_alpha_arr']
                    
                    # Add new data to big arrays for this time - Brian, 6/9/2020
                    
                    total_power_arr[i,:,:] = total_power_arr_j[:,:]
                    total_power_f_arr[i,:,:] = total_power_f_arr_j[:,:]
                    total_power_b_arr[i,:,:] = total_power_b_arr_j[:,:]
                   # glow_arr[:,:,:,i,:] = glow_arr_j[:,:,:,:]
                    glow_arr[:,0,:,i,:] = glow_arr_j[:,-1,:,:]   # Only use the front od the shock, to keep compatible with image file
                    angle_int_arr[:,:,i,:] = angle_int_arr_j[:,:,:]
                    #angle_int_f_arr[:,:,i,:] = angle_int_f_arr_j[:,:,:]
                    #angle_int_b_arr[:,:,i,:] = angle_int_b_arr_j[:,:,:]
                    freq_arr[i,:,:] = freq_arr_j[:,:]

                    #angle_powerarray_arr[:,:,:,i,:] = angle_powerarray_arr_j[:,:,:,:]
                    #angle_alpha_arr[:,:,:,i,:] = angle_alpha_arr_j[:,:,:,:]
                    
                    # Save data in file - Brian, 6/9/2020
                    
                    afterglow_data_array = {'total_power_arr': total_power_arr, 'glow_arr': glow_arr, 
                                            'angle_int_arr': angle_int_arr, 
                                            'angle_omegasobs': angle_omegasobs, 't_arr': t_arr, 
                                            'total_power_f_arr': total_power_f_arr, 
                                            'total_power_b_arr': total_power_b_arr, 
                                            #'angle_int_f_arr': angle_int_f_arr, 'angle_int_b_arr': angle_int_b_arr,
                                            'freq_arr': freq_arr,
                                            'input_params': input_params, 'input_params_names': input_params_names} #,

                                        #'angle_powerarray_arr': angle_powerarray_arr, 'angle_alpha_arr': angle_alpha_arr}

                    output = open(outputfile, 'wb')
                    pickle.dump(afterglow_data_array, output)
                    output.close()
                    
                    # remove big arrays - Brian, 6/9/2020
                
                    total_power_arr = 0.
                    total_power_f_arr = 0.
                    total_power_b_arr = 0.
                    glow_arr = 0.
                    angle_int_arr = 0.
                    angle_int_f_arr = 0.
                    angle_int_b_arr = 0.
                    freq_arr = 0.
                    
                    angle_powerarray_arr = 0.
                    angle_alpha_arr = 0.
                    
                    afterglow_data_array = 0.


                        
                    #  Save all data in a big file.  Includes hydro and radiative data - Brian 7/8/2016
                    
                #afterglow_data_array = {'total_power_arr': total_power_arr, 'glow_arr': glow_arr, 'angle_int_arr': angle_int_arr, 
                 #                           'angle_omegasobs': angle_omegasobs, 't_arr': t_arr, 'total_power_f_arr': total_power_f_arr, 
                 #                           'total_power_b_arr': total_power_b_arr, 'angle_int_f_arr': angle_int_f_arr, 'angle_int_b_arr': angle_int_b_arr,
                 #                           'freq_arr': freq_arr, 'angle_powerarray_arr': angle_powerarray_arr, 'angle_alpha_arr': angle_alpha_arr}
                  
                
#                afterglow_data_array = {'total_power_arr': total_power_arr, 'glow_arr': glow_arr, 'angle_int_arr': angle_int_arr, 
#                                            'angle_omegasobs': angle_omegasobs, 't_arr': t_arr, 'total_power_f_arr': total_power_f_arr, 
#                                            'total_power_b_arr': total_power_b_arr, 'angle_int_f_arr': angle_int_f_arr, 'angle_int_b_arr': angle_int_b_arr,
#                                            'freq_arr': freq_arr} #,
                                        #'angle_powerarray_arr': angle_powerarray_arr, 'angle_alpha_arr': angle_alpha_arr}
                
#                output = open(outputfile, 'wb')
#                pickle.dump(afterglow_data_array, output)
#                output.close()
                    
                    
                        
                        
                        
    
