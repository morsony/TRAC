import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import io
from global_var import sysvars as gv
import pickle
import functools
from matplotlib import pyplot as plt

def jitter(thetain, n_elemf, wmax, wmin, nodensity = 1, thermal = 1e-4, eef = 0.1, gammaint = 3, gammae = 2.5, alpha1 = 2, 
           alpha2 = 2, beta1 = 10, beta2 = 10, pindex = 2.5, delta = 0.1, v = gv.cl, bsq = 1, epsilon = 0.01, kappadelta1 = 1, 
           kappadelta2 = 1):
    
    #call jittermid, the wrapper that then calls the main function
    
    powerarray, alpha, omegas, single_electron, single_omegas, wjm1 = jittermid(thetain, n_elemf, wmax, wmin, nodensity, 
                                                                                thermal, eef, gammaint, gammae, 
                                                                                alpha1, alpha2, beta1, beta2, pindex, 
                                                                                delta, v, bsq, epsilon, kappadelta1, 
                                                                                kappadelta2)
              
    
    
    return powerarray, alpha, omegas, single_electron, single_omegas, wjm1
 
    
# jittermid is a wrapper funcitono that loads the inputs given by jitter into shared variables and calls power    
    
def jittermid(thetain, n_elemf, wmax, wmin, nodensity1 = 1, thermal1 = 1e-4, eef1 = 0.1, gammaint1 = 3, gammae1 = 2.5, alpha11 = 2, 
              alpha21 = 2, beta11 = 10, beta21 = 10, pindex1 = 2.5, delta1 = 0.1, v1 = gv.cl, bsq1 = 1, epsilon1 = 0.01, kappadelta11 = 1, 
              kappadelta21 = 1):
    
    # this is where the formulation to be used is determined
    #currently the switch happens at theta = np.pi/2 * 0.6 which seems to work well
    # But any values between np.pi/2*0.4 and np.pi/2*0.7 should be okay
    
    gv.theta = thetain
    
    if gv.theta >= np.pi/2*0.6:
        gv.form = 1
    else:
        gv.form = 0
        
    # Intergral Accuracy
    criterion = 1e-4
    
    gv.nodensity = nodensity1
    gv.thermal = thermal1
    gv.eef = eef1
    gv.gammaint = gammaint1
    gv.gammae = gammae1
    gv.alpha1 = alpha11
    gv.alpha2 = alpha21
    gv.beta1 = beta11
    gv.beta2 = beta21
    gv.kappadelta1 = kappadelta11
    gv.kappadelta2 = kappadelta21
    gv.pindex = pindex1
    gv.delta = delta1
    gv.v = v1
    gv.bsq = bsq1
    gv.epsilon = epsilon1
    
    
#    single_electron, single_omegas, wc = synch_single(n_elemf)

# Use to calculate with arbitrary pindex (must be >2.0
#    powerarray, alpha, omegas, = jitter_power(single_electron, single_omegas, n_elemf, wmax, wmin)
#    powerarray, alpha, omegas, single_electron, single_omegas, jitter_power_array, = jitter_power(n_elemf, wmax, wmin)

# To use stored, pre-integrated powerarray, at fixed value of pindex - Brian, 4/11/2017
#    powerarray, alpha, omegas, = jitter_power_stored(single_electron, single_omegas, n_elemf, wmax, wmin)
#    powerarray, alpha, omegas, single_electron, single_omegas= jitter_power_stored(n_elemf, wmax, wmin)

    powerarray, alpha, omegas, single_electron, single_omegas= jitter_power_memoized(n_elemf, wmax, wmin)

    wjm1 = gv.wjm
    
    return powerarray, alpha, omegas, single_electron, single_omegas, wjm1
    
    

# new function that can be memoized - Brian, 5/9/2017
@functools.lru_cache()
def jitter_power_pindex(pindex):
    gv.pindex = pindex
    n_elemf=3001 #2001
    wmax0=5.8061389777821121e29  #e18
    wmin0=5.8061389777821121e-4
    omegas = 10**(np.arange(n_elemf)/(n_elemf-1)*np.log10(wmax0/wmin0))    # changed form wmax to wmax0 - Brian, 4/10/2917  # removed - Brian, 4/6/2017
    omegas = omegas*wmin0                                                  # changed form wmin to wmin0 - Brian, 4/10/2917  # removed - Brian, 4/6/2017

    powerarray, alpha, omegas, single_electron, single_omegas, jitter_power_array, = jitter_power(n_elemf,omegas,wmin0)
    return jitter_power_array,



    
# jitter_power takes the single electron spectrum and integrates over the electron distribution
# Chnging ot call synch_single internally - Brian, 5/9/2017

#def jitter_power(single_electron, single_omegas, n_elemf, 
#                 wmax, wmin):

def jitter_power(n_elemf, 
                 wmax, wmin):

    # Call this first, no longer needed as input    
    single_electron, single_omegas, wc = synch_single(n_elemf)

    gv.intswitch = 1
    
    #intergral accuracy
    criterion = 1e-4
    
    
    # calculate the minimum gamma nd constant for the electron disribution
    
    #values = elecdist(gv.epsilon, gv.thermal)   # get the minimum gamma and distribution constant
    values = elecdist(gv.epsilon, gv.thermal,gv.gammaint,gv.nodensity)   # get the minimum gamma and distribution constant
    gv.mingamma = values['gammamin']                #lower limit for integration
    gv.k = values['constant']                      # constant in electron distribution
    

    # Here we are creating the values of gamma to sum over
    # The parameters below seem to work well
    ngammas = 1001
    
    gammas = (np.arange(ngammas)/(ngammas-1))**2 * 19 +1
    gammas0 = gammas
    
    
    # This sets up the values of omegas to be included in the final (integrated over gamma) spectrum
    # Right now, this is logarithmicall spaced from wmin to wmax
    # The accuracy of the tail end of this distubution (low omega) is dependent on the range of omegas in 
    # the single elctron spectrum
    
    powerarray3 = np.zeros(n_elemf)
    powerarray4 = np.zeros(n_elemf)
 
# Need this to get a good range in omegas - Brian, 4/10/2017   
    wmax0=5.8061389777821121e15
    wmin0=5.8061389777821121e-4
#    omegas = 10**(np.arange(n_elemf)/(n_elemf-1)*np.log10(wmax0/wmin0))    # changed form wmax to wmax0 - Brian, 4/10/2917  # removed - Brian, 4/6/2017
#    omegas = omegas*wmin0                                                  # changed form wmin to wmin0 - Brian, 4/10/2917  # removed - Brian, 4/6/2017

    omegas=wmax       # this is an input of arbitrary values of omega - Brian, 4/6/2017
    
    power1_array = np.zeros((ngammas, n_elemf))
    omegas1_array = np.zeros((ngammas, n_elemf))
    
    
    # Integrate over gamma by shifting the single electron spectrum
    
    for i in range(n_elemf):
        
        if omegas[i] > 2*gv.wjm:
            gammas = gammas0*np.sqrt(omegas[i]/(2*gv.wjm))
        else:
            gammas = gammas0
            
        omegas1 = omegas[i]/gammas**2

        #print('omegas[i]=',omegas[i],' 2*wjm=',2*gv.wjm,' omegas1[0]=',omegas1[0])

        # Interpolate in logspace
        log_power_interf = interpolate.interp1d(np.log10(single_omegas), np.log10(single_electron), bounds_error = False, fill_value = 'extrapolate')
        power1 = 10**(log_power_interf(np.log10(omegas1))) * (gv.mingamma*gammas)**(-gv.pindex)
        
        index = (omegas1 > 2*gv.wjm).nonzero()
        if index[0] != -1:
            power1[index] = 0
        
        index = (0 == np.isfinite(power1))
        if index[0] != -1:
            power1[index] = 0


        powerarray3[i] = integrate.simps(power1, gv.mingamma*gammas)
        
        power1_array[:,i] = power1
        omegas1_array[:,i] = omegas1
        

#        print(omegas[i],gv.wjm,gv.mingamma)
#        print(gammas)
#        plt.plot(omegas1,power1)
#        plt.yscale('log')
#        plt.xscale('log')
#        plt.show()
        
        # calculate alpha for the same values of omegas
        
        log_power_interf = interpolate.interp1d(np.log10(single_omegas), np.log10(single_electron), bounds_error = False, fill_value = 'extrapolate')
        power1 = 10**(log_power_interf(np.log10(omegas1))) * (gv.mingamma*gammas)**(-gv.pindex-1)
        
        index = (omegas1 > 2*gv.wjm).nonzero()
        if index[0] != -1:
            power1[index] = 0
            
#        powerarray4[i] = integrate.simps(power1, gv.mingamma*gammas)*omegas[i]**(-2)*(gv.pindex+2)*np.pi**2/gv.me *4/8/np.pi #*2  # should be omegas or omegas1? - Brian, 4/10/2017  # addind *2 to match R&L 6.52 I think? - Brian, 6/21/2017
        powerarray4[i] = integrate.simps(power1, gv.mingamma*gammas)*(omegas[i]/2/np.pi)**(-2)*(gv.pindex+2) / (8.*np.pi*gv.me) # should be omegas or omegas1? - Brian, 4/10/2017  # now compatable with R&L 6.52 and Granot, Piran & Sari 1998 eqn. 15 - Brian, 6/22/2017
        

# Now interpolate to find the right values for out frequencies (stored in wmax) - Brian, 4/10/2017

#        print('power1,powerarray3,omegas1',len(power1),len(powerarray3),len(omegas1))
#
#        log_powerarray3_interf = interpolate.interp1d(np.log10(omegas1), np.log10(powerarray3), bounds_error = False, fill_value = 'extrapolate')
#        powerarray3_wmax = 10**(log_powerarray3_interf(wmax))
#
#        log_powerarray4_interf = interpolate.interp1d(np.log10(omegas1), np.log10(powerarray4), bounds_error = False, fill_value = 'extrapolate')
#        powerarray4_wmax = 10**(log_powerarray4_interf(wmax))

        
    #power3 now contains emission, power4 alpha, and omegas the corresponding values of omega for k=1
    #multipy by k to get the correct total
    #print(gv.k)
    powerarray = gv.k*powerarray3 # _wmax   # added _wmax - Brian, 4/10/2017
    alpha = gv.k*powerarray4 # _wmax        # added _wmax - Brian, 4/10/2017
    
    prefactor = 0

    bfield = np.sqrt(gv.bsq)
    SA = np.sin(np.pi/2)
    wc = 3*gv.mingamma**2*gv.q*bfield*SA/(2*gv.me*gv.cl)   # added * SA , R&L 6.17c - Brian, 6/22/2017
    prefactor = np.sqrt(3)*gv.q**3*bfield*SA/(2*np.pi*gv.me*gv.cl**2)
    gv.wjm = wc*10
    



    print('done jitter_power')
    
    single_electron0 = np.copy(single_electron)
    single_omegas0 = np.copy(single_omegas)
    omegas0 = np.copy(omegas)
    powerarray0 = np.copy(powerarray)
    alpha0 = np.copy(alpha)
    n_elemf0 = np.copy(n_elemf)
    wmax0 = np.copy(wmax)
    wmin0 = np.copy(wmin)
    pindex0 = np.copy(gv.pindex)
    wjm0 = np.copy(gv.wjm)
    mingamma0 = np.copy(gv.mingamma)
    k0 = np.copy(gv.k)
    epsilon0 = np.copy(gv.epsilon)
    prefactor0 = np.copy(prefactor)
    bsq0 = np.copy(gv.bsq)
    
    jitter_power_array = {'single_electron0': single_electron0, 'single_omegas0': single_omegas0, 
                          'omegas0': omegas0, 'powerarray0': powerarray0, 'alpha0': alpha0,
                          'n_elemf0': n_elemf0, 'wmax0': wmax0, 'wmin0': wmin0, 'pindex0': pindex0,
                          'wjm0': wjm0, 'mingamma': mingamma0, 'k0': k0, 'epsilon0': epsilon0, 
                          'prefactor0': prefactor0, 'bsq0': bsq0}
    
    
    #output = open('powerarray_' + str(gv.pindex) + '_synch_' + str(gv.theta*180/np.pi) + '.p', 'wb')
    #pickle.dump(jitter_power_array, output)
    #output.close()

    # consistency check, for p=2.5
#    ss = (2*np.pi)**(gv.pindex+2) / (np.pi*gv.cl**2*(gv.pindex+1)) * (3*gv.q)**(-1/2) * gv.me**((2*gv.pindex+1)/2) * gv.cl**((4*gv.pindex+1)/2) * (bfield*SA)**(-1/2) * 1.1974 * omegas**(5/2)
    ss = (2*np.pi)**((gv.pindex+4)/2) / (np.pi*gv.cl**2*(gv.pindex+1)) * (bfield*SA) * (3*gv.q/(2*np.pi*gv.me**3*gv.cl**5))**(-gv.pindex/2) * (bfield*SA)**(-(gv.pindex+2)/2) * (gv.me*gv.cl/(3*gv.q*bfield*SA))**(-(gv.pindex-1)/2) * 1.1974 * omegas**(5/2)
    j0 = np.sqrt(3)*gv.q**3*bfield*SA/(2*np.pi*gv.me*gv.cl**2) / (gv.pindex+1) * k0 * 1.81511 * (omegas0*(gv.me*gv.cl/(3*gv.q*bfield*SA)))**(-(gv.pindex-1)/2)
    a0 = np.sqrt(3)*gv.q**3/(8*np.pi*gv.me)*(3*gv.q/(2*np.pi*gv.me**3*gv.cl**5))**(gv.pindex/2) * k0 * (bfield*SA)**((gv.pindex+2)/2) * 1.51588 * (omegas0/(2*np.pi))**(-(gv.pindex+4)/2)
#    a0 = np.sqrt(3)*gv.q**3/(8*np.pi) * (3*gv.q/(2*np.pi))**(gv.pindex/2) * (gv.me)**(-(gv.pindex/2+2)) * (gv.cl)**(-(gv.pindex/2+3)) * k0 * (bfield*SA)**((gv.pindex+2)/2) * 1.51588 * (omegas0/(2*np.pi))**(-(gv.pindex+4)/2)
    vl = gv.q*bfield/(2*np.pi*gv.me*gv.cl)
    a1 = gv.q**(2) / (4*gv.me*gv.cl**2) * (3)**((gv.pindex+1)/2) * k0 * (vl*SA)**((gv.pindex+2)/2) * 1.51588 * (omegas0/(2*np.pi))**(-(gv.pindex+4)/2)  # from Chapter 4 of ??? online - Brian, 6/22/2017  
    # both a0 and a1 seem to be wrong.
#    print(ss)
#    print(powerarray/alpha)
#    print(powerarray/j0)
#    print(alpha/a0)
#
#    print(wjm0)
#    print(gammas0)
#
#    plt.plot(omegas0,powerarray0)
#    plt.plot(omegas0,j0)
#    plt.plot(omegas0,alpha0)
#    plt.plot(omegas0,a0)
#    plt.plot(omegas0,a1)
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.show()
    
    return powerarray, alpha, omegas, single_electron, single_omegas, jitter_power_array
                    
                    
                    
                    
def synch_single(n_elemf):
    
    gv.intswitch = 1
    
    gv.omega = 0
    gv.gamma = 0
    gv.k = 0  #gv.kb = 0
    
    gv.tempvar1 = 0
    gv.tempvar2 = 0
    
    #intergral accuracy
    criterion = 1e-4
    
    # calculate the minimum gamma nd constant for the electron disribution
    
    #values = elecdist(gv.epsilon, gv.thermal)   # get the minimum gamma and distribution constant
    values = elecdist(gv.epsilon, gv.thermal,gv.gammaint,gv.nodensity)   # get the minimum gamma and distribution constant
    gv.mingamma = values['gammamin']                #lower limit for integration
    gv.k = values['constant']                      # constant in electron distribution
    
    bfield = np.sqrt(gv.bsq)
    SA = np.sin(np.pi/2)
    
    #Get synch frequency
    wc = 3*gv.mingamma**2*gv.q*bfield*SA/(2*gv.me*gv.cl)   # added  * SA , R&L 6.17c - Brian, 6/22/2017
    
    tot = 2.63064761e-16

    nelem = n_elemf
    
    prefactor = np.sqrt(3)*gv.q**3*bfield*SA/(2*np.pi*gv.me*gv.cl**2)
    
    gv.wjm = wc*10
    

    if gv.theta < np.pi/2:
        theta2 = gv.theta
    else:
        theta2 = np.pi/2

    restore_synch = io.idl.readsav('single_electron_synch_00.00.idl', python_dict = True)
    
    powerarray0 = restore_synch['powerarray0']
    mingamma0 = restore_synch['mingamma0']
    wjm0 = restore_synch['wjm0']
    prefactor0 = restore_synch['prefactor0']
    omegas0 = restore_synch['omegas0']
    
    single_omegas = omegas0/wjm0*gv.wjm
    single_electron = powerarray0/prefactor0*prefactor
#    single_electron = powerarray0/mingamma0*gv.mingamma

#    print(wjm0,gv.wjm)
#    print(prefactor0,prefactor)
#    print(max(powerarray0),max(single_electron))
#    plt.plot(omegas0,powerarray0)
#    plt.plot(single_omegas,single_electron)
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.show()



    bfield0 = prefactor0/np.sqrt(3)/gv.q**3/SA*(2*np.pi*gv.me*gv.cl**2)
    bsq0 = bfield0**2
    #single_electron = powerarray0*(gv.mingamma/mingamma0)**2*(gv.bsq/bsq0)*(gv.wjm/wjm0)**(-1)

    return single_electron, single_omegas, wc


# pass all values as arguements - Brian, 5/18/2017
#def elecdist(eps, ethermal):
def elecdist(eps, ethermal, gammaint, nodensity):
    
    # outputs -- values (structure with gammain, gammax, k, gammadist)
    
    # inputs -- nodensity (the local co-moving number density)
    #           eps (the fraction of the internal energy residing in the electrons)
    #           ethermal (the local co-moving internal energy)
    
#    nodensity_e = 4*gv.gammaint*gv.nodensity
    nodensity_e = 4*gammaint*nodensity #* .1
    
# Use the first one here - Brian, 7/6/2020
#    gammamin = ((gv.pindex-2)/(gv.pindex-1))*eps*ethermal/(nodensity_e*gv.me*gv.cl**2) + 1.# added 1, not completely sure this is right - Brian, 4/6/2017 - Brian, 9/17/2015  #minimum gamma
#    gammamin = ((gv.pindex-2)/(gv.pindex-1))*eps*ethermal/(nodensity_e*gv.me*gv.cl**2)/1.5 + 1.  # added 1, not completely sure this is right - Brian, 4/6/2017 - Brian, 9/17/2015  #minimum gamma


    # Adding correction for low gammamin.  This give the right energy in electrons, but a small number of electrons at gammamin ~ 1. - Brian, 5/18/2017
    #constant = (gv.pindex-1)*nodensity_e*gammamin**(gv.pindex-1)    # distribution constant

#    constant = (gv.pindex-1)*nodensity_e*gammamin**(gv.pindex-1) * (gammamin-1)/gammamin   # distribution constant - old version - Brian, 7/6/2020

# Use this one - Brian, 7/6/2020
#    constant = (gv.pindex-1)*nodensity_e*gammamin**(gv.pindex-1) * (gammamin-1)/(gammamin - (gv.pindex-2)/(gv.pindex-1)) # distribution constant - new version - Brian, 7/6/2020
    
    
    #    constant = (gv.pindex-1)*nodensity_e*gammamin**(gv.pindex-1) * (gammamin-1)/gammamin/3*1.5**(gv.pindex-1)   # distribution constant
#    constant = (gv.pindex-1)*nodensity_e*(gammamin-1)**(gv.pindex-1) * ((gammamin)/(gammamin-1))**(-gv.pindex)   # distribution constant

    # First set of corrected equations - Always add 1 to gammamin, and correct the constant so that the total energy works out right.  Fewer than 100% of electrons are always accelerated.
#    gammamin = ((gv.pindex-2)/(gv.pindex-1))*eps*ethermal/(nodensity_e*gv.me*gv.cl**2) + 1.  # added 1, not completely sure this is right - Brian, 4/6/2017 - Brian, 9/17/2015  #minimum gamma
#    constant = (gv.pindex-1)*nodensity_e*gammamin**(gv.pindex-1) * (gammamin-1)/(gammamin - (gv.pindex-2)/(gv.pindex-1))   # distribution constant - new version - Brian, 7/6/2020


    # Second set of corrected equations - Set gammamin to 1 if it is less than 1, the change the constant to accelerate fewer electrons.
    gammamin0 = ((gv.pindex-2)/(gv.pindex-1)) * (eps*ethermal/(nodensity_e*gv.me*gv.cl**2) + 1.)  # added 1 inside equation.  gammamin can still be less that 1 at this step. - Brian, 7/6/2020
    constant0 = (gv.pindex-1)*nodensity_e*gammamin0**(gv.pindex-1)  # No correction here - Brian, 7/6/2020
    
    gammamin1 = (ethermal*nodensity)/(ethermal*nodensity)
    
    constant1 = ((gv.pindex-2)*(gv.pindex-1))*eps*ethermal/(gv.me*gv.cl**2)
                                                            
    gammamin = np.where(gammamin0 < 1,gammamin1,gammamin0)
    constant = np.where(gammamin0 < 1,constant1,constant0)
    
                                                                                 
                                                                                 
    values = {'gammamin': gammamin, 'constant': constant}
    
    return values






#def jitter_power_stored(single_electron, single_omegas, n_elemf, 
#                 wmax, wmin):

def jitter_power_stored(n_elemf, 
                 wmax, wmin):
    
    gv.intswitch = 1
    
    #intergral accuracy
    criterion = 1e-4
    

    powerarray = np.zeros(n_elemf)
    alpha = np.zeros(n_elemf)
    
    # calculate the minimum gamma nd constant for the electron disribution
    
    #values = elecdist(gv.epsilon, gv.thermal)   # get the minimum gamma and distribution constant
    values = elecdist(gv.epsilon, gv.thermal,gv.gammaint,gv.nodensity)   # get the minimum gamma and distribution constant
    gv.mingamma = values['gammamin']                #lower limit for integration
    gv.k = values['constant']                      # constant in electron distribution
    


    omegas=np.copy(wmax)       # this is an input of arbitrary values of omega - Brian, 4/6/2017
    
    
    prefactor = 0

    bfield = np.sqrt(gv.bsq)
    SA = np.sin(np.pi/2)
    wc = 3*gv.mingamma**2*gv.q*bfield/(2*gv.me*gv.cl)
    prefactor = np.sqrt(3)*gv.q**3*bfield*SA/(2*np.pi*gv.me*gv.cl**2)
    gv.wjm = wc*10


    restore_synch = io.idl.readsav('powerarray_2.5_synch_00.00.idl', python_dict = True)
    
    powerarray0 = restore_synch['powerarray0']
    alpha0 = restore_synch['alpha0']
    mingamma0 = restore_synch['mingamma0']
    wjm0 = restore_synch['wjm0']
    prefactor0 = restore_synch['prefactor0']
    omegas0 = restore_synch['omegas0']
    k0 = restore_synch['k0']
    single_electron0 = restore_synch['single_electron0']
    single_omegas0 = restore_synch['single_omegas0']

    
    log_powerarray3_interf = interpolate.interp1d(np.log10(omegas0/wjm0*gv.wjm), np.log10(powerarray0*(mingamma0/gv.mingamma)**(gv.pindex-1e0)*gv.k/k0*prefactor/prefactor0), bounds_error = False, fill_value = 'extrapolate')
    powerarray = 10**(log_powerarray3_interf(np.log10(omegas)))

    log_powerarray4_interf = interpolate.interp1d(np.log10(omegas0/wjm0*gv.wjm), np.log10(alpha0*(mingamma0/gv.mingamma)**(gv.pindex)*gv.k/k0*prefactor/prefactor0*(gv.wjm/wjm0)**(-2.)), bounds_error = False, fill_value = 'extrapolate')
    alpha = 10**(log_powerarray4_interf(np.log10(omegas)))


    
    #print('done jitter_power',omegas[0],omegas0[0]/wjm0*gv.wjm,(mingamma0/gv.mingamma)**(gv.pindex-1e0)*gv.k/k0*prefactor/prefactor0,powerarray[0])
    
    single_electron0 = single_electron0 #
    single_omegas0 = single_omegas0 #
    omegas0 = omegas
    powerarray0 = powerarray
    alpha0 = alpha
    n_elemf0 = n_elemf
    wmax0 = wmax
    wmin0 = wmin
    pindex0 = gv.pindex
    wjm0 = gv.wjm
    mingamma0 = gv.mingamma
    k0 = gv.k
    epsilon0 = gv.epsilon
    prefactor0 = prefactor
    bsq0 = gv.bsq
    
    jitter_power_array = {'single_electron0': single_electron0, 'single_omegas0': single_omegas0, 
                          'omegas0': omegas0, 'powerarray0': powerarray0, 'alpha0': alpha0,
                          'n_elemf0': n_elemf0, 'wmax0': wmax0, 'wmin0': wmin0, 'pindex0': pindex0,
                          'wjm0': wjm0, 'mingamma': mingamma0, 'k0': k0, 'epsilon0': epsilon0, 
                          'prefactor0': prefactor0, 'bsq0': bsq0}
    
    
    #output = open('powerarray_' + str(gv.pindex) + '_synch_' + str(gv.theta*180/np.pi) + '.p', 'wb')
    #pickle.dump(jitter_power_array, output)
    #output.close()
    
    return powerarray, alpha, omegas, single_electron0, single_omegas0

#    return powerarray, alpha, omegas



@functools.lru_cache()
def interp_functions_memoized():

    restore_synch, = jitter_power_pindex(gv.pindex)

    gv.intswitch = 1
    criterion = 1e-4


    #powerarray = np.zeros(n_elemf)
    #alpha = np.zeros(n_elemf)

    # calculate the minimum gamma nd constant for the electron disribution

    #values = elecdist(gv.epsilon, gv.thermal)   # get the minimum gamma and distribution constant
    values = elecdist(gv.epsilon, gv.thermal,gv.gammaint,gv.nodensity)   # get the minimum gamma and distribution constant
    gv.mingamma = values['gammamin']                #lower limit for integration
    gv.k = values['constant']                      # constant in electron distribut


    prefactor = 0

    bfield = np.sqrt(gv.bsq)
    SA = np.sin(np.pi/2)
    wc = 3*gv.mingamma**2*gv.q*bfield/(2*gv.me*gv.cl)
    prefactor = np.sqrt(3)*gv.q**3*bfield*SA/(2*np.pi*gv.me*gv.cl**2)
    gv.wjm = wc*10

    powerarray0 = restore_synch['powerarray0']
    alpha0 = restore_synch['alpha0']
    #    mingamma0 = restore_synch['mingamma0']
    mingamma0 = restore_synch['mingamma']
    wjm0 = restore_synch['wjm0']
    prefactor0 = restore_synch['prefactor0']
    omegas0 = restore_synch['omegas0']
    k0 = restore_synch['k0']
    single_electron0 = restore_synch['single_electron0']
    single_omegas0 = restore_synch['single_omegas0']



#    log_powerarray3_interf = interpolate.interp1d(np.log10(omegas0/wjm0*gv.wjm), np.log10(powerarray0*(mingamma0/gv.mingamma)**(gv.pindex-1e0)*gv.k/k0*prefactor/prefactor0), bounds_error = False, fill_value = 'extrapolate')
    log_powerarray3_interf = interpolate.interp1d(np.log10(omegas0), np.log10(powerarray0), bounds_error = False, fill_value = 'extrapolate')
#    powerarray = 10**(log_powerarray3_interf(np.log10(omegas)))

#    log_powerarray4_interf = interpolate.interp1d(np.log10(omegas0/wjm0*gv.wjm), np.log10(alpha0*(mingamma0/gv.mingamma)**(gv.pindex)*gv.k/k0*prefactor/prefactor0*(gv.wjm/wjm0)**(-2.)), bounds_error = False, fill_value = 'extrapolate')
    log_powerarray4_interf = interpolate.interp1d(np.log10(omegas0), np.log10(alpha0), bounds_error = False, fill_value = 'extrapolate')
#    alpha = 10**(log_powerarray4_interf(np.log10(omegas)))

    return log_powerarray3_interf,log_powerarray4_interf




def jitter_power_memoized(n_elemf, 
                 wmax, wmin):
    
    restore_synch, = jitter_power_pindex(gv.pindex)

    gv.intswitch = 1
    
    #intergral accuracy
    criterion = 1e-4
    

    powerarray = np.zeros(n_elemf)
    alpha = np.zeros(n_elemf)
    
    # calculate the minimum gamma nd constant for the electron disribution
    
    #values = elecdist(gv.epsilon, gv.thermal)   # get the minimum gamma and distribution constant
    values = elecdist(gv.epsilon, gv.thermal,gv.gammaint,gv.nodensity)   # get the minimum gamma and distribution constant
    gv.mingamma = values['gammamin']                #lower limit for integration
    gv.k = values['constant']                      # constant in electron distribution
    


    omegas=np.copy(wmax)       # this is an input of arbitrary values of omega - Brian, 4/6/2017
    
    
    prefactor = 0

    bfield = np.sqrt(gv.bsq)
    SA = np.sin(np.pi/2)
    wc = 3*gv.mingamma**2*gv.q*bfield/(2*gv.me*gv.cl)
    prefactor = np.sqrt(3)*gv.q**3*bfield*SA/(2*np.pi*gv.me*gv.cl**2)
    gv.wjm = wc*10


#    restore_synch = io.idl.readsav('powerarray_2.5_synch_00.00.idl', python_dict = True)
    
#    restore_synch, = jitter_power_pindex(gv.pindex)


    powerarray0 = restore_synch['powerarray0']
    alpha0 = restore_synch['alpha0']
#    mingamma0 = restore_synch['mingamma0']
    mingamma0 = restore_synch['mingamma']
    wjm0 = restore_synch['wjm0']
    prefactor0 = restore_synch['prefactor0']
    omegas0 = restore_synch['omegas0']
    k0 = restore_synch['k0']
    single_electron0 = restore_synch['single_electron0']
    single_omegas0 = restore_synch['single_omegas0']


# new way - Brian, 4/15/2018
    log_powerarray3m_interf,log_powerarray4m_interf = interp_functions_memoized()

# old way - Brian, 4/15/2018
#    log_powerarray3_interf = interpolate.interp1d(np.log10(omegas0/wjm0*gv.wjm), np.log10(powerarray0*(mingamma0/gv.mingamma)**(gv.pindex-1e0)*gv.k/k0*prefactor/prefactor0), bounds_error = False, fill_value = 'extrapolate')
#    powerarray = 10**(log_powerarray3_interf(np.log10(omegas)))

#    log_powerarray4_interf = interpolate.interp1d(np.log10(omegas0/wjm0*gv.wjm), np.log10(alpha0*(mingamma0/gv.mingamma)**(gv.pindex)*gv.k/k0*prefactor/prefactor0*(gv.wjm/wjm0)**(-2.)), bounds_error = False, fill_value = 'extrapolate')
#    alpha = 10**(log_powerarray4_interf(np.log10(omegas)))


#    print('initial = ',(alpha0*(mingamma0/gv.mingamma)**(gv.pindex)*gv.k/k0*prefactor/prefactor0*(gv.wjm/wjm0)**(-2.))[0])
#    print('initial = ',gv.k,k0,prefactor/prefactor0,gv.wjm/wjm0 )
#    print('old = ',10**(log_powerarray3_interf(np.log10(1e10))))
#    print('new = ',10**(log_powerarray3m_interf(np.log10(1e10*wjm0/gv.wjm)))*(mingamma0/gv.mingamma)**(gv.pindex-1e0)*gv.k/k0*prefactor/prefactor0 )
    #print('new2 = ',10**(log_powerarray3m_interf(np.log10(1e10))))
#    print('old = ',alpha[0])
 

# new way - Brian, 4/15/2018
    powerarray = 10**(log_powerarray3m_interf(np.log10(omegas*wjm0/gv.wjm)))*(mingamma0/gv.mingamma)**(gv.pindex-1e0)*gv.k/k0*prefactor/prefactor0
    alpha = 10**(log_powerarray4m_interf(np.log10(omegas*wjm0/gv.wjm)))*(mingamma0/gv.mingamma)**(gv.pindex)*gv.k/k0*prefactor/prefactor0*(gv.wjm/wjm0)**(-2.) #-2.
 


#    print('new = ',alpha[0])
#    print('new = ',10**(log_powerarray4m_interf(np.log10(omegas[0]))) )  #*(mingamma0/gv.mingamma)**(gv.pindex)*gv.k/k0*prefactor/prefactor0*(gv.wjm/wjm0)**(-2.))
# 
#    print('new2 = ',10**(log_powerarray4m_interf(np.log10(omegas[0]))))
#
#    print('factors = ',gv.wjm/wjm0, mingamma0/gv.mingamma, gv.k/k0,prefactor/prefactor0)
#
#    print('factor2 = ',(mingamma0/gv.mingamma)**(gv.pindex)*gv.k/k0*prefactor/prefactor0*(gv.wjm/wjm0)**(-2.) )

    #print('done jitter_power',omegas[0],omegas0[0]/wjm0*gv.wjm,(mingamma0/gv.mingamma)**(gv.pindex-1e0)*gv.k/k0*prefactor/prefactor0,powerarray[0])
    
    #single_electron0 = single_electron0 #
    #single_omegas0 = single_omegas0 #
    #omegas0 = omegas
    #powerarray0 = powerarray
    #alpha0 = alpha
    #n_elemf0 = n_elemf
    #wmax0 = wmax
    #wmin0 = wmin
    #pindex0 = gv.pindex
    #wjm0 = gv.wjm
    #mingamma0 = gv.mingamma
    #k0 = gv.k
    #epsilon0 = gv.epsilon
    #prefactor0 = prefactor
    #bsq0 = gv.bsq
    #
    #jitter_power_array = {'single_electron0': single_electron0, 'single_omegas0': single_omegas0, 
    #                      'omegas0': omegas0, 'powerarray0': powerarray0, 'alpha0': alpha0,
    #                      'n_elemf0': n_elemf0, 'wmax0': wmax0, 'wmin0': wmin0, 'pindex0': pindex0,
    #                      'wjm0': wjm0, 'mingamma': mingamma0, 'k0': k0, 'epsilon0': epsilon0, 
    #                      'prefactor0': prefactor0, 'bsq0': bsq0}
    
    
    #output = open('powerarray_' + str(gv.pindex) + '_synch_' + str(gv.theta*180/np.pi) + '.p', 'wb')
    #pickle.dump(jitter_power_array, output)
    #output.close()
    
    return powerarray, alpha, omegas, single_electron0, single_omegas0


#    return powerarray, alpha, omegas
