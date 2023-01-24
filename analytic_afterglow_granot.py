import numpy as np
from matplotlib import pyplot as plt

def analytic_afterglow_granot(t_arr, wmax, z, dl):

    # Based on Granot & Sari, 2002

    # t_arr = times seconds in observer frame
    # wmax = frequencies (in omega) in observer frame

    # variables
    eiso = 1e53 #2.5e51 #1.e49                   # Total isotropic energy
    external_nodensity = 1e0 #1.e-1     # external density
    epsilon_e = 1e-1      #electron energy fraction
    eqparb = 1e-2 #1e-2 #1e-2       #magnetic energy fraction
    pindex = 2.3  # electron spectrum index

    epsilon = epsilon_e * (pindex-2)/(pindex-1)  # changing to epsilon_e_bar in Garnot & Sari 2002 notation.

#    z = 4 #10.0 #0.047 #0.5 # 0.047  # 0.09     # Redshift (need to know this for each file)

#    dl = 36237. #105113. #209.8 #2584.7 #209.8  # 414.2   
             # Luminosity distance in Mpc for z=0.09, calculated using Ned Wright's Cosmology Calculator (http://www.astro.ucla.edu/~wright/CosmoCalc.html)
             # with H_0 = 69.6, Omega_M = 0.286, Omega_vac = 0.714 
             # dl is Angular diameter distance times (1+z)**2, taking into account factor of (1+z) for time dialation, frequency change, 
             # and 2 for beaming, because source is moving away. - Brian, 4/14/2017
    
    

    t_days = t_arr/86400. # time in days in observer frame
    nu_14 = wmax/2./np.pi/1.e14    # frequencies in nu in observer frame / 1.e14 Hz
    dl_28 = dl*3.08e24/1.e28      # luminosity distance over 1.e28 cm
    e_52  = eiso/1.e52    # eiso / 1e52 erg

    print('test',nu_14)

    # for k=0, ISM:

    power_a = np.zeros((len(t_arr),len(wmax)))
    power_b = np.copy(power_a)
    power_c = np.copy(power_a)
    power_d = np.copy(power_a)
    power_e = np.copy(power_a)
    power_f = np.copy(power_a)
    power_g = np.copy(power_a)
    power_h = np.copy(power_a)

   
    for i in range(len(t_arr)):
        power_a[i,:]  = 1.18*(4.59-pindex)*1.e8 * (1+z)**(9/4) * eqparb**(-1/4) * external_nodensity**(-1/2) * e_52**(1/4) * t_days[i]**(5/4) * dl_28**(-2) * nu_14**(5/2)
        power_b[i,:]  = 4.20*(3*pindex+2)/(3*pindex-1)*1.e9 * (1+z)**(5/2) * epsilon * external_nodensity**(-1/2) * e_52**(1/2) * t_days[i]**(1/2) * dl_28**(-2) * nu_14**(2)
        power_c[i,:]  = 8.01e5 * (1+z)**(27/16) * eqparb**(-1/4) * external_nodensity**(-5/16) * e_52**(7/16) * t_days[i]**(11/16) * dl_28**(-2) * nu_14**(11/8)
        power_d[i,:]  = 27.9*(pindex-1)/(3*pindex-1) * (1+z)**(5/6) * epsilon**(-2/3) * eqparb**(1/3) * external_nodensity**(1/2) * e_52**(5/6) * t_days[i]**(1/2) * dl_28**(-2) * nu_14**(1/3)
        power_e[i,:]  = 73.0 * (1+z)**(7/6) * eqparb * external_nodensity**(5/6) * e_52**(7/6) * t_days[i]**(1/6) * dl_28**(-2) * nu_14**(1/3)
        power_f[i,:]  = 6.87 * (1+z)**(3/4) * eqparb**(-1/4) * e_52**(3/4) * t_days[i]**(-1/4) * dl_28**(-2) * nu_14**(-1/2)
        power_g[i,:]  = 0.461*(pindex-0.04)*np.exp(2.53*pindex) * (1+z)**((3+pindex)/4) * epsilon**(pindex-1) * eqparb**((pindex+1)/4) *external_nodensity**(1/2) * e_52**((3+pindex)/4) * t_days[i]**(3*(1-pindex)/4) * dl_28**(-2) * nu_14**((1-pindex)/2)
        power_h[i,:]  = 0.855*(pindex-0.98)*np.exp(1.95*pindex) * (1+z)**((2+pindex)/4) * epsilon**(pindex-1) * eqparb**((pindex-2)/4) * e_52**((2+pindex)/4) * t_days[i]**((2-3*pindex)/4) * dl_28**(-2) * nu_14**(-pindex/2)


    print(dl_28,e_52)

    # for k=2, wind:

    v_wind = 1000.      # km/s
    m_dot_wind = 1.e-5  #*24.  ;3.d-5  ; m_sun/year
    external_A = (m_dot_wind*1.98892*1.e33/(365.25*24.*3600.))/(v_wind*1.e5)/4./np.pi/1.67e-24 * 1.67e-24/5e11 # A* # for wind ;We're in a wind situation --Dominic, 11/30/13


    power_a2 = np.zeros((len(t_arr),len(wmax)))
    power_b2 = np.copy(power_a2)
    power_c2 = np.copy(power_a2)
    power_d2 = np.copy(power_a2)
    power_e2 = np.copy(power_a2)
    power_f2 = np.copy(power_a2)
    power_g2 = np.copy(power_a2)
    power_h2 = np.copy(power_a2)

   
    for i in range(len(t_arr)):
        power_a2[i,:]  = 2.96*(4.59-pindex)*1.e7 * (1+z)**(7/4) * eqparb**(-1/4) * external_A**(-1) * e_52**(3/4) * t_days[i]**(7/4) * dl_28**(-2) * nu_14**(5/2)
        power_b2[i,:]  = 1.33*(3*pindex+2)/(3*pindex-1)*1.e9 * (1+z)**(2) * epsilon * external_A**(-1) * e_52 * t_days[i] * dl_28**(-2) * nu_14**(2)
        power_c2[i,:]  = 3.28e5 * (1+z)**(11/8) * eqparb**(-1/4) * external_A**(-5/8) * e_52**(3/4) * t_days[i] * dl_28**(-2) * nu_14**(11/8)
        power_d2[i,:]  = 211*(pindex-1)/(3*pindex-1) * (1+z)**(4/3) * epsilon**(-2/3) * eqparb**(1/3) * external_A * e_52**(1/3) * dl_28**(-2) * nu_14**(1/2)
#        power_e[i,:]  = 0
        power_f2[i,:]  = 6.68 * (1+z)**(3/4) * eqparb**(-1/4) * e_52**(3/4) * t_days[i]**(-1/4) * dl_28**(-2) * nu_14**(-1/2)
        power_g2[i,:]  = 3.82*(pindex-0.18)*np.exp(2.54*pindex) * (1+z)**((5+pindex)/4) * epsilon**(pindex-1) * eqparb**((pindex+1)/4) *external_A * e_52**((1+pindex)/4) * t_days[i]**((1-3*pindex)/4) * dl_28**(-2) * nu_14**((1-pindex)/2)
        power_h2[i,:]  = 0.0381*(7.11-pindex)*np.exp(2.76*pindex) * (1+z)**((2+pindex)/4) * epsilon**(pindex-1) * eqparb**((pindex-2)/4) * e_52**((2+pindex)/4) * t_days[i]**((2-3*pindex)/4) * dl_28**(-2) * nu_14**(-pindex/2)


    # Normally, spectra will go from low to high frequency, and from less to more slef-abosrbed, B-D-G-H, B-A-G-H, B-A-H
    # For fast cooling, they will for B-C-F-H, B-C-E-F-H, B-A-H

 
  
    return power_a,power_b,power_c,power_d,power_e,power_f,power_g,power_h
#    return power_a2,power_b2,power_c2,power_d2,power_e2,power_f2,power_g2,power_h2
  





