import numpy as np
import pickle
import shelve

from matplotlib import pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy import interpolate
from global_var import sysvars as gv
import integrate_total_power_interp as itpi


#def make_image(savefile,t,f,x_num=500,y_num=500,z=0.047, z_origional=0.047, dl=209.8):

def make_image(data,time,freq,time_origional,t,f,x_num=500,y_num=500,z=0.047, z_origional=0.047, dl=209.8):

# savefile = file to load data from
# t = time index in the file data to make an image for.
# f = frequency index in the file data to make an image for.  Could be changed to use an interpolated frequency later.
# x_num = size of x dimension of output image
# y_num = size of y dimension of output image
# z = redshift of the afterglow
# z_origional = redshift at which the data in the file was created.  Usually 0.047
# dl = luminosity distacne to the afterglow

    # Luminosity distance in Mpc for z=0.09, calculated using Ned Wright's Cosmology Calculator
    # (http://www.astro.ucla.edu/~wright/CosmoCalc.html)
    # with H_0 = 69.6, Omega_M = 0.286, Omega_vac = 0.714 
    # dl is Angular diameter distance times (1+z)**2, taking into account factor of (1+z) for time dialation, frequency change, 
    # and 2 for beaming, because source is moving away. - Brian, 4/14/2017


    
    da = dl / (1.+z)**2

#savefile = './afterglow_sych_spectrum.lazzati_2017.closer25.published.small.testphi.n-4.p'


# Load all the data from the file

    #data_in = open(savefile, 'rb')
    #data = pickle.load(data_in)
    #data_in.close()

    print('data loaded')


    h = 4.135667662e-15     #eV/s
    to_ev = 4.1356691e-15 /2/np.pi  #convert omega(frequency) to eV
    Mpc = 3.08e24  # Mpc in cm



# Use intgrate_total_power_interp to load some data, just in case we need it.

#    time='0'
#
#    radio_f=np.array([8.635e8,1.4e9,3.e9,6.e9,8.e9,1.2e10,2.418e17])*h/(to_ev*2*np.pi)
#
    #radio,xray,optical,optical_ab,time,freq, \
    #jy, time_origional, glow \
    #= itpi.integrate_total_power_interp(savefile, radio_f, z, z_origional, dl, \
    #                                xray_min=0.3, xray_max=10.0, optical_f='0', time=time)

    print('Done with integrate_total_power_interp')


# Get the needed data:

    total_power_arr = data['total_power_arr']
    angle_int_arr = data['angle_int_arr']
    glow_arr = data['glow_arr']

    time_image = time[t]
    freq_image = freq[f]

    print('Time of image = ',time_image)
    print('Frequency of image = ',freq_image)


# Create array for phi angle values.  Will need this later. 
# phi needs to match how phi was create by the Blandford-McKee code.

# Note the phi only goes from 0 to 180 degrees, becuase everything is symmetrical across the jet axis (for a 1-d energy distribution coming from a 2-d hydro simulation.)

    phi_slices=len(total_power_arr[0,0,:])       # number of slices in phi direction

    phi=(np.arange(phi_slices/2)/(phi_slices/2))**2 * 90.  # the **2 is needed for the current version, must match how phi is distributed when making the file - Brian, 4/14/2017

    phi=np.concatenate(([0],phi,[90],180-phi[::-1],[180],))               

#phi_gap=(np.floor(phi[1:]*(phi_bins/360))-np.floor(phi[0:len(phi)-1]*(phi_bins/360))) * (360/phi_bins) * 2.   # d_phi between all angles, multiply by two becuase we are only ding half a circle when we make data. - Brian 4/14/2017

    phi_gap=(phi[1:]-phi[0:len(phi)-1]) * 2.   # d_phi between all angles, multiply by two becuase we are only ding half a circle when we make data. - Brian 4/14/2017

    phi_gap_arr=np.zeros([len(time_origional),len(phi_gap)])
    for i in range(len(time_origional)):
        phi_gap_arr[i,:]=np.copy(phi_gap)       # filled array of [freq,phi] containting all phi_gap


    
# Make coordinates of the points from the data file at the given time:

     # glow_arr[1,...] is the projected distance from the jet axis (z direction in Blandford-McKee code)

    
    radius_all = np.zeros([len(glow_arr[1,-1,:,t,0]),len(glow_arr[1,-1,0,t,:])+2])
    radius_all[:,1:-1] = glow_arr[1,-1,:,t,:]
    radius_all[:,0] = glow_arr[1,-1,:,t,0]
    radius_all[:,-1] = glow_arr[1,-1,:,t,-1]

    print(radius_all.shape)
    
    #radius = glow_arr[1,99,:,t,0]
    radius = radius_all[:,0]
    mid_phi = (phi[0:-1]+phi[1:])/2.
    mid_phi_full,rad_full = np.meshgrid(mid_phi,radius)
    
    
                                       
    x_coord = np.sin(mid_phi_full*np.pi/180.) * radius_all #* glow_arr[1,99,:,t,:]
    y_coord = np.cos(mid_phi_full*np.pi/180.) * radius_all #* glow_arr[1,99,:,t,:]



# Extract the data needed for this image:

    image_tp=np.zeros((len(total_power_arr[t,:,0]),len(total_power_arr[t,0,:])+2))
    image_tp[:,1:-1]=total_power_arr[t,:,:]
    image_tp[:,0]=total_power_arr[t,:,0]
    image_tp[:,-1]=total_power_arr[t,:,-1]
    
#    image_tp=total_power_arr[t,:,:]                                           
#    print(image_tp.shape)

    image_ai=np.zeros((len(angle_int_arr[f,:,t,0]),len(angle_int_arr[f,0,t,:])+2))
    image_ai[:,1:-1]=angle_int_arr[f,:,t,:]
    image_ai[:,0]=angle_int_arr[f,:,t,0]
    image_ai[:,-1]=angle_int_arr[f,:,t,-1]

#    image_ai=angle_int_arr[f,:,t,:]
#    print(image_ai.shape)


# Make the x and y coordinates to create the image on.  These are the point that the data will be interpolated to.
# Using an x and y grid in this case, but could be a r and phi grid instead if you wanted.

    y_coord_size = np.max(y_coord)-np.min(y_coord) 
    y_min = np.min(y_coord) - y_coord_size/8.    #np.min(y_coord)
    y_max = np.max(y_coord) + y_coord_size/8.    #np.max(y_coord)

# This x_max is right for making a square image and assuming it's taller than it is wide.  
# If you wanted something else, this would need to be modified.
    x_max = (y_max - y_min)/2. 
    x_min = -x_max

    
# There is only data for half the grid in phi, becuase it's symetrical.  So do everything on half the grid then will copy it later.

    x_num_gridpoints = np.int(x_num/2)   # assumes an even number of points in the x direction
    y_num_gridpoints = y_num
    
    x_gridpoints = np.linspace(x_min,x_max,x_num_gridpoints*2)
    x_gridpoints = x_gridpoints[x_num_gridpoints:]   

    x_gridpoints_full = np.concatenate([0-np.flip(x_gridpoints,0),x_gridpoints])

    y_gridpoints = np.linspace(y_min,y_max,y_num_gridpoints)

    
    x_grid,y_grid = np.meshgrid(x_gridpoints,y_gridpoints)

    # Convert the coordinates into 1-d arrays to usde for interpolation
    coords = np.zeros((len(x_coord.flatten()),2))
    coords[:,0] = x_coord.flatten()
    coords[:,1] = y_coord.flatten()


# Do the interpolation in log space:
# This will do linear interpolation.

    image_ai_flat = np.log(image_ai.flatten())
    image_interpol_log = interpolate.griddata(coords,image_ai_flat,(x_grid,y_grid),method='linear')


# convert back to linear space:
    image_interpol = np.exp(image_interpol_log)

# Reflect the image over the y axis to fill in the other side:

    image_interpol_full = np.zeros((y_num_gridpoints,2*x_num_gridpoints))
    image_interpol_full[:,0:x_num_gridpoints] = np.fliplr(image_interpol)
    image_interpol_full[:,x_num_gridpoints:] = image_interpol


# Get rid of all the NaNs where the interpolated points are outside the origional data.
# Note that if you use a different interpolation scheme, you might need to set the image outside the bounds of the data to zero another way.

    image_interpol = np.nan_to_num(image_interpol)
    image_interpol_full = np.nan_to_num(image_interpol_full)

    

## Now I'm going to convert the image to something in real units, then put it in terms of Jy/arcsec^2

# Convert the x and y coordinates to arcseconds

    x_arcsec = x_gridpoints_full/Mpc/da /(2.*np.pi/360./3600.)
    y_arcsec = y_gridpoints/Mpc/da /(2.*np.pi/360./3600.)


# Need a factor of 2*np.pi to conver from omega freqeucny to to nu frequency space.

    image_erg_cm2 = image_interpol_full * 2.*np.pi   # erg/s/cm2/Hz emitted
    # print(np.max(image_erg_cm2))


# Convert from emitted energy to reveived energy per arcsec^2
# Don't need to know the distance to do this.
    image_erg_arcsec2 = image_erg_cm2 * (2.*np.pi/360./3600.)**2.    # erg/s/Hz/arcsec2 received
    
    # print(np.max(image_erg_arcsec2))

# Convert from erg/s/Hz to Jy:
    image_Jy_arcsec2 = image_erg_arcsec2 * 1.e23   # Jy/arcsec^2




# Find the centroid of the emission along the y-axis.  Along the x-axis it's always zero for symetrical data.
    
    midval = np.sum(image_interpol)/2.

    ysum = np.sum(image_interpol,axis=1)

    ycumsum = np.cumsum(ysum)

    wheresum = np.where((ycumsum < midval),np.arange(y_num_gridpoints),y_gridpoints*0.)

    centroid = np.int(np.max(wheresum))
    centroid_cm = y_gridpoints[centroid]
    centroid_arcsec = y_arcsec[centroid]
    
    # print(centroid,centroid_cm,centroid_arcsec)
    
    
    return image_erg_cm2,image_Jy_arcsec2,x_gridpoints_full,y_gridpoints,x_arcsec,y_arcsec, \
               time_image,freq_image,centroid_cm,centroid_arcsec

