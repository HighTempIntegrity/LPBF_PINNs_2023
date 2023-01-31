import torch
import numpy as np
import sobol_seq
from pyDOE import lhs


def generator_points(samples, dim, random_seed, type_of_points, boundary):
    
    if type_of_points == "sobol":
        print(type_of_points)
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        return torch.from_numpy(data).type(torch.FloatTensor)
    
    elif type_of_points == "moving_center":
        print(type_of_points)
        skip = random_seed
        '''pts=1 #how many centered points per step
        step=1 #how many time steps in center'''
        # prob=0.25 #factor on how large the randomness of the center points is: 0= 100% centered, 1=completely random
        ring_thick = 0.3
        s_ring_rad = 0.0
        exponent = 1
        n=int(samples/10) #number of total center-bias points

        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain

        #CHANGE IF YOU CHANGE THE BUFFER Time (minimum extrema of t)!
        t0=0.1 #buffer time

        tlaser=1 #time the laser is actually on
        sum=t0+tlaser
        t0=t0/sum #normed for scaling
        tlaser=tlaser/sum #normed for scaling

        datat = np.full((samples, 1), np.nan)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        i=0
        p=1

        dataY[:, 0] = np.random.laplace(0.5, 1 / 12., samples)
        dataZ[:, 0] = np.random.beta(5, 1, samples)
        for j in range(samples-n-1): #majority of coll points sampled around laser
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-3, seed)
            datat[j, :]=rnd[0]**exponent
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            if (dim > 4):
                dataP[j, :] = rnd[1:]
        for j in range(samples - n-1, samples): #Center-bias: some points directly in laser center
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            datat[j, :] = rnd[0]**exponent

            radi =  s_ring_rad + rnd[1] * ring_thick
            theta = rnd[2] * np.pi / 2
            phi  = rnd[3] * 2 * np.pi
            dataX[j, :] = x0  + (datat[j, :] - t0) / tlaser * xspeed + radi * np.cos (phi) * np.sin(theta)
            dataY[j, :] = 0.5 + radi * np.sin (phi) * np.sin(theta)
            dataZ[j, :] = 1- radi * np.cos(theta)
            if(dim>4):
                dataP[j,:]= rnd[4:]

        if dim > 4:
            data=np.concatenate((datat, dataX, dataY, dataZ, dataP), axis=1)
        else:
            data=np.concatenate((datat, dataX, dataY, dataZ), axis=1)

        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "initial_center":
        print(type_of_points)
        skip = random_seed
        # prob=0.25 #factor on how large the randomness of the center points is: 0= 100% centered, 1=completely random
        ring_thick = 0.3
        s_ring_rad = 0.0
        n=int(samples/10) #number of total center-bias points

        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain

        #CHANGE IF YOU CHANGE THE BUFFER Time (minimum extrema of t)!
        t0=0.1 #buffer time

        tlaser=1 #time the laser is actually on
        sum=t0+tlaser
        t0=t0/sum #normed for scaling
        tlaser=tlaser/sum #normed for scaling

        datat = np.full((samples, 1), 0.1)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        i=0
        p=1

        dataY[:, 0] = np.random.laplace(0.5, 1 / 12., samples)
        dataZ[:, 0] = np.random.beta(5, 1, samples)
        for j in range(samples-n-1): #majority of coll points sampled around laser
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-3, seed)
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            if (dim > 4):
                dataP[j, :] = rnd[1:]
        for j in range(samples - n-1, samples): #Center-bias: some points directly in laser center
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
   
            radi =  s_ring_rad + rnd[1] * ring_thick
            theta = rnd[2] * np.pi / 2 
            phi  = rnd[3] * 2 * np.pi
            dataX[j, :] = x0  + radi * np.cos (phi) * np.sin(theta)
            dataY[j, :] = 0.5 + radi * np.sin (phi) * np.sin(theta)
            dataZ[j, :] = 1- radi * np.cos(theta)
            if(dim>4):
                dataP[j,:]= rnd[4:]

        if dim > 4:
            data=np.concatenate((datat, dataX, dataY, dataZ, dataP), axis=1)
        else:
            data=np.concatenate((datat, dataX, dataY, dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
