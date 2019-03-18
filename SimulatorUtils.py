import numpy as np
import ModifiedNCP_PRIORxbblocks
import scipy.integrate as integrate
from scipy import special as spec
import numpy.ma as ma
import math
import matplotlib.pyplot as plt


def CDF_c(time, T):
    '''
    Outputs the CDF function of a constant process
    Parameters
    ----------
    time:  array of floats
        Independant variable values     
    T     : float
        Total range of the time array
    Returns
    ----------
    time/T: array of floats
    '''
    return (time/T)

def CDF_g(time, A_peak, sigma, t_peak, N_g):
    '''
    Outputs the CDF function of a Gaussian
    Parameters
    ----------
    time:  array of floats
        Independant variable values   
        
    A_peak: float
        Amplitude of the Gaussian
        
    sigma: float
        Standard deviation of the Gaussian
        
    t_peak: float
        Time of the peak of the Gaussian
        
    N_g: int
        Number of expected counts in the Gaussian

    Returns
    ----------
    CDF of the Gaussian: array of floats
    '''
    return (A_peak*sigma/N_g)*np.sqrt(math.pi/2.)*(spec.erf(t_peak/(math.sqrt(2)*sigma))+spec.erf((time - t_peak)/(math.sqrt(2)*sigma))) 

def Gau(time, A_peak, sigma, t_peak):
    '''
    Outputs a Gaussian function
    Parameters
    ----------
    time:  array of floats
        Independant variable values   
        
    A_peak: float
        Amplitude of the Gaussian
        
    sigma: float
        Standard deviation of the Gaussian
        
    t_peak: float
        Time of the peak of the Gaussian
    Returns
    ----------
    The Gaussian function : array of floats
    '''
    return A_peak*np.exp(-(time-t_peak)**2./(2.*sigma**2))

def binnyboy(bin_size, T, time_evt):
    
    '''
    Bins the event data
    Parameters
    ----------
    bin_size:  int
        Duration of a bin (s)
    
    T     : int
        Duration of the observation (s)
        
    time_evt: array
        The event array to be binned (s)
    
    Returns
    ----------
    bin_array: array
        Array of all the bin times (s)

    binned_cr: array
        Array of all the binned count rates (ct/s)
        
    binned_cr_err: array
        Array of Poisson error of the binned count rates (ct/s)
    '''
    bin_array = (bin_size/2.+bin_size*np.arange(int(T/bin_size)))
    binned_cr = np.empty(len(bin_array))
    binned_cr_err = np.empty(len(bin_array))
    f = 0.
    for i in range(len(bin_array)):
        if i == 0:
            if len(bin_array) == 1:
                b = time_evt 
                binned_cr[i] = len(b)/bin_size
                binned_cr_err[i] =  math.sqrt(b.sum())/bin_size
            else:
                b = time_evt <= bin_array[i:i+2].mean()
                binned_cr[i] = b.sum()/bin_size
                binned_cr_err[i] =  math.sqrt(b.sum())/bin_size
        elif i == len(bin_array) - 1:
            a = time_evt > bin_array[i]-bin_size/2.
            #b = time_evt <= bin_array[i]+bin_size/2.
            #c = a & b
            binned_cr[i] = a.sum()/bin_size
            binned_cr_err[i] =  math.sqrt(a.sum())/bin_size
        else:
            a = time_evt > bin_array[i]-bin_size/2.
            b = time_evt <= bin_array[i]+bin_size/2.
            c = a & b
            binned_cr[i] = c.sum()/bin_size
            binned_cr_err[i] =  math.sqrt(c.sum())/bin_size
        f = f +  binned_cr[i]*bin_size
    #print(f)
    return bin_array, binned_cr, binned_cr_err

def flare_gen(T, frame_time, flare_t, flare_sig, flare_mean_cr, quiescence_cr, popt,pstd, nbootstrap=0, plot = False, alpha=5., mcrtoA=1.67):
    '''
    Generates a Gaussian flare given certain parameters
    Parameters
    ----------
    T     : float
        Duration of the observation (s)
    
    frame_time: float
        Frame time(s)
    
    flare_t: array of floats
        Time of the center of the flares (s)
        
    flare_sig: array of floats
        1 sigma of the Gaussian flares (s)
        
    flare_mean_cr: array of float
        Mean count rate of the flare (corresponding to the mean count rate of all the blocks forming a flare
        minus the count rate of quiescence) (ct/s)
        
    quiescence_cr: float
        Quiesence count rate (ct/s)

    popt: Array of float
	Best fit parameters for the Bayesian Blocks parameters

    pstd: Array of Float
        1-sigma errors on the best fit parameters for the Bayesian Blocks parameters

    nbootstrap: int
        Number of bootstraps to calculate the error of the Bayesian Blocks
        
    alpha: float
        number of sigma to integrate the counts around a flare
        
    Returns
    ----------
    time_evt: array
        Array of the event times
        
    info: holder
        Holder with a bunch of info about the BB
    '''
    t = np.linspace(0,T,int(T/frame_time+1))
    N_c = int(quiescence_cr * T)
    if type(flare_sig) == float:
        flare_sig = np.array([flare_sig])
        flare_mean_cr = np.array([flare_mean_cr])
        flare_t = np.array([flare_t])
    #N_g is the number of counts in the flare. I consider the mean cr to be the mean cr within the duration
    #of a flare (defined as +/- 2 sigma). I thus need to correct that to get the total number of counts
    flare_Amp = mcrtoA*flare_mean_cr
#    print('flare_t', flare_t, 'T', T, 'flare_sig', flare_sig)
#    print(flare_sig, np.asarray(flare_sig))
    size = len(flare_sig)
    N_g = np.zeros(size)
    if flare_Amp.any() > 0:
        for i in range(size):
            if flare_Amp[i] == 0:
                N_g[i] = 0
            else:
                if flare_t[i] < 0:
                    N_g[i] = int(integrate.quad(lambda x:  Gau(x, flare_Amp[i], flare_sig[i], flare_t[i]), 0, alpha*flare_sig[i])[0])
                if flare_t[i] > T:
                    N_g[i] = int(integrate.quad(lambda x:  Gau(x, flare_Amp[i], flare_sig[i], flare_t[i]), flare_t[i] - alpha*flare_sig[i], T)[0])
                else:
                    if flare_t[i] > alpha*flare_sig[i]:
                        if flare_t[i] < (T - alpha*flare_sig[i]):
                            N_g[i] = int(integrate.quad(lambda x:  Gau(x, flare_Amp[i], flare_sig[i], flare_t[i]), flare_t[i] - alpha*flare_sig[i], flare_t[i] + alpha*flare_sig[i])[0])
                        else:
                            N_g[i] = int(integrate.quad(lambda x:  Gau(x, flare_Amp[i], flare_sig[i], flare_t[i]), flare_t[i] - alpha*flare_sig[i], T)[0])
                    else:
                        N_g[i] = int(integrate.quad(lambda x:  Gau(x, flare_Amp[i], flare_sig[i], flare_t[i]), 0, flare_t[i] + alpha*flare_sig[i])[0])    
    cdf_c_temp = CDF_c(t, T)
    
    #These CDF functions always increase with t, therefore their highest value is their last one. To normalize
    #them I just need to divide by the last one
    cdf_c = (N_c/(N_g.sum() + N_c))*cdf_c_temp/(cdf_c_temp[-1])
    if N_g.sum() == 0:
        cdf_tot = cdf_c
#        print('No counts in the flare for this ObsID')
    else:
        cdf_g_temp = np.zeros(size*len(t)).reshape(size, len(t))
        cdf_g = np.zeros(size*len(t)).reshape(size, len(t))
        for i in range(size):
            if ~(N_g[i] == 0):
                cdf_g_temp[i] = CDF_g(t, flare_Amp[i], flare_sig[i], flare_t[i], N_g[i])
                cdf_g[i] = (N_g[i]/(N_g.sum() + N_c))*cdf_g_temp[i]/(cdf_g_temp[i][-1])
        cdf_tot = cdf_c + np.sum(cdf_g, axis = 0)
        if plot:
            plt.plot(t,cdf_tot, label='Constant + Gaussian(s)')
            plt.plot(t,cdf_c, label='Constant')
            for i in range(size):
                plt.plot(t,cdf_g[i], label='Gaussian {}'.format(i))
            plt.xlim([0,T])
            plt.ylim([0,1])
            plt.ylabel('CDF')
            plt.xlabel('time(s)')
            plt.legend()
            plt.show()

#    print(N_g, N_c)
    M = int(np.random.poisson((N_g.sum() + N_c),1))
    y = np.random.rand(M)
    y = np.sort(y)
    
    time_evt = t[np.searchsorted(cdf_tot, y)]
    
    ncp_prior = AlogN(time_evt.size, *(popt+3*pstd))
    info = ModifiedNCP_PRIORxbblocks.bsttbblock(time_evt, np.array([0]), np.array([T]),ncp_prior,nbootstrap=nbootstrap)
    return time_evt, info


def getratepu(rate, frame_time, alpha):
    '''
    Returns the pile-up corrected count rates in ct/s (i.e., returns the detected cr)
    Parameters
    ----------
    rate:  array of floats
        Array of count rates (ct/s)
    
    frame_time     : float
        Frame time (s)
        
    alpha: float
        Pile-up parameter alpha 
    
    Returns
    ----------
    ratecorr : array of floats
        Corrected count rates (ct/s)
    '''
    rate_frame = rate*frame_time
    numerator =  ( np.exp(alpha*rate_frame)-1 ) * np.exp(-1.0*rate_frame)
    denom = alpha * rate_frame
    fraction_lost =  1 - numerator/denom
    return (1-fraction_lost)*rate

def get_flare_bb_nobsnopcr(ledges, redges, counts, widths, rates, incident_cr, observed_cr, amplitude_criteria = 3, minflu = 8):
    '''
    Outputs Flare parameters for a given set of parameters from a Bayesian Blocks analysis without bootstrap. For subarray mode (Not gratings!)
    Parameters
    ----------
    ledges: array of floats
        Time of the beginning of each block (s)
        
    redges: array of floats
        Time of the end of each block (s)
        
    counts: array of int
         Total counts in each block 
         
    widths: array of floats
         Total lenght of each block (s)
         
    rates: array of floats
        Mean count rate of each block (ct/s)
        
    incident_cr: array of floats
        Incident (unpiled) count rates (ct/s). From calibration

    observed_cr: array of floats
        detected (piled) count rates (ct/s). From calibration

    amplitude_criteria: float:
        Sigma range above quiesence for a block to be considered a flare
        Default value is 3
        
    minflu: int
        minimum number of counts in a block. If lower, it is combined with a nearby block
        
    Returns
    ----------
    data:  array of floats
        Contains, in order, the time of the beginning of each block (s), the time of the end of each block (s),
        the number of counts in each block, the total lenght of 
        each block (s), the mean count rate of each block (ct/s), the standard deviation in count rate
        in each block (ct/s) and the Poisson error in each block count rate (ct/s)
    block:  array of floats
        Same as data but for pile-up corrected values and for each FLARE instead of each block (flares can be made of multiple blocks)        
    LoRate : array of floats
        Contains the mean count rate of the longest block and its standard deviation
    '''    
    #Note how many blocks there are in the obsid
    num_blocks = np.size(redges)

    l = 0     #counts the number flaring blocks (the total, not the number of individual flares)
    
    counts = np.asarray(counts)
    widths = np.asarray(widths)
    rateserr = np.sqrt(counts)/widths
    block = None
    #If there are more than 1 block, then the ones significantly above the lowest one are flares
    #EXCEPT IF THEIR FLUENCE IS LESS THAN 8 COUNTS
    del_blocks = np.array([])
    if num_blocks > 1:    
        if (counts < 8).any():
            lowfluence = np.where(counts < 8)[0]
            #print('low counts:',counts[lowfluence])
            for i in lowfluence:
                if i == 0:
                    counts[i+1] = counts[i+1] + counts[i]
                    ledges[i+1] = ledges[i]
                    widths[i+1] = widths[i+1] + widths[i]
                    rates[i+1] = counts[i+1]/widths[i+1]
                    rateserr[i+1] = np.sqrt(counts[i+1])/widths[i+1]
                    del_blocks = np.append(del_blocks,i)
                else:
                    counts[i-1] = counts[i-1] + counts[i] 
                    redges[i-1] = redges[i]
                    widths[i-1] = widths[i-1] + widths[i]
                    rates[i-1] = counts[i-1]/widths[i-1]
                    rateserr[i-1] = np.sqrt(counts[i-1])/widths[i-1]
                    del_blocks = np.append(del_blocks,i)
                    
            counts = np.delete(counts,del_blocks.astype(int))
            ledges = np.delete(ledges,del_blocks.astype(int))
            redges = np.delete(redges,del_blocks.astype(int))
            widths = np.delete(widths,del_blocks.astype(int))
            rates = np.delete(rates,del_blocks.astype(int))
            rateserr = np.delete(rateserr,del_blocks.astype(int)) 
        #Note how many blocks there are in the obsid
        num_blocks = np.size(redges)
        quies_id = np.argmax(widths)             
        flares_id = (rates-amplitude_criteria*rateserr)>(rates[quies_id] + amplitude_criteria*rateserr[quies_id])
        flares = ma.array(rates, mask = ~flares_id)
        data = np.ones(6).reshape(1,6)
        for h in range(num_blocks):
            data = np.concatenate((data, np.array([ledges[h], redges[h], counts[h], widths[h], rates[h], rateserr[h]]).reshape(1,6)))
            #print(ledges[h], redges[h], peakcr[h])
        data = np.delete(data, (0), axis = 0)
        LoRate = np.array([rates[quies_id], rateserr[quies_id]]).reshape(1,2)

        #UNPILE EACH FLARE BLOCK RIGHT NOW INSTEAD OF DOING IT AFTER COMBINING MULTIPLE BLOCKS!!!
        for p in range(num_blocks):
            if flares_id[p]:
                #print('block',p,'rate before pile-up corr ',rates[p], 'counts ', counts[p], 'rateserr ', rateserr[p])
                rates[p] = incident_cr[np.argmin(np.abs(rates[p] - observed_cr))]
                counts[p] = int(rates[p]*widths[p])
                rateserr[p] = np.sqrt(counts[p])/widths[p]
                #print('block',p,'rate after pile-up corr ',rates[p], 'counts ', counts[p], 'rateserr ', rateserr[p])

        
        #If the obsid has only one flare
        if np.size(flares[~flares.mask]) == 1:
            indice = np.where(flares_id == True)[0][0]
            if l == 0:
                block = np.array([ledges[indice], redges[indice], counts[indice], widths[indice], 
                                                  rates[indice], rateserr[indice]]).reshape(1,6)
                l = l + 1
            else:
                block = np.concatenate((block, np.array([ledges[indice], redges[indice], counts[indice],
                                                  widths[indice], rates[indice], rateserr[indice]]).reshape(1,6)), axis = 0) 
        #If there are multiple blocks significantly above quiescence, then we need to figure out how many
        #flares there are
        else:
            k = 0        #Used to spot the first flare of the obsid
            j = 0        #Used to move through the blocks
            while j < num_blocks:
                if ~flares_id[j]:
                    j = j + 1
                    continue 
                else:
                    if k == 0:                  
                        k = k + 1
                        if (j < num_blocks - 1):
                            if ~flares_id[j + 1]:
                             #if the next block isnt a flare, then this flare is made of only one block
                                if l == 0: 
                                    block = np.array([ledges[j], redges[j], counts[j], widths[j], 
                                                  rates[j], rateserr[j]]).reshape(1,6)
                                    l = l + 1
                                    j = j + 1
                                else:
                                    block = np.concatenate((block, np.array([ledges[j], redges[j], counts[j], widths[j], 
                                                  rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                    j = j + 1
                            else:
                           #But if the next block is also a flare then add the blocks until they end
                                flare_block = np.ones(6).reshape(1,6)
                                while(flares_id[j] and j < (num_blocks - 1)):
                                    flare_block = np.concatenate((flare_block, np.array([ledges[j], 
                                                     redges[j], counts[j], widths[j], 
                                                     rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                    j = j + 1
                                    
                                if flares_id[j] and j == (num_blocks - 1):
                                    flare_block = np.concatenate((flare_block, np.array([ledges[j], 
                                                     redges[j], counts[j], widths[j], 
                                                     rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                #Delete the first row that was use to initiate the array
                                flare_block = np.delete(flare_block, (0), axis = 0)
                                
                                #Finalize the block
                                if l == 0:
                                    l = l + 1
                                    block = np.array([flare_block[0,0],flare_block[-1,1],
                                                  np.sum(flare_block[:,2]), np.sum(flare_block[:,3]),
                                                  np.sum(flare_block[:,2])/float(np.sum(flare_block[:,3])),
                                                  math.sqrt(np.sum(flare_block[:,2]))/float(np.sum(flare_block[:,3]))]).reshape(1,6)
                                else:
                                    block = np.concatenate((block, np.array([flare_block[0,0],flare_block[-1,1],
                                                  np.sum(flare_block[:,2]), np.sum(flare_block[:,3]),
                                                  np.sum(flare_block[:,2])/float(np.sum(flare_block[:,3])), math.sqrt(np.sum(flare_block[:,2]))/
                                                     float(np.sum(flare_block[:,3]))]).reshape(1,6)), axis = 0)
                        else:
                            #If this is the last block, then this flare is also made up of only one block
                            if l == 0:
                                l = l + 1
                                block = np.array([ledges[j], redges[j], counts[j], widths[j], rates[j],
                                                        rateserr[j]]).reshape(1,6)
                            else:
                                block = np.concatenante((block,np.array([ledges[j], redges[j], counts[j], widths[j], rates[j],
                                                        rateserr[j]]).reshape(1,6)), axis = 0)
                            j = j + 1
                    
                    #If this isnt the first flare...
                    else:
                        if j < (num_blocks - 1):
                            #If this isnt the last block...
                            if ~flares_id[j + 1]:
                                #and if the next block isnt a flare, then this flare is made of only one block
                                block = np.concatenate((block, np.array([ledges[j], redges[j], counts[j], widths[j], 
                                                        rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                j = j + 1
                            else:
                                #If the previous block wasnt a flare and this one  and the next are then 
                                #add the blocks until they end
                                flare_block = np.ones(6).reshape(1,6)
                                while(flares_id[j] and j < (num_blocks - 1)):
                                    flare_block = np.concatenate((flare_block, np.array([ledges[j], 
                                                     redges[j], counts[j], widths[j], 
                                                     rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                    j = j + 1
                                    
                                if flares_id[j] and j == (num_blocks - 1):
                                    flare_block = np.concatenate((flare_block, np.array([ledges[j], 
                                                     redges[j], counts[j], widths[j], 
                                                     rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                #Delete the first row that was use to initiate the array
                                flare_block = np.delete(flare_block, (0), axis = 0)
                                
                                #Finalize the block
                                block = np.concatenate((block,np.array([flare_block[0,0],flare_block[-1,1],
                                                  np.sum(flare_block[:,2]), np.sum(flare_block[:,3]),
                                                  np.sum(flare_block[:,2])/float(np.sum(flare_block[:,3])),math.sqrt(np.sum(flare_block[:,2]))/
                                                  float(np.sum(flare_block[:,3]))]).reshape(1,6)), axis = 0)
                        else:
                            if ~flares_id[j-1]:
                                #If this is the last block, then this flare is also made up of only one block
                                block = np.concatenate((block, np.array([ledges[j], redges[j], counts[j], widths[j], rates[j],
                                                        rateserr[j]]).reshape(1,6)), axis = 0)
                            j = j + 1
    else:
        data = np.array([ledges, redges, counts, widths, rates, rateserr]).reshape(1,6)
        LoRate = np.array([rates, rateserr]).reshape(1,2)
    
    #Make sure the arrays are sorted in time
    if block is not None:
        block = block[np.argsort(block[:,0])]
    data = data[np.argsort(data[:,0])]
    return data, block, LoRate


def XVPget_flare_bb_nobsnopcr(ledges, redges, counts, widths, rates, Q_ratio,new_ratio,piled_cr,incident_cr,observed_cr, amplitude_criteria = 3, minflu = 8):
    '''
   
    Outputs Flare parameters for a given set of parameters from a Bayesian Blocks analysis without bootstrap. For gratings!
    Parameters
    ----------
    ledges: array of floats
        Time of the beginning of each block (s)
        
    redges: array of floats
        Time of the end of each block (s)
        
    counts: array of int
         Total counts in each block 
         
    widths: array of floats
         Total lenght of each block (s)
         
    rates: array of floats
        Mean count rate of each block (ct/s)

    Q_ratio: array of floats
 	0th/tot quiescence cr

    new_ratio: array of floats
	how the ratio of 1st/0th order counts in flares changes with cr due to pile-up

    piled_cr: array of floats
	From the calibration, piled-up cr each corresponding to a new ratio

    incident_cr: array of floats
	Unpiled 0th order flare count rates (from calibration)

    observed_cr: array of floats
	Piled 0th order flare count rates (from calibration)        

    amplitude_criteria: float:
        Sigma range above quiesence for a block to be considered a flare
        Default value is 3
        
    minflu: int
        minimum number of counts in a block. If lower, it is combined with a nearby block
        
    Returns
    ----------
    data:  array of floats
        Contains, in order, the time of the beginning of each block (s), the time of the end of each block (s),
        the number of counts in each block, the total lenght of 
        each block (s), the mean count rate of each block (ct/s), the standard deviation in count rate
        in each block (ct/s) and the Poisson error in each block count rate (ct/s)
    block:  array of floats
        Same as data, but for each FLARE instead of each block (flares can be made of multiple blocks)        
    LoRate : array of floats
        Contains the mean count rate of the longest block and its standard deviation
    '''    
    #Note how many blocks there are in the obsid
    num_blocks = np.size(redges)

    l = 0     #counts the number flaring blocks (the total, not the number of individual flares)
    
    counts = np.asarray(counts)
    widths = np.asarray(widths)
    rateserr = np.sqrt(counts)/widths
    block = None
    #If there are more than 1 block, then the ones significantly above the lowest one are flares
    #EXCEPT IF THEIR FLUENCE IS LESS THAN 8 COUNTS
    del_blocks = np.array([])
    if num_blocks > 1:    
        if (counts < 8).any():
            lowfluence = np.where(counts < 8)[0]
            print('low counts:',counts[lowfluence])
            for i in lowfluence:
                if i == 0:
                    counts[i+1] = counts[i+1] + counts[i]
                    ledges[i+1] = ledges[i]
                    widths[i+1] = widths[i+1] + widths[i]
                    rates[i+1] = counts[i+1]/widths[i+1]
                    rateserr[i+1] = np.sqrt(counts[i+1])/widths[i+1]
                    del_blocks = np.append(del_blocks,i)
                else:
                    counts[i-1] = counts[i-1] + counts[i] 
                    redges[i-1] = redges[i]
                    widths[i-1] = widths[i-1] + widths[i]
                    rates[i-1] = counts[i-1]/widths[i-1]
                    rateserr[i-1] = np.sqrt(counts[i-1])/widths[i-1]
                    del_blocks = np.append(del_blocks,i)
                    
            counts = np.delete(counts,del_blocks.astype(int))
            ledges = np.delete(ledges,del_blocks.astype(int))
            redges = np.delete(redges,del_blocks.astype(int))
            widths = np.delete(widths,del_blocks.astype(int))
            rates = np.delete(rates,del_blocks.astype(int))
            rateserr = np.delete(rateserr,del_blocks.astype(int)) 
        #Note how many blocks there are in the obsid
        num_blocks = np.size(redges)
        quies_id = np.argmax(widths)             
        flares_id = (rates-amplitude_criteria*rateserr)>(rates[quies_id] + amplitude_criteria*rateserr[quies_id])
        flares = ma.array(rates, mask = ~flares_id)
        data = np.ones(6).reshape(1,6)
        for h in range(num_blocks):
            data = np.concatenate((data, np.array([ledges[h], redges[h], counts[h], widths[h], rates[h], rateserr[h]]).reshape(1,6)))
            #print(ledges[h], redges[h], peakcr[h])
        data = np.delete(data, (0), axis = 0)
        LoRate = np.array([rates[quies_id], rateserr[quies_id]]).reshape(1,2)
            
        #Unpile each block right now instead of doing it after combining multiple blocks!!!
        for p in range(num_blocks):
            if flares_id[p]:    
                #This function gives only the "flare" pile-up corrected cr so we need to add the quiescence back
                rates[p] = XVPget_flare_unpiled(rates[p],rates[quies_id],Q_ratio,new_ratio,piled_cr,incident_cr,observed_cr) + rates[quies_id]
                counts[p] = int(rates[p]*widths[p])
                rateserr[p] = np.sqrt(counts[p])/widths[p]
        
        #If the obsid has only one flare
        if np.size(flares[~flares.mask]) == 1:
            indice = np.where(flares_id == True)[0][0]
            if l == 0:
                block = np.array([ledges[indice], redges[indice], counts[indice], widths[indice], 
                                                  rates[indice], rateserr[indice]]).reshape(1,6)
                l = l + 1
            else:
                block = np.concatenate((block, np.array([ledges[indice], redges[indice], counts[indice],
                                                  widths[indice], rates[indice], rateserr[indice]]).reshape(1,6)), axis = 0) 
        #If there are multiple blocks significantly above quiescence, then we need to figure out how many
        #flares there are
        else:
            k = 0        #Used to spot the first flare of the obsid
            j = 0        #Used to move through the blocks
            while j < num_blocks:
                if ~flares_id[j]:
                    j = j + 1
                    continue 
                else:
                    if k == 0:                  
                        k = k + 1
                        if (j < num_blocks - 1):
                            if ~flares_id[j + 1]:
                             #if the next block isnt a flare, then this flare is made of only one block
                                if l == 0: 
                                    block = np.array([ledges[j], redges[j], counts[j], widths[j], 
                                                  rates[j], rateserr[j]]).reshape(1,6)
                                    l = l + 1
                                    j = j + 1
                                else:
                                    block = np.concatenate((block, np.array([ledges[j], redges[j], counts[j], widths[j], 
                                                  rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                    j = j + 1
                            else:
                           #But if the next block is also a flare then add the blocks until they end
                                flare_block = np.ones(6).reshape(1,6)
                                while(flares_id[j] and j < (num_blocks - 1)):
                                    flare_block = np.concatenate((flare_block, np.array([ledges[j], 
                                                     redges[j], counts[j], widths[j], 
                                                     rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                    j = j + 1
                                
                                if flares_id[j] and j == (num_blocks - 1):
                                    flare_block = np.concatenate((flare_block, np.array([ledges[j], 
                                                     redges[j], counts[j], widths[j], 
                                                     rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                
                                #Delete the first row that was use to initiate the array
                                flare_block = np.delete(flare_block, (0), axis = 0)
                                
                                #Finalize the block
                                if l == 0:
                                    l = l + 1
                                    block = np.array([flare_block[0,0],flare_block[-1,1],
                                                  np.sum(flare_block[:,2]), np.sum(flare_block[:,3]),
                                                  np.sum(flare_block[:,2])/float(np.sum(flare_block[:,3])),
                                                  math.sqrt(np.sum(flare_block[:,2]))/float(np.sum(flare_block[:,3]))]).reshape(1,6)
                                else:
                                    block = np.concatenate((block, np.array([flare_block[0,0],flare_block[-1,1],
                                                  np.sum(flare_block[:,2]), np.sum(flare_block[:,3]),
                                                  np.sum(flare_block[:,2])/float(np.sum(flare_block[:,3])), math.sqrt(np.sum(flare_block[:,2]))/
                                                     float(np.sum(flare_block[:,3]))]).reshape(1,6)), axis = 0)
                        else:
                            #If this is the last block, then this flare is also made up of only one block
                            if l == 0:
                                l = l + 1
                                block = np.array([ledges[j], redges[j], counts[j], widths[j], rates[j],
                                                        rateserr[j]]).reshape(1,6)
                            else:
                                block = np.concatenante((block,np.array([ledges[j], redges[j], counts[j], widths[j], rates[j],
                                                        rateserr[j]]).reshape(1,6)), axis = 0)
                            j = j + 1
                    
                    #If this isnt the first flare...
                    else:
                        if j < (num_blocks - 1):
                            #If this isnt the last block...
                            if ~flares_id[j + 1]:
                                #and if the next block isnt a flare, then this flare is made of only one block
                                block = np.concatenate((block, np.array([ledges[j], redges[j], counts[j], widths[j], 
                                                        rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                j = j + 1
                            else:
                                #If the previous block wasnt a flare and this one  and the next are then 
                                #add the blocks until they end
                                flare_block = np.ones(6).reshape(1,6)
                                while(flares_id[j] and j < (num_blocks - 1)):
                                    flare_block = np.concatenate((flare_block, np.array([ledges[j], 
                                                     redges[j], counts[j], widths[j], 
                                                     rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                    j = j + 1
                                    
                                if flares_id[j] and j == (num_blocks - 1):
                                    flare_block = np.concatenate((flare_block, np.array([ledges[j], 
                                                     redges[j], counts[j], widths[j], 
                                                     rates[j], rateserr[j]]).reshape(1,6)), axis = 0)
                                #Delete the first row that was use to initiate the array
                                flare_block = np.delete(flare_block, (0), axis = 0)
                                
                                #Finalize the block
                                block = np.concatenate((block,np.array([flare_block[0,0],flare_block[-1,1],
                                                  np.sum(flare_block[:,2]), np.sum(flare_block[:,3]),
                                                  np.sum(flare_block[:,2])/float(np.sum(flare_block[:,3])),math.sqrt(np.sum(flare_block[:,2]))/
                                                  float(np.sum(flare_block[:,3]))]).reshape(1,6)), axis = 0)
                        else:
                            if ~flares_id[j-1]:
                                #If this is the last block, then this flare is also made up of only one block
                                block = np.concatenate((block, np.array([ledges[j], redges[j], counts[j], widths[j], rates[j],
                                                        rateserr[j]]).reshape(1,6)), axis = 0)
                            j = j + 1
    else:
        data = np.array([ledges, redges, counts, widths, rates, rateserr]).reshape(1,6)
        LoRate = np.array([rates, rateserr]).reshape(1,2)
    
    #Make sure the arrays are sorted in time
    if block is not None:
        block = block[np.argsort(block[:,0])]
    data = data[np.argsort(data[:,0])]
    return data, block, LoRate


def XVPget_flare_piled(test_cr,Q,f2z,Q_ratio,frame_time,alpha):
    '''
    Given an incoming flare cr (test_cr) with the unpiled f2z 1st/0th order flare cr ratio and a quiescence cr Q
    with a 0th/1st order Q ratio of Q_ratio, outputs the detected cr piled_flare_cr and the detected
    1st/0th flare cr ratio
    '''
    flare_0th = test_cr/(1+f2z)
    flare_1st = test_cr - flare_0th
    tot_0th = Q_ratio*Q + flare_0th
    flare_0th_piled = getratepu(tot_0th, frame_time, alpha = alpha) - Q_ratio*Q
    piled_flare_cr = flare_0th_piled + flare_1st
    ratio = flare_1st/flare_0th_piled
    return piled_flare_cr,ratio

def XVPget_flare_unpiled(det_piled_cr,det_Q,Q_ratio,new_ratio,piled_cr,incident_cr,observed_cr):
    '''
    Given a quiescence (det_Q) and a flare cr (det_piled_cr), it finds the un-piled cr of said flare using the pile-up calibration
    based on Q_ratio and the 0th/1st order ratio in flares from my data reduction of XVP data

    Q_ratio: 0th/tot quiescence cr

    new_ratio: how the ratio of 1st/0th order counts in flares changes with cr due to pile-up

    piled_cr: From the calibration, piled-up cr each corresponding to a new ratio

    det_piled_cr: Detected cr (not quie sub)

    det_Q: Detected quiescence

    incident_cr: Unpiled 0th order flare count rates (from calibration)

    observed_cr: Piled 0th order flare count rates (from calibration)
    
    outputs:
    --------------------------------
    unpiled_flare_cr: The unpiled quies sub flare cr
    '''
    new_ratio_true = new_ratio[np.argmin(np.abs(det_piled_cr - det_Q - piled_cr))]
    flare_cr_Q_sub = det_piled_cr - det_Q
    flare_0th = flare_cr_Q_sub/(1+new_ratio_true)
    flare_1st = flare_cr_Q_sub - flare_0th
    tot_0th = Q_ratio*det_Q + flare_0th
    flare_0th_unpiled = incident_cr[np.argmin(np.abs(tot_0th-observed_cr))] - Q_ratio*det_Q
    unpiled_flare_cr = flare_0th_unpiled + flare_1st
    return unpiled_flare_cr


def AlogN(N, A, B):
    '''
    Outputs a logarithmic function of N
    Parameters
    ----------
    N:  array of floats
        N values of the x variable
    
    A     : float
        Amplitude of the function
        
    B    : float
        Origin value of the function
    Returns
    ----------
    A*log(N)+B: array of floats
    '''
    return A*np.log(N) + B
