"""
Here we open transmission measurements saved as xarray.DataSet.
The transmission spectra were taken as a function of power dissipated
in 2 heaters. The frequency axis is already calibrated.

We search for the pair of heater powers that yields the most
symmetric triplets. The criteria are the spectral distance between 
the resonances of a triplet and the extinction ratio of the outer
resonances.

"""
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.signal import sosfilt
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
import pyLPD.MLtools as mlt

########### Load DataSet ############
file_dir = 'G:\\Meu Drive\\LPD Team\\Experiments\\DOPO\\Data\\R-R-R_molecule 700-550\\'
file_name = '22-02-04_18.40_chip1_R-R-R_wg-ring gap 700 nm_ring-ring gap 550 nm_TE_through port_heaterMap_10dBm'
data = xr.open_dataset(file_dir+'Calibrated data\\'+file_name+'calibrated.nc')
# data.Sample.values = range(len(data.Sample.values))
print(data.h2_power.values)
# h1 = 4
# h2 = 2
# h3 = 4
# plt.plot(data['cavity_transmission'].values[h1, h2, h3])
# plt.show()

def sin_func(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def sin_enveloped(x, a, b, c, d, e):
    return (a*x + e) * np.sin(b * x + c) + d

def cav_norm(data, envPeak_delta = None, envPeak_sg = None,
             envPeak_smooth = None, plot_steps_bool = None):
    """Normalizes cav transmission using upper background envelope. It also smooths the cav signal using a savitz golay filter.

    Args:
        data (dataframe): raw data acquired.
        envPeak_delta (float, optional): envPeak parameter.
        envPeak_sg (int, optional): envPeak parameter.
        savitz_window (int, optional): savitz golay parameter.
        savitz_order (int, optional): savitz golay parameter.
    """
    #setting default values if not given
    if envPeak_delta is None: envPeak_delta = 0.0075
    if envPeak_sg is None: envPeak_sg = 1
    if envPeak_smooth is None: envPeak_smooth = 0.0008
    if plot_steps_bool is None: plot_steps_bool=False

    #actual function
    ylower_cav, yupper_cav = mlt.envPeak(data, delta=envPeak_delta, smooth=envPeak_smooth, sg_order=envPeak_sg)# Finding lower and upper envelope
    data_norm = data/yupper_cav # Normalizing data
    
    if plot_steps_bool:
        plt.figure(figsize=(12,3))
        plt.plot(data)
        plt.plot(yupper_cav)
        plt.show()

    return data_norm

# ########### Choose triplet (wavelength) ############
# wavlen_cent = 1555.8  # nm
# # wavlen_cent = 1552.2 # nm
# c = constants.c
# freq_cent = c*1e-3/wavlen_cent  # THz
# freq_span = 0.3 # THz
# freq_i = freq_cent - freq_span/2
# freq_f = freq_cent + freq_span/2
# freq_mask_i = data.frequency.values[h1, h3, data['frequency'].values[h1, h3] < freq_f]
# freq_mask = freq_mask_i[freq_mask_i > freq_i]
# cav_mask = data.cavity_transmission.values[h1, h3, data['frequency'].values[h1, h3] < freq_f]
# cav_mask = cav_mask[freq_mask_i > freq_i]
# ind_min,_ = find_peaks(-cav_mask, height=-0.018, prominence=0.002, distance=2000)

# lambda_vec = c/freq_mask
# plt.plot(lambda_vec, cav_mask)
# plt.scatter(lambda_vec[ind_min], cav_mask[ind_min], c='r')
# plt.plot(-cav_mask)
# plt.plot(ind_min, -cav_mask[ind_min], '*', c='r')
# plt.plot(freq_mask, cav_mask)
# plt.scatter(freq_mask[ind_min], cav_mask[ind_min], c='r')
# plt.show()

# datanorm = cav_norm(cav_mask, plot_steps_bool=True, envPeak_delta=0.0022, envPeak_smooth=0.002)
# plt.plot(datanorm)
# plt.show()

########### Choose triplet (index) ############

freq_cent = 78000  # number of points
freq_span = 6000 # number of points
freq_i = freq_cent - freq_span
freq_f = freq_cent + freq_span

# cav_mask = data.cavity_transmission[h1, h2, h3, freq_i:freq_f].values

# ind_min,_ = find_peaks(-cav_mask, height=-0.9, prominence=0.1, distance=800)

# plt.plot(-cav_mask)
# plt.plot(ind_min, -cav_mask[ind_min], '*', c='r')
# plt.show()

# ########### Plot loop spectra (cropped) ############
# splitting_vec = np.zeros([len(data.h1_power.values), len(data.h3_power.values)])
# symmetry_vec = splitting_vec
# obj_func_vec = splitting_vec
# for h1, p1 in enumerate(data.h1_power.values):
#     for h3, p3 in enumerate(data.h3_power.values):
#         ############ Choose triplet ############
#         freq_mask_i = data.frequency.values[h1, h3, data['frequency'].values[h1, h3] < freq_f]
#         freq_mask = freq_mask_i[freq_mask_i > freq_i]
#         cav_mask = data.cavity_transmission.values[h1, h3, data['frequency'].values[h1, h3] < freq_f]
#         cav_mask = cav_mask[freq_mask_i > freq_i]
#         # cav_mask = cav_norm(cav_mask, plot_steps_bool=False, envPeak_delta=0.003, envPeak_smooth=0.003)
#         # cav_mask = cav_mask/sin_func(freq_mask, 0.01, 119, -24, 0.205)
#         ind_min,_ = find_peaks(-cav_mask, height=-0.014, prominence=0.002, distance=800)
#         plt.plot(freq_mask, cav_mask)
#         plt.title('h1 = %i, h3 = %i' %(h1, h3))
#         plt.scatter(freq_mask[ind_min], cav_mask[ind_min], c='r')
#         plt.show()

#         splitting_pair = abs(np.diff(freq_mask[ind_min]))
#         try:
#             splitting_vec[h1, h3] = abs(splitting_pair[1] - splitting_pair[0])
#             symmetry_vec[h1, h3] = abs(cav_mask[ind_min[2]] - cav_mask[ind_min[0]])
#         except:
#             splitting_vec[h1, h3] = np.NaN
#             symmetry_vec[h1, h3] = np.NaN

# ########### Objective function ############
# picture_dir = file_dir+'Processed data\\'
# obj_func_vec = splitting_vec**2 + symmetry_vec**2
# plt.pcolormesh(data['h3_power'].values, data['h1_power'].values, splitting_vec, shading='auto', vmin=0)
# plt.xlabel('H3 (mW)')
# plt.ylabel('H1 (mW)')
# plt.colorbar()
# plt.savefig(picture_dir+file_name+'_splittingDiff_no_norm.png')
# plt.show()
# plt.pcolormesh(data['h3_power'].values, data['h1_power'].values, symmetry_vec, shading='auto', vmin=0)
# plt.xlabel('H3 (mW)')
# plt.ylabel('H1 (mW)')
# plt.colorbar()
# plt.savefig(picture_dir+file_name+'_extinctionDiff_no_norm.png')
# plt.show()
# plt.pcolormesh(data['h3_power'].values, data['h1_power'].values, obj_func_vec, shading='auto', vmin=0)
# plt.xlabel('H3 (mW)')
# plt.ylabel('H1 (mW)')
# plt.colorbar()
# plt.savefig(picture_dir+file_name+'_opt_no_norm.png')
# plt.show()

########### Plot loop spectra (cropped) ############
splitting_vec = np.zeros([len(data.h1_power.values), len(data.h2_power.values), len(data.h3_power.values)])
symmetry_vec = splitting_vec
# obj_func_vec = splitting_vec
for h1, p1 in enumerate(data.h1_power.values):
    for h2, p2 in enumerate(data.h2_power.values):
        for h3, p3 in enumerate(data.h3_power.values):
            cav_mask = data.cavity_transmission[h1, h2, h3, freq_i:freq_f].values
            ind_min,_ = find_peaks(-cav_mask, height=-0.95, prominence=0.11, distance=600)
            
            plt.plot(cav_mask)
            plt.title('h1 = %i, h2 = %i, h3 = %i' %(h1, h2, h3))
            plt.scatter(ind_min, cav_mask[ind_min], c='r')
            plt.show()

            try:
                splitting_vec[h1, h2, h3] = abs(ind_min[2] - ind_min[1] - (ind_min[1] - ind_min[0]))
                splitting_vec[h1, h2, h3] = abs(cav_mask[ind_min[2]] - cav_mask[ind_min[0]])
            except:
                splitting_vec[h1, h2, h3] = np.NaN
                splitting_vec[h1, h2, h3] = np.NaN

picture_dir = file_dir+'Processed data\\'
for h1, p1 in enumerate(data.h1_power.values):
    plt.pcolormesh(data['h3_power'].values, data['h2_power'].values, splitting_vec[h1, :, :], shading='auto', vmin=0)
    plt.title('h1 = %i' %h1)
    plt.xlabel('H3 (mW)')
    plt.ylabel('H2 (mW)')
    plt.colorbar()
    plt.savefig(picture_dir+file_name+'_splittingDiff_no_norm.png')
    plt.show()
    plt.pcolormesh(data['h3_power'].values, data['h2_power'].values, symmetry_vec[h1, :, :], shading='auto', vmin=0)
    plt.title('h1 = %i' %h1)
    plt.xlabel('H3 (mW)')
    plt.ylabel('H2 (mW)')
    plt.colorbar()
    plt.savefig(picture_dir+file_name+'_extinctionDiff_no_norm.png')
    plt.show()

########## Plot loop spectra (cropped) ############
# h2 = 4
# h3 = 4
# for h1, p1 in enumerate(data.h1_power.values):
#     cav_mask = data.cavity_transmission[h1, h2, h3, freq_i:freq_f].values
#     ind_min,_ = find_peaks(-cav_mask, height=-0.95, prominence=0.11, distance=600)
            
#     plt.plot(cav_mask)
#     plt.title('h1 = %i, h2 = %i, h3 = %i' %(h1, h2, h3))
#     plt.scatter(ind_min, cav_mask[ind_min], c='r')
#     plt.show()

########### Load pickle dictionary ############
# data_raw = pd.read_pickle(file_dir+file_name+'.pkl')
# print(data_raw)

############ Test clean noise ############
# h1 = 0
# h3 = 1

# params, params_covariance = curve_fit(sin_func, freq_mask[28000:], cav_mask[28000:])
# print(params)
# # datafft = np.fft.fft(cav_mask)
# plt.plot(freq_mask[28000:], cav_mask[28000:])
# # plt.plot(freq_mask, sin_func(freq_mask, 0.01, 119, -24, 0.205))
# plt.plot(freq_mask[28000:], sin_func(freq_mask[28000:], *params))
# # plt.plot(freq_mask, cav_mask/sin_func(freq_mask, 0.01, 119, -24, 0.205))
# plt.show()
# plt.semilogy(abs(datafft)**2)
# plt.show()

# sos = butter(1, btype='lowpass', Wn = 1/2000, output='sos')
# sos = butter(1, btype='bandpass', Wn = (1/2000, 1/600), output='sos')
# data_pf  = sosfilt(sos, cav_mask)

# plt.plot(cav_mask[:-30000])
# plt.plot(cav_mask[28000:])
# # plt.plot(data_pf)
# plt.show()

# print(len(cav_mask))