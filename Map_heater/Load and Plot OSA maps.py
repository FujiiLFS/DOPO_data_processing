import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np


file_dir = 'G:\\Meu Drive\\LPD Team\\Experiments\\DOPO\\Data\\R-R-R_molecule 600-550\\'
file_name = '22-01-13_chip1_R-R-R_wg-ring gap 600 nm_ring-ring gap 550 nm_TE_drop port_heaterMap_20dBm.nc'
data_raw = xr.open_dataarray(file_dir+file_name)
# print(data_raw.isel(h1_current=2, h3_current=4))
# print(data_raw)
# print(len(data_raw['Power_OSA']))
# plt.plot(data_raw['Power_OSA'],'-')
# plt.show()

def convert_dBm_to_W(pdBm):
    """ This function takes pdBm
        (power in dBm) and returns
        pW (power in mW)
    """
    pW = 10**(pdBm/10)
    return pW

########### Plot single OSA spectrum (matplotlib) ############
# peak_th = -66
# h1 = 0
# h3 = 0
# volt = 26
# ind_peaks,_ = find_peaks(data_raw[h3, h1, volt, :], height=peak_th, width=7, prominence=3)
# plt.plot(data_raw[h3, h1, volt, :],'-')
# plt.plot(ind_peaks, data_raw[h3, h1, volt, :][ind_peaks], '*')
# plt.show()

# print(convert_dBm_to_W(20))
########### Plot OSA spectrum (loop) ############
peak_th = -69
h3 = 2
h1 = 2
dif_vec = np.zeros(len(data_raw['DAQ_voltage']))
# for ind in range(len(data_raw['DAQ_voltage'])):
#     ind_peaks,_ = find_peaks(data_raw[h3, h1, ind, :], height=peak_th, distance=70, width=7, prominence=3)
#     dif_vec[ind] = np.diff(convert_dBm_to_W(data_raw[h3, h1, ind, ind_peaks]))[0]
#     plt.plot(convert_dBm_to_W(data_raw[h3, h1, ind, :]))
#     plt.plot(ind_peaks, convert_dBm_to_W(data_raw[h3, h1, ind, :][ind_peaks]), '*')
#     # plt.ylim([-80, 0])
#     plt.ylim([0, 0.02])
#     plt.pause(0.5)
#     plt.clf()
# for ind in range(len(data_raw['DAQ_voltage'])):
#     ind_peaks,_ = find_peaks(data_raw[h3, h1, ind, :], height=peak_th, distance=70, width=7, prominence=3)
#     dif_vec[ind] = np.diff(data_raw[h3, h1, ind, ind_peaks])[0]
#     plt.plot(data_raw[h3, h1, ind, :])
#     plt.plot(ind_peaks, data_raw[h3, h1, ind, :][ind_peaks], '*')
#     plt.ylim([-80, 0])
#     plt.pause(0.5)
#     plt.clf()

# plt.show()
# plt.plot(dif_vec)
# plt.show()

########### Plot single OSA spectrum (xarray) ############
# spec = data_raw.isel(h1_current=3, h3_current=2, DAQ_voltage=4)
# spec.plot()
# plt.show()

########### Plot voltage map for single heater current pair ############
# plt.pcolormesh(data_raw['Power_OSA'], data_raw['DAQ_voltage'], data_raw[2, 2], cmap='magma', vmin=-80)
# # plt.pcolormesh(data_raw[2, 2], cmap='magma', vmin=-80)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Laser detuning (V)')
# plt.colorbar()
# plt.show()

########### Plot voltage map loop ############
# for h3 in range(len(data_raw['h3_current'])):
#     for h1 in range(len(data_raw['h1_current'])):
#         plt.pcolormesh(data_raw['Power_OSA'], data_raw['DAQ_voltage'], data_raw[h3, h1], cmap='magma', vmin=-70, shading='auto')
#         plt.xlabel('Wavelength (nm)')
#         plt.ylabel('Laser detuning (V)')
#         plt.colorbar()
#         plt.show()


# h3=4
# for h1 in range(len(data_raw['h1_current'])):
#     plt.pcolormesh(data_raw['Power_OSA'], data_raw['DAQ_voltage'], data_raw[h3, h1], cmap='magma', vmin=-80, shading='auto')
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Laser detuning (V)')
#     plt.colorbar()
#     plt.show()

########### Calculate efficiency ############
peak_th = -66
eff_max = np.zeros((len(data_raw['h3_current']), len(data_raw['h1_current'])))
dif_min = np.zeros((len(data_raw['h3_current']), len(data_raw['h1_current'])))
for h3 in range(len(data_raw['h3_current'])):
    for h1 in range(len(data_raw['h1_current'])):
        eff = np.zeros(len(data_raw['DAQ_voltage']))
        dif_vec = np.zeros(len(data_raw['DAQ_voltage']))
        for ind in range(len(data_raw['DAQ_voltage'])):
            ind_peaks,_ = find_peaks(data_raw[h3, h1, ind, :], height=peak_th, distance=70, width=7, prominence=3)
            peaks_W = convert_dBm_to_W(data_raw[h3, h1, ind, ind_peaks])
            dif_vec[ind] = np.diff(peaks_W)[0]
            # dif_vec[ind] = np.diff(data_raw[h3, h1, ind, ind_peaks])[0]
            if len(ind_peaks)==3:
                eff[ind] = peaks_W[2] / (peaks_W[1] + peaks_W[0])
        eff_max[h3, h1] = np.amax(eff)*100
        dif_min[h3, h1] = np.amin(abs(dif_vec))
        # plt.plot(eff)
        # plt.plot(dif_vec)
        # plt.show()
plt.pcolormesh(data_raw['h1_current'], data_raw['h3_current'], eff_max, shading='auto')
# plt.pcolormesh(data_raw['h1_current'], data_raw['h3_current'], dif_min, shading='auto')
plt.xlabel('Heater 1 (mA)')
plt.ylabel('Heater 3 (mA)')
plt.colorbar()
plt.show()

# peak_th = -66
# eff_max = np.zeros(len(data_raw['h1_current']))
# dif_min = np.zeros(len(data_raw['h1_current']))
# h3 = 0
# for h1 in range(len(data_raw['h1_current'])):
#     eff = np.zeros(len(data_raw['DAQ_voltage']))
#     dif_vec = np.zeros(len(data_raw['DAQ_voltage']))
#     for ind in range(len(data_raw['DAQ_voltage'])):
#         ind_peaks,_ = find_peaks(data_raw[h3, h1, ind, :], height=peak_th, distance=70, width=7, prominence=3)
#         peaks_W = convert_dBm_to_W(data_raw[h3, h1, ind, ind_peaks])
#         # print(peaks_W)
#         dif_vec[ind] = np.diff(peaks_W)[0]
#         if len(ind_peaks)==3:
#             eff[ind] = peaks_W[2] / peaks_W[1]
#     eff_max[h1] = np.amax(eff)*100
#     dif_min[h1] = np.amin(abs(dif_vec))
#     # plt.plot(eff)
#     # plt.show()

# plt.plot(eff_max)
# plt.plot(dif_min)
# plt.show()

