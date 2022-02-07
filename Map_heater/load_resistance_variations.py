from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt

########### Load DataSet ############
file_dir = 'G:\\Meu Drive\\LPD Team\\Experiments\\DOPO\\Data\\R-R-R_molecule 600-650\\'
file_name = '22-02-01_12.12_chip1_R-R-R_wg-ring gap 600 nm_ring-ring gap 650 nm_TE_through port_ResistenceVariation_curr_30mA'
data = pd.read_parquet(file_dir+file_name+'.parq')

plt.plot(data.R1, label='heater 1')
plt.plot(data.R3, label='heater 3')
plt.xlabel('time (s)')
plt.ylabel('resistance (ohms)')
plt.legend()
plt.savefig(file_dir+file_name+'.png')
plt.show()
