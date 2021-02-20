import ice_ocean_plume 
import numpy as np
import matplotlib.pyplot as plt


######## Set parameters #######################################################

gldep = 500 # Plume start depth (grounding line, e.g.)
plume_type = 'cone' # Line or cone plume 
volfl0 = 1e-8 # For a line plume, this is volume per glacier front width (m2/s)

T0 = 0
S0 = 0
T0freeze = False
T0melt = True
frac_melt = 1

method = 'RK45'

######## Create fake profiles of ambient temperature and salinity #############

depa = np.arange(0, 501)
Ta = depa*0 + 2.0
Sa = depa*0 + 35.0
dz_upper = 1
Sa[:100] = 35-dz_upper+np.arange(100)/100*dz_upper
Sa[100:] = 35+np.arange(401)/401/10


######## Run model ############################################################

P = ice_ocean_plume.plume(gldep, volfl0, Ta, Sa, depa, theta= 90, 
                          plume_type = plume_type,
                          T0freeze = T0freeze, T0melt = T0melt, 
                          frac_melt = frac_melt)
P.set_params()
P.solve(manual_step = False, melt_on = True, method = method, max_step = 0.5, )

######## Print some results ###################################################

print('\n## RESULTS: ##\n')
print('Neutral depth: %.1f m'%P.neut_dep)
print('Minimum depth: %.1f m'%P.min_dep)
print('\nPlume temperature at neutral depth: %.2f C'%P.T_neut)
print('Ambient temperature at neutral depth: %.2f C'%P.Ta_neut)
print('Plume salinity at neutral depth: %.2f psu'%P.S_neut)
print('Ambient salinity at neutral depth: %.2f psu'%P.Sa_neut)
print('Plume speed at neutral depth: %.2f m/s'%P.U_neut)

if plume_type == 'line':
    volfl_melt_unit = 'm² s⁻¹'
else:
    volfl_melt_unit = 'm³ s⁻¹'

print('\nInitial T, S: %.2f C, %.2f psu'%(P.T0, P.S0))


print('\nInitial volume flux: %.3f %s'%(P.volfl0, volfl_melt_unit))
print('Upward volume flux at neutral depth: %.3f %s'%(P.volfl_neut, 
                                                      volfl_melt_unit))

print('\nIntegrated melt up to terminal depth: %.3f %s'%(P.melt_total, 
                                                         volfl_melt_unit))
print('Integrated melt up to neutral depth: %.3f %s'%(P.melt_to_neutral, 
                                                         volfl_melt_unit))     


print('\nIntegrated entrained ambient water up to terminal depth: %.3f %s'%(
                                                         P.entr_total, 
                                                         volfl_melt_unit))
print('Integrated entrained ambient water up to neutral depth: %.3f %s'%(
                                                         P.entr_to_neutral, 
                                                         volfl_melt_unit))   

print('\nAt terminal depth (neutral depth or surface), the plume consists of:'
      + '\nBottom volume flux (%.2f%%) / '%(P.terminal_frac_volfl0*100)
      + 'Meltwater (%.2f%%) / '%(P.terminal_frac_melt*100)
      + 'Entrained ambient water (%.2f%%)'%(P.terminal_frac_entr*100))

######## Quick plot of profiles with terminal/neutral depths ##################

varnms = ['D_pl', 'U_pl', 'T_pl', 'Tb_pl', 'S_pl', 'Sb_pl',
          'dRho_pl', 'M_pl', 'E_pl', 'Tf_a', 'Tf_pl'] 
descs = ['Plume width', 'Upwelling speed', 'Plume temp', 
         'I-O interface temp', 'Plume sal', 
         'I-O interface sal', 'Plume-ambient\ndensity difference', 
         'Melt rate', 'Entrainment rate', 
         'Ambient ocean freezing temp',
         'Plume freezing temp']
units = ['m', 'cm/s', 'C', 'C', 'psu', 'psu',
          '*1027 kg m$^{-3}$', 'm/day', 'm/day', 'C', 'C']
factors = [1, 1e2, 1, 1, 1, 1, 1027, 86400, 86400, 1, 1]

rows = 3
cols = np.int(np.ceil(((len(varnms)+1)/rows)))

fig, ax = plt.subplots(rows, cols, sharey = True, 
                       figsize = (9, 9))
axs = ax.flatten()

for varnm, desc, unit, factor, axn, in zip(varnms, descs, units, factors, axs):
    axn.plot(getattr(P, varnm)*factor, P.dep_pl, lw = 1.5, alpha = 0.85, 
             color = 'C1',)
    axn.set_title(desc, fontweight = 'normal', fontsize = 12)
    axn.axhline(P.neut_dep, ls = '--', color = 'C0', 
                zorder = 0, alpha = 0.7, lw = 2)
    axn.axhline(P.min_dep,  ls = '--', color = 'C2', 
                zorder = 0, alpha = 0.7, lw = 2)
    axn.set_xlabel('%s [%s]'%(varnm, unit))
    axn.grid()

for mm in np.arange(rows):
    ax[mm, 0].set_ylabel('Depth [m]')

axs[2].plot(P.Ta, P.depa, ':k', label = 'Ta')
axs[4].plot(P.Sa, P.depa, ':k', label = 'Sa')
axs[2].legend(fontsize = 10, handlelength = 1)
axs[4].legend(fontsize = 10, handlelength = 1)

axs[-1].plot(P.T_pl - P.Tf_pl, P.dep_pl, lw = 1.5, alpha = 0.77)
axs[-1].grid()
axs[-1].set_title('Thermal driving', fontweight = 'normal', fontsize = 12)
axs[-1].set_xlabel('T_pl-Tf_pl [C]')

axs[-1].set_ylim(gldep, 0)

for nn in [1, 7, 8, -1]:
    axs[nn].set_xlim(0, None)

axs[0].set_xlim(0, P.D_neut*1.3)
axs[4].set_xlim(P.S_pl.max()*0.9, None)

plt.tight_layout()


fig.text(0.2, 1, 'Plume properties', fontsize = 10, color = 'C1', 
        va = 'top', ha ='left')  

fig.text(0.5, 1, 'Minimum depth', fontsize = 10, color = 'C2',
            va = 'top', ha ='center')  
fig.text(0.8, 1, 'Neutral (zero buoyancy) depth', fontsize = 10, 
         color = 'C0', va = 'top', ha ='right')      