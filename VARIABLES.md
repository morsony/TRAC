
## TRAC Variable Reference

| Variable | Physical Meaning / Use | Main Location(s) |
|---|---|---|
| `eiso`, $E_{\mathrm{iso}}(\theta)$ | Isotropic-equivalent energy; angular jet energy structure. | run; testpoint; dynamics/hydro |
| `miso`, $M_{\mathrm{iso}}(\theta)$ | Isotropic-equivalent ejecta mass; tied to energy and initial Lorentz factor. | testpoint; dynamics |
| `gamma_init`, $\Gamma$ | Initial/bulk Lorentz factor. `gamma_init='file'` uses Lazzati input. | run; testpoint; hydro |
| `rot_angle` | Viewing/off-axis rotation angle. | run; coord_change2; initial geometry |
| `k_index`, $k$ | External-density index: $\rho_{\mathrm{ext}}\propto r^{-k}$. $k=0$ for the active ISM run. | run; hydro; dynamics |
| `external_nodensity` | External number-density normalization. | run; hydro; radiation |
| `m_index`, $m$ | Blast-wave similarity/evolution index; current run sets $m=3-k$. This is not mass. | run; hydro |
| `epsilon`, $\epsilon_e$ | Electron-energy microphysics parameter; exact naming depends on the routine. | run; radiation; jitter |
| `eqparb`, $\epsilon_B$ | Magnetic-field energy microphysics parameter. | run; radiation; jitter |
| `pindex`, $p$ | Power-law index of accelerated electrons: $N(\gamma_e)\propto\gamma_e^{-p}$. | run; jitter |
| `mingamma`, $\gamma_{\min}$ | Minimum Lorentz factor of the nonthermal electron distribution. | jitter; global_var |
| `bsq`, $B$ | Magnetic-field state; jitter derives $B$ from stored $B^2$. | jitter; global_var |
| `alpha`, $\alpha$ | Synchrotron absorption coefficient in jitter microphysics. | jitter |
| `t_arr` | Time grid, with redshift handling in run/post-processing. | run; afterglow; post-processing |
| `wmax`, `wmin`, $\omega$ | Angular-frequency grid/limits; observer frequency ultimately $\nu=\omega/(2\pi)$. | run; radiation; post-processing |
| `phi_slices` | Azimuthal resolution around observer line of sight; GW170817 run uses 54. | run; hydro; afterglow |
| `s_points`, `z_points` | Spatial-grid resolution used in hydro / LOS calculation. | run; hydro |
| `shock_thickness` | Thickness parameter used for the shocked region. | run; global_var; dynamics/hydro |
| `spread` | Lateral-spreading prescription; active GW170817 run uses `spread=0`. | run; testpoint |
| `glow` | 4D hydrodynamic-state container passed from hydro to radiation. | hydro; afterglow |
| `total_power_arr` | Numerical spectral-radiation output indexed by time, frequency, $\phi$. | afterglow output; integrate/image |
| `freq_arr` | Frequency information saved with the radiation output. | afterglow; integrate |
| `angle_int_arr` | Spatial/angular radiation information used especially for resolved imaging. | afterglow; make_image |
| `flux`, `jy` | Observer flux density; Jy is the radio-friendly flux-density unit. | integrate_total_power_interp |
| `dl`, $d_L$ | Luminosity distance used in power-to-flux conversion. | post-processing |

