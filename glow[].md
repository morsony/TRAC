
## Hydrodynamic State Array: `glow`

The `glow` array stores hydrodynamic quantities used in the afterglow radiation calculations.

| Index | Variable | Physical Meaning |
|---|---|---|
| `glow[0]` | $z$ | Axial coordinate |
| `glow[1]` | $s$ | Cylindrical/projected distance from the jet axis |
| `glow[2]` | $r_{\mathrm{small}}$ | Current fluid spherical-radius quantity |
| `glow[3]` | $\theta$ | Polar-angle grid |
| `glow[4]` | $\theta'$ | Relativistically transformed/aberrated angular quantity |
| `glow[5]` | $n$ | Number density, calculated as $\rho/m_p$ |
| `glow[6]` | $P$ | Pressure/internal-energy-related quantity; exact convention needs verification |
| `glow[7]` | $\Gamma$ | Fluid Lorentz factor |
| `glow[8]` | $R_{\mathrm{shock}}$ | Shock-radius-related quantity; exact convention needs verification |
| `glow[9]` | $\xi$ | Blandford–McKee similarity variable |
| `glow[10]` | $\beta$ | Dimensionless fluid velocity, $\beta=v/c$ |
| `glow[11]` | $E_{\mathrm{iso}}$ | Interpolated isotropic-equivalent energy |
