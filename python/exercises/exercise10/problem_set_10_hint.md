# Hints

<details>
<summary><strong>1a</strong></summary>

Solve $\mathbf{K}\boldsymbol{\phi}_n=\omega_n^2\mathbf{M}\boldsymbol{\phi}_n$ and use $f_n=\omega_n/(2\pi)$ to obtain frequencies in Hz. Normalize each mode shape by its largest absolute component.

The first natural frequency is approximately $1.13\,\mathrm{Hz}$.

</details>

<details>
<summary><strong>1b</strong></summary>

Use `scipy.signal.csd` to estimate the auto- and cross-spectral densities. Look for peaks in the sum of the auto-spectral densities, and compare different segment lengths.

At an isolated resonance, estimate the magnitude of each mode-shape component relative to a reference DOF using the square root of the ratio of their auto-spectral densities. Obtain the relative sign from the real part of the cross-spectrum with the reference DOF. Choose a reference with a clear response at the resonance.

</details>

<details>
<summary><strong>1c</strong></summary>

Assemble the cross-spectral density matrix and calculate its singular value decomposition at each frequency. Look for peaks in the largest singular value. At an isolated resonance, the corresponding first left singular vector approximates the mode shape.

Mode shapes have arbitrary scale and sign; account for this when comparing them with the analytical modes.

</details>

<details>
<summary><strong>2</strong></summary>

Start with the vertical acceleration channels and inspect the singular values below 15 Hz. Peaks near 3 Hz and 6 Hz are worth investigating.

Plot the mode-shape components along the bridge separately for the two sensor rows. Do the two sides move in the same direction or in opposite directions? Use this to distinguish vertical bending from torsion.

A peak concentrated at one sensor need not represent a global bridge mode. Compare the spatial pattern before interpreting the peak.

</details>
