# DaRoC (Dynamical Absorption of Right Orbital Components) Algorithm

This document serves as a comprehensive technical specification and theoretical context for implementing the **DaRoC** algorithm in a Relativistic Quantum Chemistry (RQC) framework. It is optimized for use by GitHub Copilot and development agents to understand the core equations, mathematical workflows, and algorithmic structures required for code generation.

---

## 1. Core Philosophical & Mathematical Foundations

### 1.1 The Weyl (Chiral) Basis
Traditionally, Relativistic Quantum Chemistry (RQC) software relies on the **Dirac basis**. However, the **Weyl basis** (or *chiral basis*) offers distinct computational advantages. While it hides explicit energy contributions, it renders the Hamiltonian significantly simpler and mostly diagonal. The coupling between the two chiralities (Left and Right) is mediated strictly through the mass term, and derivative terms appear only along the diagonals. This allows each Weyl spinor to be expressed as a function of the other without requiring operator inversion.

### 1.2 Notation & Conventions
To minimize minus signs when working predominantly with Left ($\psi_L$) chirality, we define the modified helicity operator $\hat{\pi}$:
$$\hat{\pi} = -\sigma \cdot p$$
where $\sigma$ represents the vector of Pauli matrices and $p$ is the momentum operator.

The 4-component Weyl equation is expressed as:
$$E \begin{pmatrix} \psi_L \\ \psi_R \end{pmatrix} = \begin{pmatrix} \hat{V} + c \hat{\pi} & mc^2 \\ mc^2 & \hat{V} - c \hat{\pi} \end{pmatrix} \begin{pmatrix} \psi_L \\ \psi_R \end{pmatrix}$$

This expands into a system of two coupled 2-component equations:
1. $E \psi_L = \hat{V}\psi_L + c \hat{\pi}\psi_L + mc^2\psi_R$
2. $E \psi_R = \hat{V}\psi_R - c \hat{\pi}\psi_R + mc^2\psi_L$

---

## 2. Single-Electron Propagator & Green's Function Convolution

By inverting the first 2-component equation, we define the coupling operator $\hat{R}(E)$ such that $\psi_R = \hat{R}\psi_L$:
$$\hat{R} = -\frac{1}{mc^2}\left((\hat{V}-E) + c \hat{\pi}\right)$$

Substituting this into the second equation and utilizing the Pauli matrix identity $\hat{\pi}^2 = p^2 = -\nabla^2$ yields the single-electron propagator differential equation:
$$\psi_L = \frac{1}{m^2c^4} \left\{\hat{V}^2 - 2E \hat{V} + E^2 - c \left[\hat{\pi},\hat{V}\right] + c^2 \nabla^2\right\}\psi_L$$

### 2.1 Helmholtz Rearrangement
To solve this iteratively, the equation is rearranged into a non-homogeneous Helmholtz equation form:
$$\left[-\nabla^2 + \left(\frac{m^2c^4-E^2}{c^2}\right)\right]\psi_L = \frac{1}{c^2} \left\{\hat{V}^2 - 2E \hat{V} - c \left[\hat{\pi},\hat{V}\right] \right\}\psi_L$$

By defining the screening parameter $\mu^2$:
$$\mu^2 = \frac{m^2c^4-E^2}{c^2}$$

We can apply a Green's function convolution operator $\hat{G}^\mu = (-\nabla^2 + \mu^2)^{-1}$ via the integral kernel:
$$\hat{G}^\mu \star f(r) = \int \frac{e^{-\mu |r-r'|}}{4\pi |r-r'|} f(r') \, dr'$$

### 2.2 Local Potential Simplification
For purely local potentials (applicable to 1-electron and 2-electron systems), the commutator simplifies exactly to:
$$-c \left[\hat{\pi}, \hat{V}\right]\psi_L = ic\sigma \cdot \vec{\nabla}\hat{V}\psi_L$$

This results in the elegant 1-electron iteration formula:
$$\psi_L^{n+1} = \hat{G}^{\mu_n} \star \frac{\hat{\mathcal{V}}}{c^2}\psi_L^n \quad \text{where} \quad \hat{\mathcal{V}} = \hat{V}^2 - 2 E\hat{V} + i c (\sigma \cdot \vec{\nabla}\hat{V})$$

---

## 3. Many-Electron Theory (Dirac-Fock Framework)

In a many-electron system, we operate within a Dirac-Fock framework where the 4-component relativistic kinetic energy couples the Weyl spinors inside the mean-field Fock operator $\hat{\mathcal{F}}$:
$$\hat{\mathcal{F}} = c (\alpha^W \cdot p) + mc^2 \beta^W + (\hat{V}_{ce} + \hat{J} - \hat{K})\, \mathbb{I}_4$$
where $\hat{J}$ is the Coulomb operator, $\hat{K}$ is the non-local exchange operator, and $\hat{V} = \hat{V}_{ne} + \hat{J} - \hat{K}$.

### 3.1 Exact Many-Electron Coupling & Propagator
Because the Fock matrix elements $F_{ij} = \langle\Psi_i|\hat{\mathcal{F}}|\Psi_j\rangle$ are generally non-diagonal during the SCF cycle, the off-diagonal elements introduce coupling terms between different molecular orbitals:
$$E_i \Psi^i = \hat{\mathcal{F}}\Psi^i - \sum_{j \neq i} F_{ij}\Psi^j$$

This yields the **Exact Many-Electron Coupling Relation**:
$$\psi_R^i = -\frac{1}{mc^2}\left\{\left[(\hat{V}-E_i)+ c \hat{\pi}\right] \psi_L^i - \sum_{j \neq i}F_{ij}\psi_L^j\right\}$$

The corresponding **Exact Many-Electron Propagator** equation for the Left component becomes:
$$[\psi_L^i]^{n+1} = \frac{1}{c^2}\hat{G}^{\mu_n}\star \left( {\hat{\mathcal{V}}^n}[\psi_L^i]^n + 2\sum_{j \neq i} F_{ij}\left( \frac{E^n_i+E^n_j}{2}-\hat{V} \right)[\psi_L^j]^n + \sum_{j \neq i}\sum_{k \neq j} F_{ij}F_{jk} [\psi_L^k]^n\right)$$

*Crucial Implementation Note:* Due to the non-local nature of the exchange operator $\hat{K}$, the local commutator simplification does **not** hold. The full commutator must be explicitly retained:
$$\hat{\mathcal{V}} = \hat{V}^2 - 2E\hat{V} - c \left[\hat{\pi}, \hat{V}\right]$$

### 3.2 The $\hat{\mathcal{V}}$ Potential Simplification Scheme
Evaluating $\hat{V}^2$ and the explicit commutator involving non-local exchange poses severe computational bottlenecks. To circumvent this, the definition can be structurally mapped to the coupling relation:
$$\hat{\mathcal{V}}\psi_L^i = -mc^2 \,\hat{V}\psi_R^i - E_i \, \hat{V}\psi_L^i - c \hat{\pi} \, \hat{V}\psi_L^i + \sum_{j \neq i} F_{ij}\,\hat{V}\psi_L^j$$

By utilizing this identity, the evaluation of $\hat{\mathcal{V}}\psi_L^i$ relies entirely on quantities already computed during the potential action step ($\hat{V}\psi_L$ and $\hat{V}\psi_R$), eliminating the $\hat{V}^2$ and commutator overhead.

### 3.3 Inter-Electronic Overlaps via 2-Components
To compute $\hat{J}$ and $\hat{K}$ integrals efficiently using only the 2-component Left spinors, we define the metric transformation operator $\hat{\Omega}_{ij}$ to reproduce the exact 4-component overlap:
$$\langle\psi_L^i|\hat{\Omega}_{ij}|\psi_L^j\rangle \equiv \langle\Psi_i|\Psi_j\rangle$$
$$\hat{\Omega}_{ij} = 1 + \hat{R}_i^\dagger \hat{R}_j - \frac{E_i}{mc^2}\hat{R}_j - \frac{E_j}{mc^2}\hat{R}^\dagger_i - \sum_{k=1}^N \left(\frac{F_{ki}}{mc^2} \hat{R}_j + \frac{F_{kj}}{mc^2} \hat{R}_j^\dagger \right) + \dots$$
In initial SCF iterations, a starting approximation of $\hat{\Omega}_{ij} \approx 2$ can be employed to accelerate early macrocycles before enabling the exact form at close convergence.

---

## 4. Initial Guess Generation

Setting $\psi_L = \psi_R = \varphi$ (the non-relativistic solution) is unphysical; it yields a kinetic energy expectation value of exactly $mc^2$, representing a perfectly static electron. 

The correct, mathematically sound non-relativistic starting guess for generating the initial unnormalized Weyl spinors from a standard Schrödinger spatial solution $\varphi$ is:
$$\psi_L^0 = \frac{1}{\sqrt{2}}\left(1 + \frac{\sigma \cdot p}{2mc}\right)\begin{pmatrix} \varphi \\ 0 \end{pmatrix}$$
$$\psi_R^0 = \frac{1}{\sqrt{2}}\left(1 - \frac{\sigma \cdot p}{2mc}\right)\begin{pmatrix} \varphi \\ 0 \end{pmatrix}$$

This form reproduces the correct non-relativistic kinetic energy limit: $\langle T \rangle = \frac{\langle p^2 \rangle}{2m} + mc^2$.

---

## 5. The DaRoC SCF Algorithmic Workflow

The **DaRoC** (**D**ynamical **a**bsorbtion of **R**ight **O**rbital **C**omponents) method operates as an exact 4-component solver implemented at the computational cost of a 2-component method. It achieves this by strictly propagating the Left ($\psi_L$) components via Green's function convolutions and dynamically reconstructing the Right ($\psi_R$) components via the exact coupling relation. 

For maximum performance, the convolution steps should be evaluated using a **Multiwavelets basis set** (local Legendre polynomials), which provides an optimal sparse representation for integral convolution operators.

### Step-by-Step Implementation Loop

1. **Initialization:**
   * Generate the initial 2-component guesses $\psi_L^0$ and $\psi_R^0$ from the non-relativistic atomic solutions.
   * Combine them into initial 4-component spinors: $\Psi_i = \psi_L^i \oplus \psi_R^i$.
   * Orthonormalize the initial set using the **Löwdin orthonormalization procedure**.

2. **Potential Action Step:**
   * Compute the full mean-field potential $\hat{V} = \hat{V}_{ne} + \hat{J} - \hat{K}$.
   * Apply the potential to all current 4-component spinors to obtain the array of product states: $\left\{\hat{V}\Psi_i\right\}$ (which explicitly yields $\hat{V}\psi_L^i$ and $\hat{V}\psi_R^i$).

3. **Fock Matrix Construction:**
   * Compute all matrix elements of the Fock matrix: $F_{ij} = \langle\Psi_i|\hat{\mathcal{F}}|\Psi_j\rangle$.
   * Identify individual orbital energies from the diagonal elements: $E_i = F_{ii}$.

4. **Left-Chiral Component Propagation:**
   * Extract the Left components $\psi_L^i$ from the current spinor set.
   * Evaluate the Many-Electron Propagator equation using the pre-calculated Fock matrix elements $F_{ij}$ and the simplified potential scheme $\hat{\mathcal{V}}\psi_L^i$.
   * Perform the Green's function convolution $\hat{G}^{\mu_n} \star \dots$ to generate the updated intermediate Left components: $\tilde{\psi}_L^{N+1}$.

5. **Right-Chiral Component Absorption (Reconstruction):**
   * Using the newly updated $\tilde{\psi}_L^{N+1}$ components along with the *previous* step's potential actions and Fock matrix elements, dynamically calculate the balanced intermediate Right components $\tilde{\psi}_R^{N+1}$ via the Exact Many-Electron Coupling Relation:
     $$\tilde{\psi}_R^{i, N+1} = -\frac{1}{mc^2}\left\{\left[(\hat{V}-E_i)+ c \hat{\pi}\right] \tilde{\psi}_L^{i, N+1} - \sum_{j \neq i}F_{ij}\tilde{\psi}_L^{j, N+1}\right\}$$

6. **Assembly & Orthonormalization:**
   * Concatenate the intermediate components to form updated 4-component spinors: $\tilde{\Psi}_i^{N+1} = \tilde{\psi}_L^{i, N+1} \oplus \tilde{\psi}_R^{i, N+1}$.
   * Pass the full set $\left\{\tilde{\Psi}_i^{N+1}\right\}$ through a **Löwdin Orthonormalization** step to yield the final orthonormalized spinors for this macrocycle: $\left\{\Psi_i^{N+1}\right\}$.

7. **Convergence Check:**
   * Evaluate the norm of the difference between successive iterations for the Left component: $||\psi_L^{N+1} - \psi_L^N||$.
   * If $||\psi_L^{N+1} - \psi_L^N|| \ge \delta$ (where $\delta$ is the user-defined threshold), set $N \leftarrow N+1$ and loop back to **Step 2**.
   * If the difference is $< \delta$, terminate the loop; the solution has achieved 4-component exactness.