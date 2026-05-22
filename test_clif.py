from argparse import RawDescriptionHelpFormatter
from math import e
from multiprocessing.forkserver import SIGNED_STRUCT
import numpy as np
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1
from orbital4c import complex_fcn as cf
from orbital4c import nuclear_potential as nucpot
import starting_guess as sg
# Import quaternionic classes

import Clifford as clif
from orbital4c import orbital as orb
# == Define Environment ====
position = [0.0, 0.0, 0.0]
charge = 80
prec = 1e-6
thr = prec * 10
ordr = int(-np.log10(prec) + 4)
box = int(np.ceil(float(50/charge)))
mra = vp.MultiResolutionAnalysis(box=[-box, box], order=ordr, max_depth=25)
light_speed = 137.0359895
clif.ClifFunc.mra = mra
#orb.orbital4c.mra = mra
cf.complex_fcn.mra = mra
#orb.orbital4c.light_speed = light_speed
clif.ClifFunc.light_speed = light_speed
#QuatOrbital.light_speed = light_speed

# == Define Starting Guess ====
Psi = clif.ClifFunc()




recompute = False
if recompute:
    A = sg.make_NR_starting_guess(position=position, charge=charge, mra=mra, prec=prec/10)
    (A['La'].real).saveTree("NR_Guess_La")

NR_phi = vp.FunctionTree(mra)
NR_phi.loadTree("NR_Guess_La")
Psi = clif.Clif_starting_guess_NR(NR_phi, prec/10, light_speed)

Psi.print_all_Sqnorms()


# === Potential ===
potential = vp.FunctionTree(mra)
if recompute:
    print("Harrison potential")
    Peps = vp.ScalingProjector(mra, prec/10)
    f = lambda x: nucpot.coulomb_HFYGB(x, position, charge, prec/10)
    potential = Peps(f)
    potential.saveTree(f"potential")
potential.loadTree(f"potential")


# === Ground State Calculation ===
error_norm = 1


c2 = light_speed**2
old_energy = 0
delta_e = 1
idx = 0
tmp = clif.ClifFunc()
idx_array = []
T_array = []
V_array = []
energy_array = []
norm_diff_array = []

hd_psi = clif.ClifFunc()
v_psi = clif.ClifFunc()
add_psi = clif.ClifFunc()
tmp = clif.ClifFunc()
new_orbital = clif.ClifFunc()
delta_psi = clif.ClifFunc()


while (idx < 100 and ( (delta_e > prec/10) or (error_norm > thr))):
    print()
    print('$ Iteration', idx)
    v_psi.apply_potential(-1.0, potential, Psi, prec) # Looks like it is working
    # ENERGY EVALUATION
    hd_psi = clif.apply_Dirac_Hestenes_hamiltonian(light_speed, prec, Psi) # Looks like it is working

    add_psi = hd_psi + v_psi
    energy = Psi.dot(add_psi).real
    #T = hd_psi.dot(Psi).real
    #V = v_psi.dot(Psi).real
    T = 0
    V = 0
    print()
    #print('     <T_Dirac>', T-c2)
    print('     <T_class>', Psi.classicT())
    #print('     <V>', V)
    print('     <H>', energy)
    print('     Energy',energy - c2)
    T_array.append(T-c2)
    V_array.append(V)
    energy_array.append(energy - c2)
    idx_array.append(idx)
    norm_diff_array.append(error_norm)

    #ideal_energy = -3532.192093162126
    # propagator
    mu = orb.calc_dirac_mu(energy, light_speed)
    tmp = v_psi.apply_helmoltz(mu, prec)
    tmp.normalize()
    tmp.crop(prec)

    new_orbital = clif.apply_Dirac_Hestenes_hamiltonian(light_speed, prec, tmp, shift = energy)

    if(idx > 25):
        new_orbital = new_orbital + Psi
    new_orbital.crop(prec)
    new_orbital.normalize()

    # ERROR EVALUATION  
    delta_psi = new_orbital - Psi
    error_norm = np.sqrt(delta_psi.squaredNorm())
    delta_e = np.abs(energy - old_energy)
    print()
    print('     Energy',energy - light_speed**2)
    old_energy = energy
    
    
    print('     Converged? ', error_norm, ' > ', prec, '  ----  ', delta_e, ' > ',prec/10)
    Psi = new_orbital
    if error_norm < thr:
        print('Convergence achieved!')
        break
    idx += 1


T_array.append(T-c2)
V_array.append(V)
energy_array.append(energy - c2)
idx_array.append(idx)
norm_diff_array.append(error_norm)

idx += 1

print()
print() 
# List of the array indices
print("|idx| T contrib. \t V contrib. \t E contrib. \t norm diff")
for i in range(idx):
    print(f"| {i} | {T_array[i]}\t{V_array[i]}\t{energy_array[i]}\t{norm_diff_array[i]}|")

print("---------------------------------------------------------------------------")
print("Final Energy", energy - light_speed**2)
print("---------------------------------------------------------------------------")
print("NON RELATIVISTIC ENERGY", V + Psi.classicT())#

print("---------------------------------------------------------------------------")
Psi.print_all_Sqnorms()
print("---------------------------------------------------------------------------")
print(Psi)
