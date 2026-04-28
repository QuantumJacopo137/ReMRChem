from math import e
import numpy as np
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1
import starting_guess as sg
# Import quaternionic classes
from quaternionic.quat_function import QuatFunction
from quaternionic.quat_orbital import CompQuatOrbital, QuatOrbital
from quaternionic import quat_function as qfun
from quaternionic import quat_orbital as qorb
from orbital4c import orbital as orb
from orbital4c import complex_fcn as cf
from orbital4c import nuclear_potential as nucpot

# == Define Environment ====
position = [0.0, 0.0, 0.0]
charge = 1
prec = 1e-6
thr = prec * 10
ordr = int(-np.log10(prec) + 4)
mra = vp.MultiResolutionAnalysis(box=[-50, 50], order=ordr, max_depth=25)
light_speed = 137.0359895
QuatFunction.mra = mra
orb.orbital4c.mra = mra
cf.complex_fcn.mra = mra
orb.orbital4c.light_speed = light_speed
QuatFunction.light_speed = light_speed
QuatOrbital.light_speed = light_speed

# == Define Starting Guess ====
qf = QuatFunction()
qf1 = QuatFunction()
Qo = QuatOrbital()

#A = sg.make_NR_starting_guess(position=position, charge=1, mra=mra, prec=prec/10)
#(A['La'].real).saveTree("NR_Guess_La")

Qo = sg.make_NR_starting_guess_quaternion(position=position, charge=1, mra=mra, prec=prec/10, light_speed=light_speed, name="NR_Guess_La")

# SECONDO ME CI DEVE ESSERE UN ERRORE NELLA GENERAZIONE STARTING GUESS, DATO CHE VIENE ALPHA = 0
Qo.normalize()
Qo.print_all_Sqnorms()

# === Potential ===
print("Harrison potential")
Peps = vp.ScalingProjector(mra, prec/10)
potential = vp.FunctionTree(mra)
#f = lambda x: nucpot.coulomb_HFYGB(x, position, charge, prec/10)
#potential = Peps(f)
#potential.saveTree(f"potential")

potential.loadTree(f"potential")





# === Ground State Calculation ===
error_norm = 1


c2 = light_speed**2
old_energy = 0
delta_e = 1
idx = 0
tmp = QuatOrbital()
idx_array = []
T_array = []
V_array = []
energy_array = []
norm_diff_array = []
while (idx < 20 and (delta_e > prec/10 or error_norm > thr)):
    print()
    print('$ Iteration', idx)
    # ENERGY EVALUATION
    hd_psi = qorb.apply_dirac_hamiltonian(Qo, prec, der = 'ABGV') # Looks like it is working
    v_psi = qorb.apply_potential(-1.0, potential, Qo, prec) # Looks like it is working
    
    add_psi = hd_psi + v_psi
    energy = Qo.dot(add_psi).real
    T = hd_psi.dot(Qo).real
    V = v_psi.dot(Qo).real
    print()
    print('     <T_Dirac>', T-c2)
    print('     <T_class>', Qo.classicT())
    print('     <V>', V)
    print('     <H>', energy)
    print('     Energy',energy - c2)
    T_array.append(T-c2)
    V_array.append(V)
    energy_array.append(energy - c2)
    idx_array.append(idx)
    norm_diff_array.append(error_norm)

    
    # propagator 
    mu = orb.calc_dirac_mu(energy, light_speed)
    tmp = qorb.apply_helmholtz(v_psi, mu, prec)
    tmp.crop_Top_Bot(prec)

    new_orbital = qorb.apply_dirac_hamiltonian(tmp, prec, der = 'ABGV', shift = energy)
    if(idx > 10):
        new_orbital = new_orbital + Qo
    new_orbital.crop_Top_Bot(prec)
    new_orbital.normalize()

    # ERROR EVALUATION  
    delta_psi = new_orbital - Qo
    error_norm = np.sqrt(delta_psi.squaredNorm())
    delta_e = np.abs(energy - old_energy)
    print()
    print('     Energy',energy - light_speed**2)
    old_energy = energy
    
    
    print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec/10)
    Qo = new_orbital
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
print("NON RELATIVISTIC ENERGY", V + Qo.classicT())#
