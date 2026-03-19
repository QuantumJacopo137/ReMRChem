########## Define Enviroment #################
from scipy.differentiate import derivative
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import orbital_2c as orb2c
from orbital4c import nuclear_potential as nucpot
from orbital4c import r3m as r3m
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1

import argparse
import numpy as np
import numpy.linalg as LA
import sys, getopt

import one_electron as oneel
import two_electron as twoel
import Four_el as fourel
import starting_guess as sg
import Lazy_4_el as lazy4el

import importlib
importlib.reload(orb)

import fileinput
import sys


light_speed = 137.0359895           # default 137.03599913900001
derivative = "ABGV"                 # possible values: PH, ABGV, BS
order = 9                           # max order of the polynomials in the MRA
box = 18                            # size of the box in atomic units (half box length)     
prec = 1.0e-3                       # precision threshold for the MRA operations
thr  = 1.0e-3                       # error threshold for SCF convergence
auto_box = False

readPotential    = True             # bool
computePotential = False            # bool 
savePotential    = True             # bool
potential = "coulomb_HFYGB"         # possible values: point_charge coulomb_HFYGB homogeneus_charge_sphere gaussian

continue_run     = False            # bool
readOrbitals     = False            # bool
saveOrbitals     = True             # bool
saveGuess        = False

scf              = True
ktrs             = True
D2               = False            # bool
two_electrons    = False            # bool
four_el          = True             # bool
two_components   = True             # bool


molecule = [ ["Be", 4, 0.1, 0.2, 0.3, 0, 0] ]

#
# 1. This code works now only for atoms and up to two electrons with KTRS
# 2. Orbital guess obtained by using NR hydrogenionic 1s orbital
# 3. The input file is a Python code mostly containing variable allocation
# 4. Input parsing is executing that python input after reading it
# 5. Nuclear potential selected manually (Gaussian now to reproduce Harrison's results)
#


if (auto_box):
    box = int(np.ceil(float(50/molecule[0][1])))

################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-box, box], order=order, max_depth=25)
orb.orbital4c.mra = mra
orb2c.orbital2c.mra = mra
orb.orbital4c.light_speed = light_speed
orb2c.orbital2c.light_speed = light_speed
cf.complex_fcn.mra = mra

charge = molecule[0][1]
position = [molecule[0][2],molecule[0][3],molecule[0][4]]
radius = molecule[0][5]
epsilon = molecule[0][6]



################### Define V potential ######################
Peps = vp.ScalingProjector(mra, prec/10)
V_tree = vp.FunctionTree(mra)

if(computePotential):
    typenuc = potential
    f = 0
    if(potential == "gaussian"):
        print("Gaussian potential")
        f = lambda x: nucpot.gaussian_potential(x, position, charge, epsilon)
        V_tree = Peps(f)
    elif(potential == "coulomb_HFYGB"):
        print("Harrison potential")
        f = lambda x: nucpot.coulomb_HFYGB(x, position, charge, prec/10)
        V_tree = Peps(f)
    elif(potential == "point_charge"):
        print("point charge potential")
        f = lambda x: nucpot.point_charge(x, position, charge)
        V_tree = Peps(f)
    elif(potential == "fermi_dirac"):
        print("Fermi Dirac potential")
        if radius == 0:
            with open("Half_Charge_Radius.txt", "r") as f:
                half_charge_radius_dict = {}
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        symbol, value = parts
                        if symbol == molecule[0][0]:
                            HCR = float(value)
                            break
        else:
            HCR = radius

        print(f"-> Using Half Charge Radius for {molecule[0][0]}: {HCR}")
            
        V_tree = nucpot.Fermi_Dirac(position, charge, box, mra, order, prec, HCR)
    else:
        exit(-1)
    #V_tree = Peps(f)
elif(readPotential):
    V_tree.loadTree(f"potential")
if(savePotential):
    V_tree.saveTree(f"potential")


# showing the variables used
print()
print("------------------------------------")
print("      Calculation parameters ")
print("------------------------------------")
print("light_speed =", light_speed)
print("derivative =", derivative)
print("Nuclear Potential Type =", potential)
print("box =", box)
print("precision =", prec)
print("order =", order)
print("threshold =", thr)
print("Charge =", charge)
        
print("Number of Atoms = ", len(molecule))
print(molecule)
print()
print()
############################ START WITH CALCULATION ##################################

# Use this as a sandbox for testing all the subroutines

L_array = [orb2c.orbital2c(),
           orb2c.orbital2c(),
           orb2c.orbital2c(),
           orb2c.orbital2c()]

R_array = [orb2c.orbital2c(),
           orb2c.orbital2c(),
           orb2c.orbital2c(),
           orb2c.orbital2c()]
# In case, to generate the L spinors here it is:
#spinorb_array = fourel.init_4_spinors(position, charge, mra, prec)

# Otherwise, read them from file

readOrbitals = True

Dirac_array = [orb.orbital4c() for i in range(4)]


if not continue_run:
    if (not readOrbitals):
        L_array[0] = sg.make_NR_starting_guess(position, 2.45744, mra, prec,comp=2,  n=2, l=0)
        L_array[1] = L_array[0].ktrs(prec)
        L_array[2] = sg.make_NR_starting_guess(position, 3.67137, mra, prec, comp=2, n=1, l=0)
        L_array[3] = L_array[2].ktrs(prec)


    for i in range(4):
        name = f"W_spinor{i}"
        #spinorb_array[i].save(name)
        if not readOrbitals:
            L_array[i].save(name)
        
        L_array[i].read(name)
            
        tmp = L_array[i].sigma_p(prec, derivative)
        # REMEMBER THAT I DEFINED SIGMAS AS -SIGMAS, SO THE SIGN IN THE FORMULA IS CHANGED
        R_array[i] = L_array[i] - (1/(2*light_speed)) * tmp
        L_array[i] = L_array[i] + (1/(2*light_speed)) * tmp
    

    for i in range(4):
        Dirac_array[i] = lazy4el.Weyl_to_Dirac(L_array[i], R_array[i])
        Dirac_array[i].normalize()

else:
    for i in range(4):
        name = f"Last_run_spinor_{i}"
        Dirac_array[i].read(name)



Dirac_array, Fock_matrix =lazy4el.scf_4el(Dirac_array, V_tree, mra, prec, max_iter=2)

for i in range(4):
    Dirac_array[i].save(f"Last_run_spinor_{i}")












# ========== END OF CALCULATION, PRINTING RESULTS ================
output_file = "output_test2.txt"
oneel.write_and_print(output_file, "")
print("********************************")

if two_components:
    oneel.write_and_print(output_file, "-> 2 COMPONENTS CALCULATION ")
else:
    oneel.write_and_print(output_file, "-> 4 COMPONENTS CALCULATION ")      

oneel.write_and_print(output_file, "PARAMETERS:")
oneel.write_and_print(output_file, f"molecule    = {molecule}")
oneel.write_and_print(output_file, f"D2          = {D2}")
oneel.write_and_print(output_file, f"prec        = {-int(np.log10(prec))}")
oneel.write_and_print(output_file, f"order       = {order}")
oneel.write_and_print(output_file, f"derivative  = {derivative}")
oneel.write_and_print(output_file, f"box         = {box}")
if position == [0.0, 0.0, 0.0]:
    centerd = True
else:
    centerd = False
oneel.write_and_print(output_file, f"centered    = {centerd}")

oneel.write_and_print(output_file, "")
oneel.write_and_print(output_file, "-> ID calculation ")
oneel.write_and_print(output_file, "-------------------------------")
oneel.write_and_print(output_file, f"{molecule[0][0]} {int(centerd)} {int(D2)} {-int(np.log10(prec))} {derivative} {box}")
oneel.write_and_print(output_file, "-------------------------------")
