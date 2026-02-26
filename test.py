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

import importlib
importlib.reload(orb)

import fileinput
import sys

# read the input
input_blob = ""
print("Input file:", sys.argv[1])
output_file = "outputs/output" + sys.argv[1].replace("inputs/input", "")
print("Output file:", output_file)
for line in fileinput.input():
    input_blob += line


## WILL THIS REMAIN?
# new additionsS
# modifico.


# Clear the output file before writing anything
with open(output_file, "w") as f:
    pass

oneel.write_and_print(output_file,"**************************************")

oneel.write_and_print(output_file, "INPUT FILE CONTENT:")
oneel.write_and_print(output_file,input_blob)
oneel.write_and_print(output_file,"**************************************")
print()
exec(input_blob)

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
#############################START WITH CALCULATION###################################

if two_components:
    if not four_el:
        Weyl_Spinor = orb2c.orbital2c()
        Weyl_Spinor = sg.make_NR_starting_guess(position, charge, mra, prec, comp = 2)

        Weyl_Spinors = oneel.gs_Weyl_1e(Weyl_Spinor, V_tree, mra, prec, thr, derivative, charge, output_file)
        print()
    else:
        #spinorb_array = fourel.init_4_spinors(position, charge, mra, prec)
        spinorb_array = [orb2c.orbital2c() for i in range(4)]
        for i in range(4):
            name = f"W_spinor{i}"
            spinorb_array[i].read(name)

        en_guess_array = []
        en_guess_array.append(fourel.analytic_1s(light_speed, 1, -1, charge))
        en_guess_array.append(fourel.analytic_1s(light_speed, 1, -1, charge))
        en_guess_array.append(fourel.analytic_1s(light_speed, 2, -1, charge))
        en_guess_array.append(fourel.analytic_1s(light_speed, 2, -1, charge))

        print("Initial energy guesses:")
        for i in range(4):
            print(f"Spinor {i+1} energy guess: {en_guess_array[i]-light_speed**2} Ha, {en_guess_array[i]} a.u.")

        #spinorb_array = fourel.Lowdin_orthonormalize_4c(spinorb_array, mra, prec)
        spinorb_array, en_guess_array = fourel.scf_4el_cheat(spinorb_array, en_guess_array, V_tree, mra, prec, thr, derivative, output_file)
        
        
        #spinorb_array, en_guess_array = fourel.gs_Weyl_4e(spinorb_array, en_guess_array, V_tree, mra, prec, thr, derivative, charge, output_file, 1)
        
        print()
        print("NEW component norms:")
        for i in range(4):
            print(f"Spinor {i+1} norm:")
            orb2c.print_norm_debug(spinorb_array[i])    
        
        print()
        print("Energies:")
        for i in range(4):
            print(f"Spinor {i+1} energy: {en_guess_array[i]-light_speed**2} Ha")



else:
    spinorb1 = orb.orbital4c()
    spinorb2 = orb.orbital4c()
    if readOrbitals:
        orbitalName = "spinorb1"
        spinorb1.read(orbitalName)
    else:
        print("Generating starting guess...")
        spinorb1 = sg.make_NR_starting_guess(position, charge, mra, prec)

        print("NEW component norms:")
        orb.print_norm_debug(spinorb1)
    spinorb2 = spinorb1.ktrs(prec)
    if (two_electrons):
        print("Generating second orbital by KTRS...")
        print("NEW component norms:")
        orb.print_norm_debug(spinorb2)
        print("Int overlap between orbitals:", spinorb1.dot(spinorb2).real)

    if saveGuess:
        spinorb1.save("guess1")

    run_D_1e       = scf and not D2 and not two_electrons
    run_D2_1e      = scf and     D2 and not two_electrons
    run_D_2e       = scf and not D2 and     two_electrons and not ktrs
    run_D2_2e      = scf and     D2 and     two_electrons and not ktrs
    run_D_2e_ktrs  = scf and not D2 and     two_electrons and     ktrs
    run_D2_2e_ktrs = scf and     D2 and     two_electrons and     ktrs



    if run_D_1e:
        spinorb1 = oneel.gs_D_1e(spinorb1, V_tree, mra, prec, thr, derivative, charge, output_file)

    if run_D2_1e:
        spinorb1 = oneel.gs_D2_1e(spinorb1, V_tree, mra, prec, thr, derivative, charge, output_file)

    if run_D_2e:
        print("NOT PROPERLY TESTED")
        exit(-1)
        spinorb1, spinorb2 = twoel.coulomb_gs_gen([spinorb1, spinorb2], V_tree, mra, prec, derivative)

    if run_D2_2e:
        print("NOT PROPERLY TESTED")
        exit(-1)
        spinorb1, spinorb2 = twoel.coulomb_2e_D2([spinorb1, spinorb2], V_tree, mra, prec, derivative)

    if run_D_2e_ktrs:
        spinorb1, spinorb2 = twoel.coulomb_gs_2e(spinorb1, V_tree, mra, prec, thr, derivative, output_file)

    if run_D2_2e_ktrs:
        spinorb1, spinorb2 = twoel.coulomb_2e_D2_J([spinorb1, spinorb2], V_tree, mra, prec, thr, derivative, output_file)

    if runGaunt:
        twoel.calcGauntPert(spinorb1, spinorb2, mra, prec)

    if runGaugeA:
        twoel.calcGaugePertA(spinorb1, spinorb2, mra, prec)

    if runGaugeB:
        twoel.calcGaugePertB(spinorb1, spinorb2, mra, prec)

    if runGaugeC:
        twoel.calcGaugePertC(spinorb1, spinorb2, mra, prec)

    if runGaugeD:
        twoel.calcGaugePertD(spinorb1, spinorb2, mra, prec)

    if runGaugeDelta:
        twoel.calcGaugeDelta(spinorb1, spinorb2, mra, prec)

    if saveOrbitals:
        spinorb1.save("spinorb1")



oneel.write_and_print(output_file, "")

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
