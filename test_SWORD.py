########## Define Enviroment #################
import time
from scipy.differentiate import derivative
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import orbital_2c as orb2c
from orbital4c import nuclear_potential as nucpot
from orbital4c import r3m as r3m
import plotter as plt
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
import starting_guess as sg
import Lazy_4_el as lazy4el

import importlib
importlib.reload(orb)

import fileinput
import sys


light_speed = 137.03599913900001           # default 137.03599913900001
derivative = "BS"                 # possible values: PH, ABGV, BS
prec = 1.0e-8                       # precision threshold for the MRA operations
thr  = prec*10                   # error threshold for SCF convergence
order = int(-np.log10(prec)) +4                           # max order of the polynomials in the MRA
box = 15                            # size of the box in atomic units (half box length)     
auto_box = True

readPotential    = True             # bool
computePotential = True            # bool 
savePotential    = True             # bool
potential = "fermi_dirac"         # possible values: point_charge coulomb_HFYGB homogeneus_charge_sphere gaussian

continue_run     = False            # bool
readOrbitals     = False            # bool
saveOrbitals     = True             # bool
saveGuess        = False

SWORD_Method     = False             # bool
one_electron     = True             # boolv


molecule = [ ["Hg", 80, 0.0, 0.0, 0.0, 0.0 , 0] ]

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
    elif(potential == "homogeneus_charge_sphere"):
        print("homogeneus charge sphere potential")
        f = lambda x: nucpot.homogeneus_charge_sphere_1973(x, position, charge, radius)
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









#start_guess = sg.make_NR_starting_guess(position, charge, mra, prec)
#Weyl_method = oneel.gs_SWORD_1e(start_guess, V_tree, mra, prec, thr, derivative, charge, niter=20)

#print()
#start_guess = sg.make_NR_starting_guess(position, charge, mra, prec)
#Dirac_method = oneel.gs_D_1e(start_guess, V_tree, mra, prec, thr, derivative, charge, niter=20)


#L_Weyl = orb2c.Dirac_to_Weyl(Weyl_method)
#D_Weyl = orb2c.Dirac_to_Weyl(Weyl_method, False)

#L_Dirac = orb2c.Dirac_to_Weyl(Dirac_method)
#D_Dirac = orb2c.Dirac_to_Weyl(Dirac_method, False)


#norm_rat = D_Dirac.norm()/ D_Weyl.norm()
#L_Weyl = L_Weyl * norm_rat
#D_Weyl = D_Weyl * norm_rat

#diff = orb2c.orbital2c()
#c2 = light_speed**2
#diff = D_Dirac - L_Weyl + (4466.3688191756555/c2)*L_Weyl

#v_psi = orb.apply_potential(-1.0, V_tree, Weyl_method, prec)



#diff = L_Dirac - L_Weyl


#plt.plot_scalar_slice_3D( diff.density(prec), 0.01, 161, title="Difference D-L")
#plt.plot_scalar_slice_3D(diff.density(prec), 0.01, 161, title="Difference D-L (log)", log_scale=True)
#plt.plot_scalar_slice_3D( diff.density(prec), 0.001, 161, title="Difference D-L")
#plt.plot_scalar_slice_3D(diff.density(prec), 0.001, 161, title="Difference D-L (log)", log_scale=True)
#plt.plot_scalar_slice_3D( diff.density(prec), 0.0001, 161, title="Difference D-L")
#plt.plot_scalar_slice_3D(diff.density(prec), 0.0001, 161, title="Difference D-L (log)", log_scale=True)
#diff = diff + (1/(c2)) * orb2c.Dirac_to_Weyl(v_psi)
#diff = D_Dirac-D_Weyl

#plt.plot_scalar_slice_3D( diff.density(prec), 0.01, 161, title="Difference D-L")
#plt.plot_scalar_slice_3D(diff.density(prec), 0.01, 161, title="Difference D-L (log)", log_scale=True)
#plt.plot_scalar_slice_3D( diff.density(prec), 0.001, 161, title="Difference D-L")
#plt.plot_scalar_slice_3D(diff.density(prec), 0.001, 161, title="Difference D-L (log)", log_scale=True)
#plt.plot_scalar_slice_3D( diff.density(prec), 0.0001, 161, title="Difference D-L")
#plt.plot_scalar_slice_3D(diff.density(prec), 0.0001, 161, title="Difference D-L (log)", log_scale=True)

#diff = diff + (1/light_speed) * L_Weyl.sigma_p(prec)

 

#plt.plot_scalar_slice_3D( diff.density(prec), 0.001, 100, title="Difference D-L")
#plt.plot_scalar_slice_3D(diff.density(prec), 0.001, 100, title="Difference D-L (log)", log_scale=True)
#plt.plot_scalar_slice_3D( diff.density(prec), 0.0001, 100, title="Difference D-L")
#plt.plot_scalar_slice_3D(diff.density(prec), 0.0001, 100, title="Difference D-L (log)", log_scale=True)


#Difference = Dirac_method - Weyl_method
#L_diff = orb2c.Dirac_to_Weyl(Difference)
#D_diff = orb2c.Dirac_to_Weyl(Difference, False)
#plt.plot_scalar_slice_3D(L_diff.density(prec), 0.001, 100, title="Difference in L component", vmin=0, vmax=100)
#plt.plot_scalar_slice_3D(D_diff.density(prec), 0.001, 100, title="Difference in D component", vmin=0, vmax=100)


#plt.plot_scalar_slice_3D(L_diff.density(prec), 0.001, 100, title="Difference in L component (log)", log_scale=True)
#plt.plot_scalar_slice_3D(D_diff.density(prec), 0.001, 100, title="Difference in D component (log)", log_scale=True)




#exit(0)
############################ START WITH CALCULATION ##################################

# Use this as a sandbox for testing all the subroutines

# In case, to generate the L spinors here it is:
#spinorb_array = fourel.init_4_spinors(position, charge, mra, prec)

# Otherwise, read them from file



if not one_electron:
    Dirac_array = [orb.orbital4c() for i in range(4)]


    if not continue_run:
        L_array = [orb2c.orbital2c(),
                orb2c.orbital2c(),
                orb2c.orbital2c(),
                orb2c.orbital2c()]

        R_array = [orb2c.orbital2c(),
                orb2c.orbital2c(),
                orb2c.orbital2c(),
                orb2c.orbital2c()]
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





    if SWORD_Method:
        print("***********************************")
        print( " FOUR ELECTRON SWORD CALCULATION:")
        print("***********************************")

        Dirac_array, Fock_matrix =lazy4el.scf_4el(Dirac_array, V_tree, mra, prec/10, auto_save=False)
    else:   
        print("***********************************")
        print( " FOUR ELECTRON DIRAC CALCULATION:")
        print("***********************************")
        Dirac_array, Fock_matrix = lazy4el.scf_4e_4c(Dirac_array, V_tree, mra, prec, auto_save=False)

    for i in range(4):
        Dirac_array[i].save(f"Last_run_spinor_{i}")
else:
    

    if SWORD_Method:
        print("***********************************")
        print( " ONE ELECTRON SWORD CALCULATION:")
        print("***********************************")

        
        spinorb1 = orb.orbital4c()
        if not readOrbitals:
            #spinorb1 = sg.make_NR_starting_guess_with_pot(position, charge, mra, prec/10, V_tree)
            spinorb1 = sg.make_NR_starting_guess(position, charge, mra, prec)
            init_guess = spinorb1
            #spinorb1 = sg.make_NR_starting_guess(position, charge, mra, prec/10)
        else :
            spinorb1.read(f"Last_run_spinor_1el")
            print("Read spinor from file: Last_run_spinor_1el")
        
        spinorb1.normalize()
        output_file = "output_SWORD_1el"
        
        output_file = output_file +"_last.txt"
        print("output file: ", output_file)
        print()
        start_time = time.time()
        spinorb1 = oneel.gs_D_1e(init_guess, V_tree, mra, 1e-3, thr*10, derivative, charge,niter=2)
        spinorb1 = oneel.gs_SWORD_1e(spinorb1, V_tree, mra, prec, thr, derivative, charge, output_file)
        
        
      
    elif not SWORD_Method:
        print("***********************************")
        print( " ONE ELECTRON DIRAC CALCULATION:")
        print("***********************************")

        spinorb1 = orb.orbital4c()
        
        if not readOrbitals:
            init_guess = sg.make_NR_starting_guess(position, charge, mra, prec)
        else :
            spinorb1.read(f"Last_run_spinor_1el")
            print("Read spinor from file: Last_run_spinor_1el")
        spinorb1.normalize()
        output_file = "output_Dirac_1el"
        #output_file = output_file + str(molecule[0][1]) + str(int(-np.log10(prec))) + ".txt"
        output_file = output_file +"_last.txt"
        start_time = time.time()
        spinorb1 = oneel.gs_D_1e(init_guess, V_tree, mra, 1e-3, thr*10, derivative, charge,niter=2)
        spinorb1 = oneel.gs_D_1e(spinorb1, V_tree, mra, prec, thr, derivative, charge, output_file)


    if saveOrbitals:
        spinorb1.save(f"Last_run_spinor_1el")

   


 



    






# ========== END OF CALCULATION, PRINTING RESULTS ================
output_file = "output_test2.txt"
oneel.write_and_print(output_file, "")
print("********************************")

if SWORD_Method:
    oneel.write_and_print(output_file, "-> 2 COMPONENTS CALCULATION ")
else:
    oneel.write_and_print(output_file, "-> 4 COMPONENTS CALCULATION ")      

oneel.write_and_print(output_file, "PARAMETERS:")
oneel.write_and_print(output_file, f"molecule    = {molecule}")
oneel.write_and_print(output_file, f"SWORD       = {SWORD_Method}")
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
oneel.write_and_print(output_file, f"{molecule[0][0]} {int(centerd)} {int(SWORD_Method)} {-int(np.log10(prec))} {derivative} {box}")
oneel.write_and_print(output_file, "-------------------------------")
print ("*****************************************")
print ("-> TIME ELAPSED: %s seconds" % (time.time() - start_time))
print ("*****************************************")

