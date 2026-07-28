from ast import mod
import time
from Four_el import F_matrix
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import orbital_2c as orb2c
from orbital4c import operators as oper
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from vampyr import vampyr3d as vp
import numpy as np
import numpy.linalg as LA
import sys, getopt
import plotter as plt

def write_and_print(of, text):
    print(text)
    with open(of, "a") as f:
        f.write(text + "\n")

def analytic_1s(light_speed, n, k, Z):
    alpha = 1/light_speed
    gamma = orb.compute_gamma(k,Z,alpha)
    tmp1 = n - np.abs(k) + gamma
    tmp2 = Z * alpha / tmp1
    tmp3 = 1 + tmp2**2
    return light_speed**2 / np.sqrt(tmp3)

def gs_D_1e(spinorb1, potential, mra, prec, thr, derivative, charge,  output_file="output",niter=100):
    
    error_norm = 1

    light_speed = spinorb1.light_speed
    c2 = light_speed**2
    old_energy = 0
    delta_e = 1
    idx = 0
    while (idx < niter and (delta_e > prec/10 or error_norm > thr)):
        print()
        spinorb1.crop(prec)
        print('$ Iteration', idx)
        
        print("\n-> Calculating Energy...")
        start_energy = time.time()
        
        hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
        v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
        add_psi = hd_psi + v_psi
        add_psi.crop(prec)
        energy = spinorb1.dot(add_psi).real
        print('     Tot - Energy',energy )
        print('     Ele - Energy', energy - light_speed**2)
        end_energy = time.time()
        print(f"Energy calculation time: {end_energy - start_energy:.4f} seconds\n")

        print("\n-> Propagating...")
        start_propagation = time.time()
        mu = orb.calc_dirac_mu(energy, light_speed)

        
        

        tmp1 = orb.apply_dirac_hamiltonian(v_psi, prec, energy, der = derivative)
        tmp = orb.apply_helmholtz(tmp1, mu, prec)
        #tmp = orb.apply_helmholtz(v_psi, mu, prec)
#       tmp = orb.apply_dirac_hamiltonian(v_psi, prec, energy, der = derivative)
        tmp.crop(prec)

        #print("-> Applying Dirac Hamiltonian...")
        #new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy, der = derivative)
        new_orbital = tmp
        new_orbital.normalize()
        end_propagation = time.time()
        print(f"Propagation time: {end_propagation - start_propagation:.4f} seconds\n")

        #print("-> Damping...")
        #start_damping = time.time()
        #if(idx > 10):
        #    new_orbital = new_orbital + spinorb1
        #new_orbital.crop(prec)
        #new_orbital.normalize()
        #end_damping = time.time()
        #print(f"Damping time: {end_damping - start_damping:.4f} seconds\n")

        
        delta_psi = new_orbital - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)

        delta_e = np.abs((energy - old_energy) / (energy-c2))

        old_energy = energy
        spinorb1 = new_orbital
        print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec)
        if (error_norm < thr or idx == niter or delta_e < prec/10 ):
            print("Converged!")
            break
        idx += 1
        
    
    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    add_psi = hd_psi + v_psi
    energy = spinorb1.dot(add_psi).real
    energy_1s = analytic_1s(light_speed, 1, -1, charge)

    #beta_v_psi = v_psi.beta2()
    #ap_psi = spinorb1.alpha_p(prec, derivative)
    #cke = spinorb1.classicT()
    #psi_beta_v_vpsi = spinorb1.dot(beta_v_psi).real
    #psi_ap_V_psi = ap_psi.dot(v_psi).real
    #psi_V2_psi = v_psi.dot(v_psi).real
    #cpe = psi_beta_v_vpsi + psi_ap_V_psi/light_speed + 0.5 * psi_V2_psi / c2
    #classic_energy = cke + cpe
    
    #energy_kutzelnigg = c2*(np.sqrt(1+2*classic_energy/c2)-1)

    print()
    print() 
    printing_string = f"Exact Energy = {energy_1s - c2}"
    write_and_print(output_file,printing_string)

    printing_string = f"Dirac Energy = {energy - c2}"
    write_and_print(output_file,printing_string)
    #printing_string = f"Kutze Energy = {energy_kutzelnigg}"
    #write_and_print(output_file,printing_string)
    #printing_string = f"Error Kutze  = {energy_kutzelnigg - energy_1s + light_speed**2}"
    #write_and_print(output_file,printing_string)
    printing_string = f"Error Dirac  = {energy - energy_1s}"
    write_and_print(output_file,printing_string)
    printing_string = f"Delta Energy = {energy - old_energy}"
    write_and_print(output_file,printing_string)
    #write_and_print(output_file,f'Dirac - Kutzelnigg = {energy - energy_kutzelnigg - light_speed**2}')


    return spinorb1


def gs_D2_1e(spinorb1, potential, mra, prec, thr, derivative, charge, output_file="output", niter=100):
    error_norm = 1
    delta_e = 1
    light_speed = spinorb1.light_speed
    c2 = light_speed * light_speed
    old_energy = 0
    idx = 0
    while (idx < niter and (delta_e > prec/10 or error_norm > thr)):
        print("$ Iteration ", idx )
        v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec) 
        vv_psi = orb.apply_potential(-0.5/c2, potential, v_psi, prec)
        beta_v_psi = v_psi.beta2()
        apV_psi = v_psi.alpha_p(prec, derivative)
        ap_psi = spinorb1.alpha_p(prec, derivative)
        Vap_psi = orb.apply_potential(-1.0, potential, ap_psi, prec)
        anticom = apV_psi + Vap_psi
#        anticom.cropLargeSmall(prec)
#        beta_v_psi.cropLargeSmall(prec)
#        vv_psi.cropLargeSmall(prec)
        RHS = beta_v_psi + vv_psi + anticom * (0.5/light_speed)
        RHS.cropLargeSmall(prec)
        cke = spinorb1.classicT()
        cpe = (spinorb1.dot(RHS)).real
        #print("Classic-like energies:", "cke =", cke,"cpe =", cpe,"cke + cpe =", cke + cpe)
        classic_energy = cke + cpe
        energy = c2*(np.sqrt(1+2*classic_energy/c2)-1)
        mu = orb.calc_non_rel_mu(cke+cpe)
        new_orbital = orb.apply_helmholtz(RHS, mu, prec)
        #if(idx > 10):
        if(idx > 3):
            new_orbital = 2 * new_orbital 
            new_orbital =  new_orbital + spinorb1
        new_orbital.cropLargeSmall(prec)
        new_orbital.normalize()
        delta_psi = new_orbital - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        #print("Error =", error_norm)
        delta_e = np.abs(energy - old_energy)
        #print('Delta E', delta_e)
        print('     Energy',energy)
        old_energy = energy
        spinorb1 = new_orbital 
        print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec/10)
        idx += 1
        #print(new_orbital)
        spinorb1.save("spinorb1")
    
    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    add_psi = hd_psi + v_psi
    energy_dirac = spinorb1.dot(add_psi).real
    
    beta_v_psi = v_psi.beta2()
    ap_psi = spinorb1.alpha_p(prec, derivative)
    
    cke = spinorb1.classicT()
    psi_beta_v_vpsi = spinorb1.dot(beta_v_psi).real
    psi_ap_V_psi = ap_psi.dot(v_psi).real
    psi_V2_psi = v_psi.dot(v_psi).real
    cpe = psi_beta_v_vpsi + psi_ap_V_psi/light_speed + 0.5 * psi_V2_psi / c2
    classic_energy = cke + cpe
    energy = c2*(np.sqrt(1+2*classic_energy/c2)-1)
    energy_1s = analytic_1s(light_speed, 1, -1, charge)

    print()
    print()
    printing_string = f'Exact Energy = {energy_1s - light_speed**2}'
    write_and_print(output_file,printing_string)
    printing_string = f'Dirac Energy = {energy_dirac - light_speed**2}'
    write_and_print(output_file,printing_string)
    printing_string = f'Kutze Energy = {energy}'
    write_and_print(output_file,printing_string)
    printing_string = f'Error Kutze  = {energy - energy_1s + light_speed**2}'
    write_and_print(output_file,printing_string)
    write_and_print(output_file, f'Error Dirac  = {energy_dirac - energy_1s}')
    write_and_print(output_file, f'Delta Energy = {energy - old_energy}')
    write_and_print(output_file, f'Dirac - Kutzelnigg = {energy_dirac - energy - light_speed**2}')   
    return spinorb1




def gs_Weyl_1e(Weyl_L, potential, mra, prec, thr, derivative, charge, output_file="output", niter=100):    
    error_norm = 1

    light_speed = Weyl_L.light_speed
    c2 = light_speed**2
    
    old_energy = analytic_1s(light_speed, 1, -1, charge) # usiong a reasonable starting energy, as we are dealing with core el
    delta_e = 1
    idx = 0
    Weyl_R = orb2c.orbital2c()
    New_Weyl_L = orb2c.orbital2c()

    while (idx < niter or ((delta_e >= prec) or (error_norm >= thr))):
        print("===============================")
        print('$ Iteration', idx)
        print("===============================")

        # Check that is normalized
        Weyl_L.normalize()
        #print("Weyl_L NORMS:")
        #orb2c.print_norm_debug(Weyl_L)





        # STARTING BY COMPUTING THE CONVOLUTION TERM
        print("-> Computing Big V Psi...")
        V_Psi = orb2c.apply_potential(-1.0, potential, Weyl_L, prec)
        V_Psi_term = - 2 *old_energy * V_Psi
        #print(" V_Psi_term")
        #orb2c.print_norm_debug((1/c2)*V_Psi_term)
                
        VV_term = orb2c.apply_potential(-1.0, potential, V_Psi, prec)
        #print(" VV_term")
        #orb2c.print_norm_debug((1/c2)*VV_term)

        Helicity_L = Weyl_L.sigma_p(prec, derivative)
        Comm_term_AB = orb2c.apply_potential(-1.0, potential, Helicity_L, prec)
        Comm_term_BA = V_Psi.sigma_p(prec, derivative)
        Comm_term =  Comm_term_AB -  Comm_term_BA
        #print(" Comm_term")
        #orb2c.print_norm_debug((1/light_speed)*Comm_term)


        Big_V_Psi = (1/c2) * VV_term + (1/c2) * V_Psi_term + (1/light_speed) * Comm_term
        
        #print(" Big_V_Psi")
        #orb2c.print_norm_debug(Big_V_Psi)


        # CALCULATE THE NEXT STEP L-SPINOR
        print("-> Propagating...")
        mu_old = orb2c.calc_dirac_mu(old_energy, light_speed)
        # CONVOLUTE
        New_Weyl_L = orb2c.apply_helmholtz(Big_V_Psi, mu_old, prec)
        #print("NEW NORMS:")
        #orb2c.print_norm_debug(New_Weyl_L)

    
        # CALCULATE THE R SPINOR
        print("-> Calculating R spinor...")
        #Weyl_R = New_Weyl_L.Restricted_Kinetic_Balance(V_Psi, old_energy, derivative, prec)
        Weyl_R = New_Weyl_L.apply_R(V_Psi, old_energy, derivative, prec)

        #print("New_Weyl_R NORMS:")
        #orb2c.print_norm_debug(New_Weyl_L)
        #print("Weyl_R NORMS:")
        #orb2c.print_norm_debug(Weyl_R)



        # CALCULATE NEW ENERGY
        print("-> CALCULATING ENERGY...")
        energy = orb2c.calc_energy_Weyl_2c(New_Weyl_L, Weyl_R,potential,prec)
        print()
        print("ENERGY VALUES:")
        print('     Tot - Energy', energy)
        print('     Ele - Energy', energy - c2)

        diff_psi =(1/New_Weyl_L.norm()) * New_Weyl_L - Weyl_L
        deltasq = diff_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        delta_e = np.abs((energy - old_energy)/energy)
        Weyl_L = New_Weyl_L
        print("------------------------------------------------------------------------------------------------")
        print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec)
        print("------------------------------------------------------------------------------------------------")
        print("\n\n\n")    

        if (error_norm < thr):
            print("Converged!")
            break
        elif (idx == niter):
            print("Maximum number of iterations reached!")
            break
        elif (delta_e < prec):
            print("Energy converged!")
            break

        #print("New_Weyl_R NORMS:")
        #orb2c.print_norm_debug(New_Weyl_L)
        #print("Weyl_R NORMS:")
        #orb2c.print_norm_debug(Weyl_R)
        #print()

        #mu_new = orb2c.calc_dirac_mu(energy, light_speed)
        #mu_ratio = mu_new / mu_old

        #print('     Mu old  : ', mu_old)
        #print('     Mu new  : ', mu_new)
        #print('     Mu ratio: ', mu_ratio)
        old_energy = energy
        
        idx += 1

    
    spinorb1 = orb.orbital4c()
    #Weyl_R = Weyl_L.Restricted_Kinetic_Balance(V_Psi, -0.4830400869541336 + c2, derivative, prec, L_to_R = True)
    spinorb1.copy_components(La = New_Weyl_L['alpha'])
    spinorb1.copy_components(Lb = New_Weyl_L['beta'])
    spinorb1.copy_components(Sa = Weyl_R['alpha'])
    spinorb1.copy_components(Sb = Weyl_R['beta'])
    spinorb1.normalize()


    #hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    #v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    #add_psi = hd_psi + v_psi

    #kin_ev = spinorb1.dot(hd_psi).real
    #print("Kinetic Energy Contribution:", kin_ev)
    #print("Kinetic Energy Contribution -mc^2:", kin_ev - c2)
    #print("Potential Energy Contribution:", spinorb1.dot(v_psi).real)
    #print("Numerator Energy Contribution:", spinorb1.dot(add_psi).real - c2)


    #energy = spinorb1.dot(add_psi).real
    energy_1s = analytic_1s(light_speed, 1, -1, charge)

    #beta_v_psi = v_psi.beta2()
    #ap_psi = spinorb1.alpha_p(prec, derivative)
    #cke = spinorb1.classicT()
    #psi_beta_v_vpsi = spinorb1.dot(beta_v_psi).real
    #psi_ap_V_psi = ap_psi.dot(v_psi).real
    #psi_V2_psi = v_psi.dot(v_psi).real
    #cpe = psi_beta_v_vpsi + psi_ap_V_psi/light_speed + 0.5 * psi_V2_psi / c2
    #classic_energy = cke + cpe
    
    #printing_string = f"Classic-like energies: cke = {cke}, cpe = {cpe}, cke + cpe = {classic_energy}"
    #write_and_print(output_file,printing_string)
    #energy_kutzelnigg = c2*(np.sqrt(1+2*classic_energy/c2)-1)

    print()
    print() 
    printing_string = f"Exact Energy = {energy_1s - c2}"
    write_and_print(output_file,printing_string)

    printing_string = f"Converged Energy = {old_energy - c2}"
    write_and_print(output_file,printing_string)
   #printing_string = f"Kutze Energy = {energy_kutzelnigg}"
    #write_and_print(output_file,printing_string)
    #printing_string = f"Error Kutze  = {energy_kutzelnigg - energy_1s + light_speed**2}"
    #write_and_print(output_file,printing_string)
    #printing_string = f"Error Dirac  = {energy - energy_1s}"
    #write_and_print(output_file,printing_string)
    #printing_string = f"Delta Energy = {energy - old_energy}"
    #write_and_print(output_file,printing_string)
    #write_and_print(output_file,f'Dirac - Kutzelnigg = {energy - energy_kutzelnigg - light_speed**2}')


    return spinorb1




def gs_SWORD_1e(spinorb1, potential, mra, prec, thr, derivative, charge,  output_file="output",niter=100):
    
    error_norm = 1

    light_speed = spinorb1.light_speed
    c2 = light_speed**2
    old_energy = 0
    delta_e = 1
    idx = 0

    new_spinor = orb.orbital4c()

    # Triasl funct rel energy
    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    #v_psi.crop(prec)
    print(hd_psi.dot (spinorb1).real, v_psi.dot(spinorb1).real)
    add_psi = hd_psi + v_psi
    energy = spinorb1.dot(add_psi).real
    print('     Tot - Energy',energy )
    print('     Ele - Energy', energy - light_speed**2)
    

    old_energy = energy



    while (idx < niter and (delta_e > prec/10 or error_norm > thr)):
        print()
        print('$ Iteration', idx)
        
        
        # PROPAGATE
        print("\n-> Propagating...")
        start_prop = time.time()
        print("PRINTING SPINOR :")
        orb.print_norm_debug(spinorb1)
        print("PRINTING V_PSI :")
        orb.print_norm_debug(v_psi)
        
        new_spinor = SWORD_propagator(spinorb1, v_psi, energy, prec)    

        end_prop = time.time()
        print(f"Propagation time: {end_prop - start_prop:.4f} seconds")

        # BALANCE
        print("\n-> Balancing...")
        start_Balance = time.time()
        new_spinor.normalize()
        v_psi = orb.apply_potential(-1.0, potential, new_spinor, prec)
        
        print("PRINTING SPINOR :")
        orb.print_norm_debug(new_spinor)
        print("PRINTING V_PSI :")
        orb.print_norm_debug(v_psi)
        new_spinor = balance_Dirac_spinor(new_spinor, v_psi, energy, prec)
      
        new_spinor.crop(prec)
        new_spinor.normalize()
        end_Balance = time.time()
        print(f"Balancing time: {end_Balance - start_Balance:.4f} seconds")

        # ENERGY
        print("\n-> Calculating Energy...")
        start_energy = time.time()
        hd_psi = orb.apply_dirac_hamiltonian(new_spinor, prec, der = derivative)
        v_psi = orb.apply_potential(-1.0, potential, new_spinor, prec)
        print("PRINTING SPINOR :")
        orb.print_norm_debug(new_spinor)
        print("PRINTING V_PSI :")
        orb.print_norm_debug(v_psi)
        add_psi = hd_psi + v_psi
        energy = new_spinor.dot(add_psi).real
        end_energy = time.time()
        print(f"Energy calculation time: {end_energy - start_energy:.4f} seconds")
        print("\nENERGY VALUES:")
        print('     Tot - Energy',energy )
        print('     Ele - Energy', energy - light_speed**2)
        print("==============================================================")
        

        # CHECK CONVERGENCE
        delta_psi = new_spinor - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        delta_e = np.abs((energy - old_energy)/(energy-c2))


        
        print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec)
        old_energy = energy
        spinorb1 = new_spinor

        print("\n TOT TIME: ", time.time() - start_prop, " seconds, of which:")
        print("   Propagation: ", end_prop - start_prop, " seconds")
        print("   Balancing: ", end_Balance - start_Balance, " seconds")
        print("   Energy: ", end_energy - start_energy, " seconds")

        if (error_norm < thr or idx == niter or delta_e < prec/10):
            print("Converged!")
            break

        idx += 1
    
    
    print()
    print() 
    print("############################################################################")
    energy_1s = analytic_1s(light_speed, 1, -1, charge)
    printing_string = f"Exact Energy = {energy_1s - c2}"
    write_and_print(output_file,printing_string)

    printing_string = f"SWORD Energy = {energy - c2}"
    write_and_print(output_file,printing_string)
    
    printing_string = f"Delta Energy = {energy_1s -energy}"
    write_and_print(output_file,printing_string)


    return spinorb1

def SWORD_propagator(spinor, V_Psi, energy, prec, Propagating_Left = True):
    light_speed = spinor.light_speed
    c2 = light_speed**2
    V_L = orb2c.orbital2c()                 # V * Psi_L
    V_R = orb2c.orbital2c()                 # V * Psi_R
    Big_V_Psi_LR = orb2c.orbital2c()         # the big term in the numerator of the exact propagator, which is given by -c^2 * V * Psi_R + F_ii * V * Psi_L - c * V * sigma_p * Psi_L
    tmp_new_spinor_LR = orb2c.orbital2c()    


    V_L = orb2c.Dirac_to_Weyl(V_Psi)
    V_R = orb2c.Dirac_to_Weyl(V_Psi, weyl_L=False)

    if (Propagating_Left):
        Big_V_Psi_LR = -1.0 *  V_R - (energy/c2) * V_L - (1.0/light_speed) * V_L.sigma_p(prec) # last time i have changed the sign of  F_matrix[i,i].real * V_L
    else:
        Big_V_Psi_LR = -1.0 *  V_L - (energy/c2) * V_R + (1.0/light_speed) * V_R.sigma_p(prec) # last time i have changed the sign of  F_matrix[i,i].real * V_L
    
    
    mu = orb2c.calc_dirac_mu(energy, light_speed)
    tmp_new_spinor_LR = orb2c.apply_helmholtz(Big_V_Psi_LR, mu, prec)
    
   
    Other_comp = orb2c.orbital2c()
    if (Propagating_Left):
        new_dirac_spinor = orb2c.Weyl_to_Dirac(tmp_new_spinor_LR, Other_comp)
    else:
        new_dirac_spinor = orb2c.Weyl_to_Dirac(Other_comp, tmp_new_spinor_LR)
    
    #new_dirac_spinor.normalize()
    new_dirac_spinor.crop(prec)
    
    return new_dirac_spinor

def balance_Dirac_spinor(spinor, V_Psi, energy, prec, calculate_R = True, plot = False):
    # this is gonna take the first 2 components of a general spinor and compute the second 2 components using the balance condition. This is gonna be used in the exact propagator to compute the new spinor array after one iteration of the SCF procedure. The balance condition is given by:
    L_spinor = orb2c.orbital2c()
    R_spinor = orb2c.orbital2c()

    c2 = (spinor.light_speed)**2
    print("BALANCING USING ENERGY:", energy)
    
    if (calculate_R):
        L_spinor = orb2c.Dirac_to_Weyl(spinor)
        R_spinor = L_spinor.apply_R(orb2c.Dirac_to_Weyl(V_Psi), energy, 'ABGV', prec)
    else:
        R_spinor = orb2c.Dirac_to_Weyl(spinor, weyl_L=False)
        L_spinor = R_spinor.apply_R(orb2c.Dirac_to_Weyl(V_Psi, weyl_L=False), energy, 'ABGV', prec, L_to_R = False)

    if plot:
        L_density = L_spinor.density(prec)
        plt.plot_scalar_slice(L_density, 0.15, 300, title="L_density", vmin=-5, vmax=15)
        R_spinor = L_spinor + ((energy-c2)/c2)* L_spinor
        R_density = R_spinor.density(prec)
        plt.plot_scalar_slice(R_density, 0.15, 300, title="L + eps/c2 L",  vmin=-5, vmax=15)
        R_spinor = R_spinor - (1/c2)* orb2c.Dirac_to_Weyl(V_Psi)
        R_density = R_spinor.density(prec)
        plt.plot_scalar_slice(R_density, 0.15, 300, title="L + eps/c2 L - 1/c2 V_Psi",  vmin=-5,vmax=15)    

        R_spinor = R_spinor - (1/spinor.light_speed) * L_spinor.sigma_p(prec/100)
        R_density = R_spinor.density(prec)
        plt.plot_scalar_slice(R_density, 0.15, 300, title="L + eps/c2 L + 1/c2 V_Psi + 1/c sigma_p L",  vmin=-5,vmax=15)


    
    spinor = orb2c.Weyl_to_Dirac(L_spinor, R_spinor)
    print("Spinor after balancing:")
    orb.print_norm_debug(spinor)
    spinor.normalize()
    spinor.crop(prec)

    return spinor

def plot_Dirac_spinor(spinor, prec, title="Dirac Spinor", vmin=-5, vmax=15):
    L_spinor = orb2c.Dirac_to_Weyl(spinor)
    R_spinor = orb2c.Dirac_to_Weyl(spinor, weyl_L=False)
    L_density = L_spinor.density(prec)
    R_density = R_spinor.density(prec)
    plt.plot_scalar_slice(L_density, 0.15, 300, title=title + " - L Density", vmin=vmin, vmax=vmax)
    plt.plot_scalar_slice(R_density, 0.15, 300, title=title + " - R Density", vmin=vmin, vmax=vmax)