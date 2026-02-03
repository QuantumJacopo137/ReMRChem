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

def gs_D_1e(spinorb1, potential, mra, prec, thr, derivative, charge,  output_file="output",niter=1):
    
    error_norm = 1

    light_speed = spinorb1.light_speed
    c2 = light_speed**2
    old_energy = 0
    delta_e = 1
    idx = 0
    while (idx < niter and (delta_e > prec/10 or error_norm > thr)):
        print()
        print('$ Iteration', idx)
        
        #print('Norms:')
        #print('Large Norm:', np.sqrt(spinorb1.squaredLargeNorm()))
        #print('Small Norm:', np.sqrt(spinorb1.squaredSmallNorm()))
        hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
        v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
        add_psi = hd_psi + v_psi
        energy = spinorb1.dot(add_psi).real
        print('     Tot - Energy',energy )
        print('     Ele - Energy', energy - light_speed**2)
        mu = orb.calc_dirac_mu(energy, light_speed)
        tmp = orb.apply_helmholtz(v_psi, mu, prec)
#       tmp = orb.apply_dirac_hamiltonian(v_psi, prec, energy, der = derivative)
        tmp.cropLargeSmall(prec)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy, der = derivative)

#        new_orbital =  orb.apply_helmholtz(tmp, mu, prec)
        if(idx > 10):
            new_orbital = new_orbital + spinorb1
        new_orbital.cropLargeSmall(prec)
        new_orbital.normalize()

        print("NEW NORMS:")
        orb.print_norm_debug(new_orbital)
        delta_psi = new_orbital - spinorb1
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        #print('Error', error_norm)
        delta_e = np.abs(energy - old_energy)
        #print('Delta E', delta_e)
        
        old_energy = energy
        spinorb1 = new_orbital
        print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec/10)
        idx += 1
        #print(new_orbital)
    
    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    add_psi = hd_psi + v_psi
    energy = spinorb1.dot(add_psi).real
    energy_1s = analytic_1s(light_speed, 1, -1, charge)

    beta_v_psi = v_psi.beta2()
    ap_psi = spinorb1.alpha_p(prec, derivative)
    cke = spinorb1.classicT()
    psi_beta_v_vpsi = spinorb1.dot(beta_v_psi).real
    psi_ap_V_psi = ap_psi.dot(v_psi).real
    psi_V2_psi = v_psi.dot(v_psi).real
    cpe = psi_beta_v_vpsi + psi_ap_V_psi/light_speed + 0.5 * psi_V2_psi / c2
    classic_energy = cke + cpe
    
    #printing_string = f"Classic-like energies: cke = {cke}, cpe = {cpe}, cke + cpe = {classic_energy}"
    #write_and_print(output_file,printing_string)
    energy_kutzelnigg = c2*(np.sqrt(1+2*classic_energy/c2)-1)

    print()
    print() 
    printing_string = f"Exact Energy = {energy_1s - c2}"
    write_and_print(output_file,printing_string)

    printing_string = f"Dirac Energy = {energy - c2}"
    write_and_print(output_file,printing_string)
    printing_string = f"Kutze Energy = {energy_kutzelnigg}"
    write_and_print(output_file,printing_string)
    printing_string = f"Error Kutze  = {energy_kutzelnigg - energy_1s + light_speed**2}"
    write_and_print(output_file,printing_string)
    printing_string = f"Error Dirac  = {energy - energy_1s}"
    write_and_print(output_file,printing_string)
    printing_string = f"Delta Energy = {energy - old_energy}"
    write_and_print(output_file,printing_string)
    write_and_print(output_file,f'Dirac - Kutzelnigg = {energy - energy_kutzelnigg - light_speed**2}')


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
    #old_energy = -1.000000005995389 + light_speed**2
    old_energy = analytic_1s(light_speed, 1, -1, charge) # usiong a reasonable starting energy, as we are dealing with core el
    #old_energy = -162.70443328635884 + c2
    delta_e = 1
    idx = 0
    Weyl_R = orb2c.orbital2c()
    while (idx < niter and (delta_e > prec/10 or error_norm > thr)):
        print()
        print('$ Iteration', idx)
        # Check that is normalized
        Weyl_L.normalize()
        print("Weyl_L NORMS:")
        orb2c.print_norm_debug(Weyl_L)





        # STARTING BY COMPUTING THE CONVOLUTION TERM
        print(">Computing Big V Psi")
        V_Psi = orb2c.apply_potential(-1.0, potential, Weyl_L, prec)
        V_Psi_term = - 2 *old_energy * V_Psi
        print(" V_Psi_term")
        orb2c.print_norm_debug((1/c2)*V_Psi_term)
                
        VV_term = orb2c.apply_potential(-1.0, potential, V_Psi, prec)
        print(" VV_term")
        orb2c.print_norm_debug((1/c2)*VV_term)

        Helicity_L = Weyl_L.sigma_p(prec, derivative)
        Comm_term_AB = orb2c.apply_potential(-1.0, potential, Helicity_L, prec)
        Comm_term_BA = V_Psi.sigma_p(prec, derivative)
        Comm_term =  Comm_term_AB -  Comm_term_BA
        print(" Comm_term")
        orb2c.print_norm_debug((1/light_speed)*Comm_term)


        Big_V_Psi = (1/c2) * VV_term + (1/c2) * V_Psi_term + (1/light_speed) * Comm_term
        
        print(" Big_V_Psi")
        orb2c.print_norm_debug(Big_V_Psi)


        # CALCULATE THE NEXT STEP L-SPINOR
        mu_old = orb2c.calc_dirac_mu(old_energy, light_speed)
        # CONVOLUTE
        New_Weyl_L = orb2c.apply_helmholtz(Big_V_Psi, mu_old, prec)
        print("NEW NORMS:")
        orb2c.print_norm_debug(New_Weyl_L)

        

        #print("Potential")
        #orb2c.print_norm_debug(V_Psi)

        # CALCULATE THE R SPINOR
        Weyl_R = New_Weyl_L.Restricted_Kinetic_Balance(V_Psi, old_energy, derivative, prec)

        print("New_Weyl_R NORMS:")
        orb2c.print_norm_debug(New_Weyl_L)
        print("Weyl_R NORMS:")
        orb2c.print_norm_debug(Weyl_R)



        # CALCULATE NEW ENERGY
        print("CALCULATING ENERGY...")
        energy = orb2c.calc_energy_Weyl_2c(New_Weyl_L, Weyl_R,potential,prec)
        print()
        print("ENERGY VALUES:")
        print('     Tot - Energy', energy)
        print('     Ele - Energy', energy - c2)

        diff_psi =(1/New_Weyl_L.norm()) * New_Weyl_L - Weyl_L
        deltasq = diff_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        delta_e = np.abs(energy - old_energy)
        Weyl_L = New_Weyl_L

        print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec/10)


        print("New_Weyl_R NORMS:")
        orb2c.print_norm_debug(New_Weyl_L)
        print("Weyl_R NORMS:")
        orb2c.print_norm_debug(Weyl_R)
        print()

        mu_new = orb2c.calc_dirac_mu(energy, light_speed)
        mu_ratio = mu_new / mu_old

        print('     Mu old  : ', mu_old)
        print('     Mu new  : ', mu_new)
        print('     Mu ratio: ', mu_ratio)
        old_energy = energy


        idx += 1
        print("-----------------------------------")
        print()

    
    spinorb1 = orb.orbital4c()
    #Weyl_R = Weyl_L.Restricted_Kinetic_Balance(V_Psi, -0.4830400869541336 + c2, derivative, prec, L_to_R = True)
    spinorb1.copy_components(La = New_Weyl_L['alpha'])
    spinorb1.copy_components(Lb = New_Weyl_L['beta'])
    spinorb1.copy_components(Sa = Weyl_R['alpha'])
    spinorb1.copy_components(Sb = Weyl_R['beta'])
    spinorb1.normalize()


    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    add_psi = hd_psi + v_psi

    kin_ev = spinorb1.dot(hd_psi).real
    print("Kinetic Energy Contribution:", kin_ev)
    print("Kinetic Energy Contribution -mc^2:", kin_ev - c2)
    print("Potential Energy Contribution:", spinorb1.dot(v_psi).real)
    print("Numerator Energy Contribution:", spinorb1.dot(add_psi).real - c2)


    energy = spinorb1.dot(add_psi).real
    energy_1s = analytic_1s(light_speed, 1, -1, charge)

    beta_v_psi = v_psi.beta2()
    ap_psi = spinorb1.alpha_p(prec, derivative)
    cke = spinorb1.classicT()
    psi_beta_v_vpsi = spinorb1.dot(beta_v_psi).real
    psi_ap_V_psi = ap_psi.dot(v_psi).real
    psi_V2_psi = v_psi.dot(v_psi).real
    cpe = psi_beta_v_vpsi + psi_ap_V_psi/light_speed + 0.5 * psi_V2_psi / c2
    classic_energy = cke + cpe
    
    #printing_string = f"Classic-like energies: cke = {cke}, cpe = {cpe}, cke + cpe = {classic_energy}"
    #write_and_print(output_file,printing_string)
    energy_kutzelnigg = c2*(np.sqrt(1+2*classic_energy/c2)-1)

    print()
    print() 
    printing_string = f"Exact Energy = {energy_1s - c2}"
    write_and_print(output_file,printing_string)

    printing_string = f"Dirac Energy = {energy - c2}"
    write_and_print(output_file,printing_string)
    printing_string = f"Kutze Energy = {energy_kutzelnigg}"
    write_and_print(output_file,printing_string)
    printing_string = f"Error Kutze  = {energy_kutzelnigg - energy_1s + light_speed**2}"
    write_and_print(output_file,printing_string)
    printing_string = f"Error Dirac  = {energy - energy_1s}"
    write_and_print(output_file,printing_string)
    printing_string = f"Delta Energy = {energy - old_energy}"
    write_and_print(output_file,printing_string)
    write_and_print(output_file,f'Dirac - Kutzelnigg = {energy - energy_kutzelnigg - light_speed**2}')


    return spinorb1
