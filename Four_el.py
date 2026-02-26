from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import orbital_2c as orb2c
from orbital4c import operators as oper
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
import starting_guess as sg
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

def init_4_spinors(position, charge, mra, prec):
    spinorb_array = []
    for i in range(4):
        spinorb_array.append(orb2c.orbital2c())

    spinorb_array[0] = sg.make_NR_starting_guess(position, charge, mra, prec, comp = 2, n=1, l=0)
    spinorb_array[1] = spinorb_array[0].ktrs(prec)
    spinorb_array[2] = sg.make_NR_starting_guess(position, charge, mra, prec, comp = 2, n=2, l=0)
    spinorb_array[3] = spinorb_array[2].ktrs(prec)

    return spinorb_array

def print_4_orbitals_norm(spinorb_array):
    for i in range(4):
        print(f"Spinor {i+1} norms:")
        orb2c.print_norm_debug(spinorb_array[i])


def guess_Fock_matrix(spinorb_array, potential, mra, prec):
    F_matrix = np.zeros((4,4), dtype=complex)
    light_speed = spinorb_array[0].light_speed
    J_tree, J_Psi_array, K_Psi_array = J_K_Psi_4c(spinorb_array, mra, prec)
    for i in range(4):
        for j in range(4):
            V_psi = orb2c.apply_potential(-1.0, potential, spinorb_array[j], prec)

            FV_Psi = J_Psi_array[j] - K_Psi_array[j] + V_psi
            FV_expv = orb2c.dot(spinorb_array[i], FV_Psi)
            tmp = spinorb_array[j].sigma_p(prec, 'BS')
            T_Psi = tmp.sigma_p(prec, 'BS')
            T_Psi = 0.5 * T_Psi + light_speed**2 * spinorb_array[j]
            T_expv = orb2c.dot(spinorb_array[i], T_Psi)
            F_matrix[i,j] =  T_expv + FV_expv 
            print(f"Fock matrix element F[{i},{j}] = {F_matrix[i,j]}")
    return F_matrix


def compute_S_matrix_4c(spinorb_array, mra, prec):
    S_matrix = np.zeros((4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            S_matrix[i,j] = orb2c.dot(spinorb_array[i], spinorb_array[j])
    return S_matrix

def Lowdin_orthonormalize_4c(spinorb_array, mra, prec):
    S_matrix = compute_S_matrix_4c(spinorb_array, mra, prec)
    eigvals, eigvecs = LA.eig(S_matrix)
    S_inv_sqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ inv(eigvecs)

    orthonormal_spinor_array = []
    for i in range(4):
        new_spinor = orb2c.orbital2c()
        for j in range(4):
            new_spinor = new_spinor + (S_inv_sqrt[j,i]) * spinorb_array[j]
        orthonormal_spinor_array.append(new_spinor)
    return orthonormal_spinor_array


def K_f(spinor, spinorb_array, prec):
    mra = spinorb_array[0].mra
    P = vp.PoissonOperator(mra, prec/10)
    K_f_spinor = orb2c.orbital2c()
    K_f_spinor.setZero()
    
    for i in range(4):
        ol_density = 4 * np.pi * (spinorb_array[i].overlap_density(spinor, prec))
        conv_ol = cf.apply_poisson(ol_density, mra,P, prec)

        K_f_spinor = K_f_spinor + spinorb_array[i].single_func_multiply(conv_ol, prec)

    return K_f_spinor


def J_K_Psi_4c(spinorb_array, mra, prec):
    J_Psi_array = []
    K_Psi_array = []
    for i in range(4):
        J_Psi_array.append(orb2c.orbital2c().setZero())
        K_Psi_array.append(orb2c.orbital2c().setZero())

    contr_1 = spinorb_array[0].density(prec)
    #contr_2 = contr_1      -> if ktrs then they are the same
    contr_3 = spinorb_array[2].density(prec)
    #contr_4 = contr_3      -> if ktrs then they are the same



    # J part
    #rho_total = contr_1 + contr_2 + contr_3 + contr_4
    P = vp.PoissonOperator(mra, prec/10)
    conv_1 = P(contr_1)
    conv_3 = P(contr_3)
    J_tree = 8 *np.pi * (conv_1 +  conv_3)
    

    for i in range(4):
        J_Psi_array[i] = J_tree * spinorb_array[i]


    # K part, diagonal
    K_Psi_array[0] = 4 * np.pi * conv_1 * spinorb_array[0]
    K_Psi_array[1] = 4 * np.pi * conv_1 * spinorb_array[1]
    K_Psi_array[2] = 4 * np.pi * conv_3 * spinorb_array[2]
    K_Psi_array[3] = 4 * np.pi * conv_3 * spinorb_array[3]


    
    # FOR EACH SPINOR IN THE ORBITAL GROUP
    for i in range(4):
        # APPLY K_j \Psi_i
        for j in range(4):
            if i != j:
                ol_density = 4 * np.pi * (spinorb_array[j].overlap_density(spinorb_array[i], prec))
                conv_ol = cf.apply_poisson(ol_density, mra,P, prec)

                K_Psi_array[i] = K_Psi_array[i] + spinorb_array[j].single_func_multiply(conv_ol, prec)


    return J_tree, J_Psi_array, K_Psi_array


def gs_Weyl_4e(spinor_array, energy_guess_array, potential, mra, prec, thr, derivative, charge, output_file="output", niter=100):    
    error_norm = 1

    light_speed = spinor_array[0].light_speed
    c2 = light_speed**2
    

    new_spinor_array = [
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c()]
    
    new_energy_array = [0.0, 0.0, 0.0, 0.0]

    R_Psi_array = [
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c()]

    delta_e = 1
    idx = 0



    Fock_matrix = guess_Fock_matrix(spinor_array, potential, mra, prec)
    print("F matrix:")
    # Nicely print F_matrix to terminal
    print(f"F matrix shape: {Fock_matrix.shape}")
    if Fock_matrix.size == 0:
        print("F matrix is empty")
    else:
        if np.iscomplexobj(Fock_matrix):
            for i, row in enumerate(Fock_matrix):
                row_str = "  ".join(f"({el.real:+.6e} , {el.imag:+.6e}j)" for el in row)
                print(f"[{i:2d}] {row_str}")
        else:
            for i, row in enumerate(Fock_matrix):
                row_str = "  ".join(f"{el:+.10e}" for el in row)
                print(f"[{i:2d}] {row_str}")


    # start the SCF loop
    while (idx < niter and (delta_e > prec/10 or error_norm > thr)):
        print()
        print('$ Iteration', idx)
        # 1) CHECK THAT IS NORMALIZED
        print("Weyl_L NORMS:")
        for i in range(4):
            print(f"Spinor {i+1} norm:")
            orb2c.print_norm_debug(spinor_array[i])


        # 2) COMPUTE J AND K TERMS, as well as the multiplicative J_tree potential
        print("> Computing J and K terms")
        J_tree, J_Psi_array, K_Psi_array = J_K_Psi_4c(spinor_array, mra, prec)
        print("     J and K norms:")
        for i in range(4):
            print(f"       J_Psi {i+1} norm:")
            orb2c.print_norm_debug(J_Psi_array[i])
            print(f"       K_Psi {i+1} norm:")
            orb2c.print_norm_debug(K_Psi_array[i])
        print()


        
        # 3) PROPAGATE EACH ELECTRON, as well as the ktrs partner
        print("\033[91m> PROPAGATING ELECTRON 1 and 3\033[0m")
        new_spinor_array[0], new_energy_array[0] = propagate_Weyl_1e(spinor_array, energy_guess_array[0], J_tree, K_Psi_array, Fock_matrix, 0, potential, prec, derivative, light_speed)
        print("\033[91m> PROPAGATING ELECTRON 2 and 4\033[0m")
        new_spinor_array[2], new_energy_array[2] = propagate_Weyl_1e(spinor_array, energy_guess_array[2], J_tree, K_Psi_array, Fock_matrix, 2, potential, prec, derivative, light_speed)
        

        # 4) KTRS PARTNERS
        new_spinor_array[1] = new_spinor_array[0].ktrs(prec)
        new_spinor_array[3] = new_spinor_array[2].ktrs(prec)
        new_energy_array[1] = new_energy_array[0]
        new_energy_array[3] = new_energy_array[2]


        # 5) ORTHONORMALIZE THE SET
        print("> Orthonormalizing the 4 spinors")
        new_spinor_array = Lowdin_orthonormalize_4c(new_spinor_array, mra, prec)

        S_mat = compute_S_matrix_4c(new_spinor_array, mra, prec)
        print(" Orthonormalization S-matrix:")
        print(S_mat)

        print(" NEW Weyl_L NORMS:")
        for i in range(4):
            print(f"Spinor {i+1} norm:")
            orb2c.print_norm_debug(new_spinor_array[i])
        print()


        # CALCULATE DIFFERENCE AND UPDATE
        error_norm = 0
        delta_e = 0
        for i in range(4):
            diff_psi =(1/new_spinor_array[i].norm()) * new_spinor_array[i] - spinor_array[i]
            deltasq = diff_psi.squaredNorm()
            error_norm += np.sqrt(deltasq)
            delta_e += np.abs(new_energy_array[i] - energy_guess_array[i])
            spinor_array[i] = new_spinor_array[i]
            energy_guess_array[i] = new_energy_array[i]

        error_norm = error_norm / 4
        delta_e = delta_e /4

        print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec/10) 
        idx += 1

    new_energy_array[1] = new_energy_array[0]
    new_energy_array[3] = new_energy_array[2]



    return  new_spinor_array, new_energy_array


def propagate_Weyl_1e(spinor_array, energy_guess,J_tree, K_Psi_array, Fock_matrix, el_id, Nuclear_potential, prec, derivative, light_speed):    

    c2 = light_speed**2
    old_energy = energy_guess

    Nuc_el_pot_tree =  J_tree - Nuclear_potential

    # Create a R spinor as a placeholder
    Weyl_R = orb2c.orbital2c()
    print()
    
    # STARTING BY COMPUTING THE CONVOLUTION TERM
    V_Psi = orb2c.apply_potential(1.0, Nuc_el_pot_tree, spinor_array[el_id], prec)
    V_Psi = V_Psi - K_Psi_array[el_id]
    V_Psi_term = - 2 * old_energy * (V_Psi)
    print(" V_Psi_term")
    orb2c.print_norm_debug((1/c2)*V_Psi_term)

    VV_term = orb2c.apply_potential(1.0, Nuc_el_pot_tree, V_Psi, prec)
    VV_term = VV_term - K_f(V_Psi, spinor_array, prec)
    print(" VV_term")
    orb2c.print_norm_debug((1/c2)*VV_term)

    Helicity_L = spinor_array[el_id].sigma_p(prec, derivative)
    Comm_term_AB = orb2c.apply_potential(1.0, (-Nuclear_potential + J_tree), Helicity_L, prec)
    Comm_term_AB = Comm_term_AB - K_f(Helicity_L, spinor_array, prec)
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
    #New_Weyl_L = orb2c.apply_helmholtz(Big_V_Psi, mu_old, prec)
    New_Weyl_L = spinor_array[el_id]
    print("NEW NORMS:")
    orb2c.print_norm_debug(New_Weyl_L)

    
    
    # CALCULATE THE R SPINOR
    Weyl_R = New_Weyl_L.apply_R(V_Psi, old_energy, derivative, prec)

    print("New_Weyl_R NORMS:")
    orb2c.print_norm_debug(New_Weyl_L)
    print("Weyl_R NORMS:")
    orb2c.print_norm_debug(Weyl_R)



    # CALCULATE NEW ENERGY
    print("CALCULATING ENERGY...")
    #energy = calc_energy(New_Weyl_L, Weyl_R, spinor_array, Nuc_el_pot_tree, prec)
    energy = calc_energy_N_el(Nuc_el_pot_tree, New_Weyl_L, old_energy, derivative, spinor_array, prec)
    print()
    green = "\033[92m"
    reset = "\033[0m"
    print(f"{green}ENERGY VALUES:\n     Tot - Energy {energy}\n     Ele - Energy {energy - c2}{reset}\n")

    return New_Weyl_L, energy



def Diagonal_Term_Weyl_Hamiltonian(Weyl_Spinor, prec, potential, spinor_array, derivative, chirality_L = True):
    # Will compute V \pm c * sigma . p, depending on the chirality of the Weyl spinor
    light_speed = Weyl_Spinor.light_speed
    VPsi = orb2c.apply_potential(1.0, potential, Weyl_Spinor, prec)
    VPsi = VPsi - K_f(Weyl_Spinor, spinor_array, prec)
    sigma_p_Weyl = Weyl_Spinor.sigma_p(prec, derivative)
    
    if chirality_L:
        result = VPsi + light_speed * sigma_p_Weyl
    else:
        result = VPsi - light_speed * sigma_p_Weyl
    return result

def calc_energy(Psi_L, Psi_R, spinor_array, potential, prec):
    light_speed = Psi_L.light_speed
    print("     Light speed in calc_energy_Weyl_2c:", light_speed)

    braket_LL = Psi_L.squaredNorm()
    print("     braket_LL", braket_LL)

    braket_LR = orb2c.dot(Psi_L,Psi_R).real
    print("     braket_LR", braket_LR)

    braket_RR = Psi_R.squaredNorm()
    print("     braket_RR", braket_RR)

    Dirac_Sq_Norm = braket_LL + braket_RR 

    tmp = Diagonal_Term_Weyl_Hamiltonian(Psi_L, prec, potential, spinor_array, 'BS', chirality_L = True)
    tmp2 = Diagonal_Term_Weyl_Hamiltonian(Psi_R, prec, potential, spinor_array, 'BS', chirality_L = False)

    exp_val_L = orb2c.dot(Psi_L,tmp).real
    print("     exp_val_L", exp_val_L/(Dirac_Sq_Norm))
    exp_val_R = orb2c.dot(Psi_R,tmp2).real
    print("     exp_val_R", exp_val_R/(Dirac_Sq_Norm))

    energy = exp_val_R + exp_val_L + 2 * braket_LR * (light_speed**2)
    print("     energy numerator", energy)
    energy *= (1.0/(Dirac_Sq_Norm))

    VJ_ev = orb2c.dot(Psi_L, orb2c.apply_potential(1.0, potential, Psi_L, prec)).real + orb2c.dot(Psi_R, orb2c.apply_potential(1.0, potential, Psi_R, prec)).real
    VJ_ev *= (1.0/(Dirac_Sq_Norm))
    print("     V + J expectation value:", VJ_ev)
    K_EV =  (orb2c.dot(Psi_L, K_f(Psi_L, spinor_array, prec)).real + orb2c.dot(Psi_R, K_f(Psi_R, spinor_array, prec)).real)




    
    return energy


def propagate_Weyl_1e_cheat(spinor_array, R_spinor_array, energy_array,J_tree, K_Psi_array, Fock_matrix, pi_psi_array, V_psi_array, el_id, Nuclear_potential, prec, derivative, light_speed):    

    c2 = light_speed**2
    
    old_energy = energy_array[el_id]

    Nuc_el_pot_tree =  J_tree - Nuclear_potential

    # Create a R spinor as a placeholder
    print()
    
    # STARTING BY COMPUTING THE CONVOLUTION TERM
    V_Psi = orb2c.apply_potential(1.0, Nuc_el_pot_tree, spinor_array[el_id], prec)
    V_Psi = V_Psi - K_Psi_array[el_id]
    V_Psi_term = - 2 * old_energy * (V_Psi)
    print(" V_Psi_term")
    orb2c.print_norm_debug((1/c2)*V_Psi_term)

    VV_term = orb2c.apply_potential(1.0, Nuc_el_pot_tree, V_Psi, prec)
    VV_term = VV_term - K_f(V_Psi, spinor_array, prec)
    print(" VV_term")
    orb2c.print_norm_debug((1/c2)*VV_term)

    Helicity_L = spinor_array[el_id].sigma_p(prec, derivative)
    Comm_term_Vpi = orb2c.apply_potential(1.0, (-Nuclear_potential + J_tree), Helicity_L, prec)
    Comm_term_Vpi = Comm_term_Vpi - K_f(Helicity_L, spinor_array, prec)
    Comm_term_piV = V_Psi.sigma_p(prec, derivative)
    Comm_term =  Comm_term_Vpi -  Comm_term_piV
    print(" Comm_term")
    orb2c.print_norm_debug((1/light_speed)*Comm_term)


    Big_V_Psi = (1/c2) * VV_term + (1/c2) * V_Psi_term + (1/light_speed) * Comm_term
    
    print(" Big_V_Psi")
    orb2c.print_norm_debug(Big_V_Psi)


    # CALCULATE THE NEXT STEP L-SPINOR
    mu_old = orb2c.calc_dirac_mu(old_energy, light_speed)
    # CONVOLUTE

    L_cont = orb2c.orbital2c()
    R_cont = orb2c.orbital2c()
    for j in range(4):
        if j != el_id:
            L_cont = L_cont + (V_psi_array[j]-energy_array[j]*spinor_array[j] + light_speed * pi_psi_array[j]) * Fock_matrix[el_id,j]
            R_cont = R_cont + Fock_matrix[el_id,j] * light_speed *light_speed * R_spinor_array[j]
            
    tot_to_convolute = Big_V_Psi - L_cont + R_cont 
    New_Weyl_L = orb2c.apply_helmholtz(tot_to_convolute, mu_old, prec)
    #New_Weyl_L = spinor_array[el_id]
    print("NEW NORMS:")
    orb2c.print_norm_debug(New_Weyl_L)

    

    print("New_Weyl_R NORMS:")
    orb2c.print_norm_debug(New_Weyl_L)



    # CALCULATE NEW ENERGY
    #print("CALCULATING ENERGY...")
    #energy = calc_energy(New_Weyl_L, Weyl_R, spinor_array, Nuc_el_pot_tree, prec)
    #energy = calc_energy_N_el(Nuc_el_pot_tree, New_Weyl_L, old_energy, derivative, spinor_array, prec)
   
    return New_Weyl_L



def Simple_2c_SCF(spinor_array, energy_guess_array, potential, mra, prec, thr, derivative, charge, output_file="output", niter=100):
    # This is a simple SCF loop for the 2c Weyl spinor case, without the J and K terms, to test the propagation and energy calculation
    error_norm = 1

    light_speed = spinor_array[0].light_speed
    c2 = light_speed**2
    

    new_spinor_array = [
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c()]
    
    new_energy_array = [0.0, 0.0, 0.0, 0.0]

    pi_psi_array = [
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c()
    ]

    V_Psi_array = [
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c()
    ]

    R_Psi_array = [
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c(),
        orb2c.orbital2c()]



    J_tree, J_Psi_array, K_Psi_array = J_K_Psi_4c(spinor_array, mra, prec, True)
    for i in range(4):
            V_Psi_array[i] = orb2c.apply_potential(-1.0, potential, spinor_array[i], prec)
            V_Psi_array[i] = V_Psi_array[i] - K_Psi_array[i] + J_Psi_array[i]

            pi_psi_array[i] = spinor_array[i].sigma_p(prec, derivative) 
            R_Psi_array[i] = (1./c2) * (-V_Psi_array[i] + energy_guess_array[i] * spinor_array[i] - light_speed * pi_psi_array[i])

    print("     J and K norms:")
    for i in range(4):
        print(f"       J_Psi {i+1} norm:")
        orb2c.print_norm_debug(J_Psi_array[i])
        print(f"       K_Psi {i+1} norm:")
        orb2c.print_norm_debug(K_Psi_array[i])
    print()
    idx = 0 
    while (idx < niter and (delta_e > prec/10 or error_norm > thr)):
            
            print()
            print('$ Iteration', idx)
            # 1) CHECK THAT IS NORMALIZED
            print("Weyl_L NORMS:")
            for i in range(4):
                print(f"Spinor {i+1} norm:")
                orb2c.print_norm_debug(spinor_array[i])
            
          


            # 2) COMPUTE J AND K TERMS, as well as the multiplicative J_tree potential
            print("> Computing J and K terms")


            # Soon after i compute all my (sigma * p) Psi as well as VPsi

            # 4) Calculate Fock Matrix
            Fock_matrix = np.zeros((4,4), dtype=complex)
            Fock_matrix = calc_Fock(spinor_array, V_Psi_array,R_Psi_array, energy_guess_array, derivative, mra, prec)

            
            # 3) PROPAGATE EACH ELECTRON, as well as the ktrs partner
            print("\033[91m> PROPAGATING ELECTRON 1 and 3\033[0m")
            new_spinor_array[0] = propagate_Weyl_1e_cheat(spinor_array, energy_guess_array[0], J_tree, K_Psi_array, Fock_matrix, 0, potential, prec, derivative, light_speed)
            print("\033[91m> PROPAGATING ELECTRON 2 and 4\033[0m")
            new_spinor_array[2] = propagate_Weyl_1e_cheat(spinor_array, energy_guess_array[2], J_tree, K_Psi_array, Fock_matrix, 2, potential, prec, derivative, light_speed)
            

            # 4) KTRS PARTNERS
            new_spinor_array[1] = new_spinor_array[0].ktrs(prec)
            new_spinor_array[3] = new_spinor_array[2].ktrs(prec)

            # 4.5) CALCULATE NEW ENERGIES
            J_tree, J_Psi_array, K_Psi_array = J_K_Psi_4c(new_spinor_array, mra, prec, False)
            for i in range(4):
                V_Psi_array[i] = orb2c.apply_potential(-1.0, potential, new_spinor_array[i], prec)
                V_Psi_array[i] = V_Psi_array[i] - K_Psi_array[i] + J_Psi_array[i]

                pi_psi_array[i] = new_spinor_array[i].sigma_p(prec, derivative) 
                R_Psi_array[i] = (1./c2) * (-V_Psi_array[i] + energy_guess_array[i] * new_spinor_array[i] - light_speed * pi_psi_array[i])
                new_energy_array[i] = calc_energy_N_el(potential, new_spinor_array[i], energy_guess_array[i], derivative, spinor_array, prec)
                print(f"New energy for spinor {i+1}: {new_energy_array[i]}")






            # 5) ORTHONORMALIZE THE SET
            print("> Orthonormalizing the 4 spinors")
            new_spinor_array = Lowdin_orthonormalize_4c(new_spinor_array, mra, prec)

            S_mat = compute_S_matrix_4c(new_spinor_array, mra, prec)
            print(" Orthonormalization S-matrix:")
            print(S_mat)

            print(" NEW Weyl_L NORMS:")
            for i in range(4):
                print(f"Spinor {i+1} norm:")
                orb2c.print_norm_debug(new_spinor_array[i])
            print()


            # CALCULATE DIFFERENCE AND UPDATE
            error_norm = 0
            delta_e = 0
            for i in range(4):
                diff_psi =(1/new_spinor_array[i].norm()) * new_spinor_array[i] - spinor_array[i]
                deltasq = diff_psi.squaredNorm()
                error_norm += np.sqrt(deltasq)
                delta_e += np.abs(new_energy_array[i] - energy_guess_array[i])
                spinor_array[i] = new_spinor_array[i]
                energy_guess_array[i] = new_energy_array[i]

            error_norm = error_norm / 4
            delta_e = delta_e /4

            print('     Converged? ', error_norm, ' > ', thr, '  ----  ', delta_e, ' > ',prec/10) 
            idx += 1

    new_energy_array[1] = new_energy_array[0]
    new_energy_array[3] = new_energy_array[2]



    return  new_spinor_array, new_energy_array


def J_K_Psi_BALANCED(spinorb_array, R_spinorb_array, mra, prec):
    J_Psi_array = []
    K_Psi_array = []
    K_Psi_array_R = []
    for i in range(4):
        J_Psi_array.append(orb2c.orbital2c().setZero())
        K_Psi_array.append(orb2c.orbital2c().setZero())
        K_Psi_array_R.append(orb2c.orbital2c().setZero())

    
    contr_1 = spinorb_array[0].density(prec) + R_spinorb_array[0].density(prec)
    contr_3 = spinorb_array[2].density(prec) + R_spinorb_array[2].density(prec)
    

    # J part
    #rho_total = contr_1 + contr_2 + contr_3 + contr_4
    P = vp.PoissonOperator(mra, prec/10)
    conv_1 = P(contr_1)
    conv_3 = P(contr_3)
    J_tree = 8 *np.pi * (conv_1 +  conv_3)

    for i in range(4):
        J_Psi_array[i] = J_tree * spinorb_array[i]
    # K part, diagonal
    K_Psi_array[0] = 4 * np.pi * conv_1 * spinorb_array[0]
    K_Psi_array[1] = 4 * np.pi * conv_1 * spinorb_array[1]
    K_Psi_array[2] = 4 * np.pi * conv_3 * spinorb_array[2]
    K_Psi_array[3] = 4 * np.pi * conv_3 * spinorb_array[3]

    K_Psi_array_R[0] = 4 * np.pi * conv_1 * R_spinorb_array[0]
    K_Psi_array_R[1] = 4 * np.pi * conv_1 * R_spinorb_array[1]
    K_Psi_array_R[2] = 4 * np.pi * conv_3 * R_spinorb_array[2]
    K_Psi_array_R[3] = 4 * np.pi * conv_3 * R_spinorb_array[3]


    # FOR EACH SPINOR IN THE ORBITAL GROUP
    for i in range(4):
        # APPLY K_j \Psi_i
        for j in range(4):
            if i != j:
                ol_density = 4 * np.pi * (spinorb_array[j].overlap_density(spinorb_array[i], prec) + R_spinorb_array[j].overlap_density(R_spinorb_array[i], prec))
                conv_ol = cf.apply_poisson(ol_density, mra,P, prec)

                K_Psi_array[i] = K_Psi_array[i] + spinorb_array[j].single_func_multiply(conv_ol, prec)
                K_Psi_array_R[i] = K_Psi_array_R[i] + R_spinorb_array[j].single_func_multiply(conv_ol, prec)

    


    return J_tree, J_Psi_array, K_Psi_array, K_Psi_array_R


def calc_Fock(spinor_array, V_Psi_array,R_Psi_array, energy_guess_array, deriv, mra, prec):
    F_matrix = np.zeros((4,4), dtype=complex)
    light_speed = spinor_array[0].light_speed
    
    
    for i in range(4):
        for j in range(4):
            overlap_L = orb2c.dot(spinor_array[i], spinor_array[j])
            overlap_R = orb2c.dot(R_Psi_array[i], R_Psi_array[j])
            F_matrix[i,j] = (overlap_L + overlap_R) * energy_guess_array[j]


    return F_matrix








# ==========================================================================================================


def scf_4el_cheat(spinor_array, energy_guess_array, potential, mra, prec, thr, derivative, output_file="output", niter=5):
    
    error_norm = 1
    delta_e = 1
    idx = 0

    light_speed = spinor_array[0].light_speed
    c2 = light_speed**2
    
    new_spinor_array = [orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c()]
    R_spinor_array = [orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c()]
    V_Psi_array_L = [orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c()]
    V_Psi_array_R = [orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c()]
    pi_Psi_array_L = [orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c(), orb2c.orbital2c()]
    Fock_term_propagator = orb2c.orbital2c()

    new_energy_array = np.zeros(4)
    
    # GET ESTIMATE OF J AND K TERMS FOR THE FIRST ITERATION
    F_ij, J_ij, K_ij, J_tree, J_Psi_array, K_Psi_array, K_Psi_array_R, R_spinor_array = F_matrix(spinor_array, R_spinor_array, potential, mra, prec, derivative, True)
    print("Initial F matrix:")
    print(F_ij)

    for i in range(4):
        new_energy_array[i] = F_ij[i,i].real / (spinor_array[i].squaredNorm() + R_spinor_array[i].squaredNorm())
        print("L norm:", spinor_array[i].norm() , "R norm:", R_spinor_array[i].norm())
        print(f"Initial energy estimate for spinor {i+1}: {new_energy_array[i]}")
        print()
    
            
    

    J_and_V = J_tree - potential
    print("    Precomputing V_Psi and pi_Psi for the 4 spinors:")
    for i in range(4):
        V_Psi_array_L[i] = orb2c.apply_potential(1.0, J_and_V, spinor_array[i], prec)
        V_Psi_array_L[i] = V_Psi_array_L[i] - K_Psi_array[i]

        V_Psi_array_R[i] = orb2c.apply_potential(1.0, J_and_V, R_spinor_array[i], prec)
        V_Psi_array_R[i] = V_Psi_array_R[i] - K_Psi_array_R[i]

        pi_Psi_array_L[i] = spinor_array[i].sigma_p(prec, derivative) 




    
    while (idx < niter and (delta_e > prec/10 or error_norm > thr)):
        # 

        print("==============================")
        print(f"> Iteration {idx + 1}")
        print("==============================")

        print()
        print("    Calculating the Fock term for the propagator and the Big_V_Psi for the 4 spinors:")
        for i in range(4):
            Big_V_Psi = -c2 * V_Psi_array_R[i] + energy_guess_array[i] * V_Psi_array_L[i] - light_speed * V_Psi_array_L[i].sigma_p(prec, derivative)         
            Fock_term_propagator.setZero()
            for j in range(4):
                if j == i:
                    continue
                Fock_term_propagator = Fock_term_propagator + F_ij[i,j] * ( R_spinor_array[j] +  light_speed * pi_Psi_array_L[j] + energy_guess_array[j] * spinor_array[j] - V_Psi_array_L[j]) 
            
            Big_V_Psi = (1/c2) * (Big_V_Psi + Fock_term_propagator)

            # PROPAGATE

            mu_old = orb2c.calc_dirac_mu(new_energy_array[i], light_speed)
            new_spinor_array[i] = orb2c.apply_helmholtz(Big_V_Psi, mu_old, prec)
            # TODO: Find the R component from the new L component, using the relation R = (1/c2) * (energy * L - V*L - c * sigma * p L)
            
        
        J_tree, J_Psi_array, K_Psi_array, K_Psi_array_R, R_spinor_array = J_K_Psi_BALANCED(new_spinor_array, R_spinor_array, mra, prec)

        J_and_V = J_tree - potential
        print("    Precomputing V_Psi and pi_Psi for the 4 spinors:")
        for i in range(4):
            V_Psi_array_L[i] = orb2c.apply_potential(1.0, J_and_V, spinor_array[i], prec)
            V_Psi_array_L[i] = V_Psi_array_L[i] - K_Psi_array[i]

            V_Psi_array_R[i] = orb2c.apply_potential(1.0, J_and_V, R_spinor_array[i], prec)
            V_Psi_array_R[i] = V_Psi_array_R[i] - K_Psi_array_R[i]

            pi_Psi_array_L[i] = spinor_array[i].sigma_p(prec, derivative) 




        R_spinor_array[i] = new_spinor_array[i].apply_R(V_Psi_array_L[i], energy_guess_array[i], derivative, prec)   


        print()
        print("    Orthonormalizing the new spinor array")        
        new_spinor_array, R_spinor_array = Lowdin_orthonormalize_2c_cheat(new_spinor_array, R_spinor_array, mra, prec)
        for i in range(4):
            print(f"SpinorL {i+1} norm after orthonormalization:")
            orb2c.print_norm_debug(new_spinor_array[i])
        print()
        for i in range(4):
            print(f"SpinorR {i+1} norm after orthonormalization:")
            orb2c.print_norm_debug(R_spinor_array[i])

        print()
        print("    Calculating new J and K terms for the updated spinors")

        F_ij, J_ij, K_ij, J_tree, J_Psi_array, K_Psi_array, K_Psi_array_R, R_spinor_array= F_matrix(new_spinor_array, R_spinor_array, potential, mra, prec, derivative, False)
        for i in range(4):
            new_energy_array[i] = F_ij[i,i].real / (new_spinor_array[i].squaredNorm() + R_spinor_array[i].squaredNorm())
            print()
            print(f"New energy estimate for spinor {i+1}: {new_energy_array[i]-c2}")
            print()
        print()
        print("==============================")

        spinor_array = new_spinor_array
        energy_guess_array = new_energy_array
        
        
        idx += 1
        

    return new_spinor_array, new_energy_array





def J_and_K(spinor_array, mra, prec, is_guess = False, R_spinor_array = None):
    J_ij = np.zeros((4,4), dtype=complex)
    K_ij = np.zeros((4,4), dtype=complex)

 
    if is_guess:
        R_spinor_array = spinor_array
    J_tree, J_Psi_array, K_Psi_array, K_Psi_array_R = J_K_Psi_BALANCED(spinor_array, R_spinor_array, mra, prec)

    for i in range(4):
        for j in range(4):
            J_ij[i,j] = orb2c.dot(spinor_array[i], J_Psi_array[j]) + orb2c.dot(R_spinor_array[i], J_tree * R_spinor_array[j])
            K_ij[i,j] = orb2c.dot(spinor_array[i], K_Psi_array[j]) + orb2c.dot(R_spinor_array[i], K_Psi_array_R[j])
            J_ij[j,i] = J_ij[i,j].conjugate()
            K_ij[j,i] = K_ij[i,j].conjugate()

    return J_ij, K_ij, J_tree, J_Psi_array, K_Psi_array, K_Psi_array_R

def F_matrix(spinor_array, R_spinor_array, potential, mra, prec, derivative, is_guess = False):
    F_matrix = np.zeros((4,4), dtype=complex)
    light_speed = spinor_array[0].light_speed
    c2 = light_speed**2

    if is_guess:
        R_spinor_array = [orb2c.orbital2c(),
                         orb2c.orbital2c(),
                         orb2c.orbital2c(),
                         orb2c.orbital2c()]
        for i in range(4):
            tmp = spinor_array[i].sigma_p(prec, derivative)
            R_spinor_array[i] = spinor_array[i] - (1/(2*light_speed)) * tmp 
            spinor_array[i] = spinor_array[i] + (1/(2*light_speed)) * tmp



    J_ij, K_ij, J_tree, J_Psi_array, K_Psi_array, K_Psi_array_R = J_and_K(spinor_array, mra, prec, is_guess, R_spinor_array)

    for i in range(4):
        for j in range(i,4):
            V_ij = orb2c.dot(spinor_array[i], orb2c.apply_potential(-1.0, potential, spinor_array[j], prec)) + orb2c.dot(R_spinor_array[i], orb2c.apply_potential(-1.0, potential, R_spinor_array[j], prec))
            cpi_Psi_L = light_speed * spinor_array[j].sigma_p(prec, derivative)
            cpi_Psi_R = light_speed * R_spinor_array[j].sigma_p(prec, derivative)
            cpi_ij = orb2c.dot(spinor_array[i], cpi_Psi_L) - orb2c.dot(R_spinor_array[i], cpi_Psi_R)
            T_ij = cpi_ij + c2 * orb2c.dot(spinor_array[i], R_spinor_array[j]) + c2 * orb2c.dot(R_spinor_array[i], spinor_array[j])
            F_matrix[i,j] = V_ij + J_ij[i,j] - K_ij[i,j] + T_ij
            F_matrix[j,i] = F_matrix[i,j].conjugate()
            
            if i == j:
                norm_Dirac = spinor_array[i].squaredNorm() + R_spinor_array[i].squaredNorm()
                print(F_matrix[i,j]/(norm_Dirac), "---->", (F_matrix[i,j]/(norm_Dirac) - c2.real))
                print(f"cpi_ij for i={i}, j={j}: {cpi_ij/norm_Dirac}")
                print(f"T_ij for i={i}, j={j}: {T_ij/norm_Dirac}", "---->", (T_ij)/norm_Dirac -c2.real)
                print(f"V_ij for i={i}, j={j}: {V_ij/norm_Dirac}")
                print(f"J_ij for i={i}, j={j}: {J_ij[i,j]/norm_Dirac}")
                print(f"K_ij for i={i}, j={j}: {K_ij[i,j]/norm_Dirac}")
                print()
    return F_matrix, J_ij, K_ij, J_tree, J_Psi_array, K_Psi_array, K_Psi_array_R, R_spinor_array




def Lowdin_orthonormalize_2c_cheat(L_array, R_array, mra, prec, debug = False):
    S_matrix_L = compute_S_matrix_4c(L_array, mra, prec)
    S_matrix_R = compute_S_matrix_4c(R_array, mra, prec)
    S_matrix = S_matrix_L + S_matrix_R
    if debug:
        print("S_matrix_L:")
        print(S_matrix_L)
        print("S_matrix_R:")
        print(S_matrix_R)
        print("Total S_matrix:")
        print(S_matrix)

    eigvals, eigvecs = LA.eig(S_matrix)
    S_inv_sqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ inv(eigvecs)

    orthonormal_spinor_array_L = []
    orthonormal_spinor_array_R = []
    for i in range(4):
        new_spinor_L = orb2c.orbital2c()
        new_spinor_R = orb2c.orbital2c()

        for j in range(4):
            new_spinor_L = new_spinor_L + (S_inv_sqrt[j,i]) * L_array[j]
            new_spinor_R = new_spinor_R + (S_inv_sqrt[j,i]) * R_array[j]
        L_norm_inv = 1.0/new_spinor_L.norm()
        orthonormal_spinor_array_L.append(new_spinor_L * L_norm_inv)
        orthonormal_spinor_array_R.append(new_spinor_R * L_norm_inv)

    return orthonormal_spinor_array_L, orthonormal_spinor_array_R
