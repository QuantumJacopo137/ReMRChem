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


def scf_4el(spinorb_array, potential, mra, prec, max_iter=10, auto_save = False, Dampen_alpha = 0.5):
    # Initialize the starting guess
    light_speed = spinorb_array[0].light_speed
    c2 = light_speed**2
    spinorb_array[1] = spinorb_array[0].ktrs()
    spinorb_array[3] = spinorb_array[2].ktrs()

    # allocate some memory for these
    spinorb_array_new = [orb.orbital4c() for i in range(4)]

    # Make sure the initial guess is orthonormalized
    spinorb_array = Dirac_Lowdin_orthonormalization(spinorb_array, prec, verbose=True)
    
    # Compute 2-el terms
    V_Psi_array = J_K_Psi(spinorb_array, spinorb_array, mra, prec, True)
   
   # add the nuclear potential contribution to V_Psi_array
    print()
    for i in range(4):
        V_Psi_i = orb.apply_potential(-1.0, potential, spinorb_array[i], prec)
        V_Psi_array[i] = V_Psi_array[i] + V_Psi_i
    print()
    
    # Compute the F matrix for the initial guess
    F_ij = F_matrix(spinorb_array, V_Psi_array, prec, True)
    print()
    print("----------------- Initial F matrix -----------------")
    print_matrix(F_ij)
    print("----------------------------------------------------")

    print()
    
    for iteration in range(max_iter):
        print("\n===== SCF Iteration: ", iteration, " =====\n")

        #F_ij[0,0]  = c2 - 4.73
        #F_ij[2,2]  = c2 - 0.309
        #F_ij[1,1] = F_ij[0,0]
        #F_ij[3,3] = F_ij[2,2]

        # Propagate the spinors using the exact propagator, which is given by the formula:
        print("\n-> Propagating the spinors using the exact propagator... \n")
        spinorb_array_new = exact_propagator(spinorb_array, V_Psi_array, F_ij, prec, False)

        print("\n-> Balancing the Dirac spinors... \n")


        spinorb_array_new = balance_Dirac_spinor(spinorb_array_new, V_Psi_array, F_ij, prec, True)
        # Most of the time this is needed 
        spinorb_array_new = Dampen_iteration(spinorb_array, spinorb_array_new, Dampen_alpha, prec)


        print("\n-> Orthonormalizing the Dirac spinors... \n")
        spinorb_array_new = Dirac_Lowdin_orthonormalization(spinorb_array_new, prec, verbose=False)
        #print("\n \t > After orthonormalization: \n")

        print("\n-> Calculating the new V_Psi_array... \n")
        #for i in range(4):
        #    V_Psi_array[i].setZero()
        V_Psi_array = J_K_Psi(spinorb_array_new, spinorb_array_new, mra, prec, verbose=True)

        # add the nuclear potential contribution to V_Psi_array
        for i in (0,2):
            V_Psi_array[i] = V_Psi_array[i] + orb.apply_potential(-1.0, potential, spinorb_array_new[i], prec)
            V_Psi_array[i+1] = V_Psi_array[i].ktrs()
        
        # compute the new F matrix
        print("\n-> Computing the new F matrix... \n")
        F_ij = F_matrix(spinorb_array_new, V_Psi_array, prec, verbose = True)
        # print it
        print_matrix(F_ij)
        print()

        norm_diff = 0.0

        for i in range(4):
            norm_diff_i = (spinorb_array_new[i] - spinorb_array[i]).squaredNorm()
            print(f"Norm of the difference between the new and old spinor {i}: {np.sqrt(norm_diff_i)}")
            norm_diff += norm_diff_i
            spinorb_array[i].setZero()
            spinorb_array[i] = spinorb_array_new[i]
        norm_diff = np.sqrt(norm_diff)
        print()
        print("-> Norm difference for spinor: ", norm_diff)
        print()
        print("------------------------------------------------------")
        print(" => Average norm difference per spinor: ", norm_diff/4, " |")
        print("------------------------------------------------------")
        Dampen_alpha = norm_diff
        if norm_diff < prec/10:
            print(f"SCF converged after {iteration} iterations.")
            break

    return spinorb_array_new, F_ij


def print_array_SqNorms(spinor_array):
    for i in range(4):
        print(f"Norm of spinor {i}: {spinor_array[i].squaredNorm():.5e}")
    return


def Dampen_iteration(old_array, new_array, dampen_coeff = 1.0, prec = 1.0e-5):
    dampen_coeff = max(0.0, min(1.0, dampen_coeff)) # ensure that the dampen coefficient is between 0 and 1
    dampen_coeff = np.sqrt(dampen_coeff) # take the square root of the dampen coefficient to make the damping less aggressive
    for i in range(4):
        new_array[i] = dampen_coeff * new_array[i] + (1-dampen_coeff) * old_array[i]
        new_array[i].crop(prec/10) # crop the new spinor to remove small values that can cause numerical instability in the next iterations
    return new_array


def print_matrix(matrix, zero_thr = 1.0e-5):
    
    dim_i = matrix.shape[0]
    dim_j = matrix.shape[1]

    real_M = np.zeros((dim_i, dim_j))
    imag_M = np.zeros((dim_i, dim_j))


    for i in range(dim_i):
         for j in range(dim_j):
            real_M[i,j] = matrix[i,j].real
            imag_M[i,j] = matrix[i,j].imag

    for i in range(dim_i):
        for j in range(dim_j):
            re = real_M[i,j]
            im = imag_M[i,j]
            if re >= 0:
                print(f"+{re:05.3e}{' + ' if im >= 0 else ' - '}{abs(im):05.3e}j\t", end = "\t")
            else:
                print(f"{re:05.3e}{' + ' if im >= 0 else ' - '}{abs(im):05.3e}j\t", end = "\t")
        print()
    return
      



def Dirac_Lowdin_orthonormalization(spinor_array, prec, verbose = False):
    # compute the overlap matrix S
    S_matrix = np.zeros((4,4), dtype=complex)
    is_diag = True
    for i in range(4):
        for j in range(4):
            S_matrix[i,j] = spinor_array[i].dot(spinor_array[j])
            if i != j and not abs(S_matrix[i,j]) > prec:
                is_diag = False




    if verbose:
        print("Intial overlap matrix S:")
        print_matrix(S_matrix)

    if is_diag:
        print("Overlap matrix is already diagonal, skipping Lowdin orthonormalization.")
        return spinor_array

    # compute the inverse square root of the S matrix
    eigvals, eigvecs = LA.eig(S_matrix)
    D_inv_sqrt = np.diag(1.0/np.sqrt(eigvals))
    S_inv_sqrt = eigvecs @ D_inv_sqrt @ inv(eigvecs)

    # apply the transformation to the spinor array
    orthonormal_array = [orb.orbital4c() for i in range(4)]
    for i in range(4):
        orthonormal_array[i] = S_inv_sqrt[i,0] * spinor_array[0] + S_inv_sqrt[i,1] * spinor_array[1] + S_inv_sqrt[i,2] * spinor_array[2] + S_inv_sqrt[i,3] * spinor_array[3]

    if verbose:
        for i in range(4):
            for j in range(4):
                S_matrix[i,j] = orthonormal_array[i].dot(orthonormal_array[j])
            
        print("\nFinal overlap matrix S after Lowdin orthonormalization:")
        print_matrix(S_matrix)

        print()
   

    return orthonormal_array

     
def exact_propagator(spinor_array, V_Psi_array, F_matrix, prec, exact = True):
    # var declaration
    light_speed = spinor_array[0].light_speed
    c2 = light_speed**2
    new_dirac_spinor_array = [orb.orbital4c() for i in range(4)]

    L_spinor_array = [orb2c.orbital2c() for i in range(4)] # an array of 2-Weyl spinors that will hold the left components of the Dirac spinors
    R_spinor_array = [orb2c.orbital2c() for i in range(4)] # an array of 2-Weyl spinors that will hold the right components of the Dirac spinors


    # section the spinor, and separate the chiral contributions
    for i in range(4):
        L_spinor_array[i] = Dirac_to_Weyl(spinor_array[i])
        R_spinor_array[i] = Dirac_to_Weyl(spinor_array[i], weyl_L=False)
        

    # compute the exact propagator
    V_L = orb2c.orbital2c()                 # V * Psi_L
    V_R = orb2c.orbital2c()                 # V * Psi_R
    Big_V_Psi_L = orb2c.orbital2c()         # the big term in the numerator of the exact propagator, which is given by -c^2 * V * Psi_R + F_ii * V * Psi_L - c * V * sigma_p * Psi_L
    tmp_new_spinor_L = orb2c.orbital2c()    

    for i in (0,2):
        # Reset the temporary new spinor to zero before each iteration
        tmp_new_spinor_L.setZero()
        

        # Compute the V * Psi_L and V * Psi_R terms for the current spinor
        V_L = Dirac_to_Weyl(V_Psi_array[i])
        V_R = Dirac_to_Weyl(V_Psi_array[i], weyl_L=False)

        # Compute the big term in the numerator of the exact propagator
        Big_V_Psi_L = -c2 * V_R - F_matrix[i,i].real * V_L - light_speed * V_L.sigma_p(prec/10) # last time i have changed the sign of  F_matrix[i,i].real * V_L
        
        # Sum term of the exact propagator, which is given by the sum over j != i of F_ij * (c^2 * Psi_R_j + c * sigma_p * Psi_L_j - V * Psi_L_j + F_jj * Psi_L_j)
        if exact:
            for j in range(4):
                if j != i:
                    tmp_new_spinor_L = tmp_new_spinor_L + F_matrix[i,j] * (c2 * R_spinor_array[j] + light_speed * L_spinor_array[j].sigma_p(prec) - Dirac_to_Weyl(V_Psi_array[j]) + F_matrix[i,i] * L_spinor_array[j])
        else:
            for j in range(4):
                if j != i or abs(F_matrix[i,j]) > prec/10:
                    tmp_new_spinor_L.setZero()
                    tmp_new_spinor_L +=  -F_matrix[i,j] * Dirac_to_Weyl(V_Psi_array[j], weyl_L=False)

        # Sum the big term and the sum term, and divide by c^2 to get the new left spinor
        tmp_new_spinor_L = (1.0/c2)*(tmp_new_spinor_L + Big_V_Psi_L)

        # Store the norm of the original spinor before applying the convolution, so that we can compare it with the norm of the new spinor after applying the convolution
        norm_pre_conv = L_spinor_array[i].norm()
        
        # Apply the helmholtz convolution to the new left spinor, and then combine it with the right spinor to get the new Dirac spinor
        mu = orb2c.calc_dirac_mu(F_matrix[i,i].real, light_speed)
        tmp_new_spinor_L = orb2c.apply_helmholtz(tmp_new_spinor_L, mu, prec)
        
        # Now check the new norm adter it
        norm_post_conv = tmp_new_spinor_L.norm()
        # Find the ratio
        ratio_norm = norm_pre_conv / norm_post_conv
        
        print(f"Spinor {i}: norm before convolution = {norm_pre_conv:.3e}, norm after convolution = {norm_post_conv:.3e}, ratio = {ratio_norm:.3e}")
        
        # Last step is to combine them so that the ratio of the norms is preserved, and store the new Dirac spinor in the new_dirac_spinor_array
        new_dirac_spinor_array[i] = Weyl_to_Dirac(ratio_norm * tmp_new_spinor_L, R_spinor_array[i])

    new_dirac_spinor_array[1] = new_dirac_spinor_array[0].ktrs()
    new_dirac_spinor_array[3] = new_dirac_spinor_array[2].ktrs()

    return new_dirac_spinor_array


def Dirac_to_Weyl(dirac_spinor, weyl_L = True):
    # Take the 4-component Dirac spinor and return the 2-component Weyl spinor. If weyl_L is True, return the left-handed Weyl spinor, otherwise return the right-handed Weyl spinor.
    weyl_spinor = orb2c.orbital2c()
    weyl_spinor.setZero()
    if weyl_L:
        weyl_spinor["alpha"] = dirac_spinor["La"]
        weyl_spinor["beta"] = dirac_spinor["Lb"]
    else:
        weyl_spinor["alpha"] = dirac_spinor["Sa"]
        weyl_spinor["beta"] = dirac_spinor["Sb"]
    return weyl_spinor


def Weyl_to_Dirac(weyl_spinor_L, weyl_spinor_R):
    # Build a 4-component Dirac spinor from the left-handed and right-handed Weyl spinors. The left-handed Weyl spinor will be the first 2 components of the Dirac spinor, and the right-handed Weyl spinor will be the last 2 components of the Dirac spinor.
    dirac_spinor = orb.orbital4c()
    dirac_spinor.setZero()

    dirac_spinor["La"] = weyl_spinor_L.__getitem__("alpha")
    dirac_spinor["Lb"] = weyl_spinor_L.__getitem__("beta")
    dirac_spinor["Sa"] = weyl_spinor_R.__getitem__("alpha")
    dirac_spinor["Sb"] = weyl_spinor_R.__getitem__("beta")

    return dirac_spinor


def balance_Dirac_spinor(spinor_array, V_Psi_array, F_ij, prec, one_el_balance = False):
    # this is gonna take the first 2 components of a general spinor and compute the second 2 components using the balance condition. This is gonna be used in the exact propagator to compute the new spinor array after one iteration of the SCF procedure. The balance condition is given by:
    L_spinor_array = [orb2c.orbital2c() for i in range(4)] 
    R_spinor_array = [orb2c.orbital2c() for i in range(4)]

    c2 = (spinor_array[0].light_speed)**2
    for i in range(4):
        L_spinor_array[i] = Dirac_to_Weyl(spinor_array[i])
        #print(f"Norm of L spinor {i} before balancing: {L_spinor_array[i].norm():.3e}")
    
    for i in (0,2):
        R_spinor_array[i] = L_spinor_array[i].apply_R(Dirac_to_Weyl(V_Psi_array[i]), F_ij[i,i], 'BS', prec)
        if (not one_el_balance):
            for j in range(4):
                if abs(F_ij[i,j]) < prec/10 or abs(F_ij[i,j]) > 1.0:
                    continue
                if j != i:
                    print("L spinor ", j,  " has a norm ", L_spinor_array[j].norm(), " and COEFF = ", F_ij[i,j]/c2)
                    tmp = (F_ij[i,j]/c2) * L_spinor_array[j]
                    print("norm balance contribution for spinor ", i, " from spinor ", j, ": ", tmp.norm())
                    R_spinor_array[i] += tmp


        R_spinor_array[i+1] = R_spinor_array[i].ktrs()

        
    for i in range(4):
        spinor_array[i] = Weyl_to_Dirac(L_spinor_array[i], R_spinor_array[i])
        spinor_array[i].normalize()

    return spinor_array
        

def F_matrix(spinor_array, V_Psi_array, prec, verbose = False, shift = 0.0):
    F_matrix = np.zeros((4,4), dtype=complex)
    light_speed = spinor_array[0].light_speed
    c2 = light_speed**2
    
    for i in (0,2):
        print()
        if verbose:
                print(f"Calculating F[{i},{i}]...")
        F_psi_j=  orb.apply_dirac_hamiltonian(spinor_array[i], prec, -shift)
        
        T = spinor_array[i].dot(F_psi_j)
        print(f"$ T_REL = {T.real-c2},       <T_REL - T_NR> = {T.real - spinor_array[i].classicT() - c2} ")
        L = Dirac_to_Weyl(spinor_array[i])
        R = Dirac_to_Weyl(spinor_array[i], weyl_L=False)
        print(f" -> < L | R > 2 *c2 - c2    = { (2 * (orb2c.dot(L,R)).real - 1) * c2}")
        print(f" -> < L | c\pi | L >        = {light_speed * (orb2c.dot(L,L.sigma_p(prec))).real}")
        print(f" -> < R | c\pi | R >        = {light_speed * (orb2c.dot(R,R.sigma_p(prec))).real}")
        print(f"$ ")
        print()
        print(f"$ < V_tot > = {spinor_array[i].dot(V_Psi_array[i]).real}")
        
        F_psi_j += V_Psi_array[i]
        F_matrix[i,i] = spinor_array[i].dot(F_psi_j)
        F_matrix[i+1,i+1] = F_matrix[i,i]
        print()


    if verbose:
        print(f"Calculating F[0,2]")
    F_psi_j=  orb.apply_dirac_hamiltonian(spinor_array[2], prec/10, -shift) 
    
    T = spinor_array[0].dot(F_psi_j)
    V = spinor_array[0].dot(V_Psi_array[2])
    
    F_matrix[0,2] = (T  + V ) 
    F_matrix[2,0] = np.conj(F_matrix[0,2])
    F_matrix[1,3] = np.conj(F_matrix[0,2]) 
    F_matrix[3,1] = F_matrix[0,2]
    if verbose:
        print(f"Calculating F[0,3]")
    
    F_psi_j= V_Psi_array[3] + orb.apply_dirac_hamiltonian(spinor_array[3], prec, -shift)
    normf = spinor_array[0].dot(spinor_array[3])
    F_matrix[0,3] = (spinor_array[0].dot(F_psi_j)) 
    F_matrix[3,0] = np.conj(F_matrix[0,3])
    F_matrix[1,2] = -np.conj(F_matrix[0,3])
    F_matrix[2,1] = -F_matrix[0,3]

    print("-----------------------------------------------")
    for i in range(4):
        print(f"Orbtial energy for spinor {i}: {F_matrix[i,i].real - c2}")
    print("-----------------------------------------------")

    return F_matrix


def J_K_Psi(Psi_array, spinor_array, mra, prec, verbose = False):
    # Returns J - K built with spinor_array and acting on Psi_array.
    # Psi_array is the one it acts upon
    # spinor_array is the one that generates the density for the J and K operators
    if (len(Psi_array) != len(spinor_array)):
        raise ValueError("Psi_array and spinor_array must have the same length.")
    
    if (len(Psi_array) != 4):
        raise ValueError("Psi_array and spinor_array must have length 4.")
    

    if verbose:
        print("Calculating K_Psi contributions...") 
    K_PSI_ARRAY = K_Psi(Psi_array, spinor_array, mra, prec)

    if verbose : 
        print("Calculating J_Psi contributions...")
    J_PSI_ARRAY = J_Psi(Psi_array, spinor_array, mra, prec)

    J_m_K_PSI_ARRAY = [orb.orbital4c() for i in range(4)]
    for i in (0,2):
        if verbose:
            print(f"Calculating < J > and < K > contributions for spinor {i}...")
            print(f"< J > contribution for spinor {i}: {spinor_array[i].dot(J_PSI_ARRAY[i]).real}")
            print(f"< K > contribution for spinor {i}: {spinor_array[i].dot(K_PSI_ARRAY[i]).real}\n")
        J_m_K_PSI_ARRAY[i] = J_PSI_ARRAY[i] - K_PSI_ARRAY[i]
        J_m_K_PSI_ARRAY[i].crop(prec/10)
        J_m_K_PSI_ARRAY[i+1] = J_m_K_PSI_ARRAY[i].ktrs()



    return J_m_K_PSI_ARRAY


def J_Psi(Psi_array, spinor_array, mra, prec):
    # Psi_array is the one it acts upon
    # spinor_array is the one that generates the density for the J operator
    if (len(Psi_array) != len(spinor_array)):
        raise ValueError("Psi_array and spinor_array must have the same length.")
    
    if (len(Psi_array) != 4):
        raise ValueError("Psi_array and spinor_array must have length 4.")
    
    # Define the Poisson operator
    P = vp.PoissonOperator(mra, prec/10)
    
    # Calculate the density for the 2 inependent spinors
    contr_1 = spinor_array[0].overlap_density(spinor_array[0], prec/10)
    contr_3 = spinor_array[2].overlap_density(spinor_array[2], prec/10)

    # Multiply by 2 to account for the contributions of the other 2 spinors, which are related by ktrs
    tot_density = 2 * (contr_1.real + contr_3.real)

    # Apply the Poisson operator to the total density to get the J potential, and then apply it to the Psi_array to get the J_PSI_ARRAY. Remember that the Poisson operator in Vampyr is automultiplied by 4\pi, so we need to multiply the density by 4\pi before applying the Poisson operator.
    J_tree = P(tot_density) 

    J_PSI_ARRAY = [orb.orbital4c() for i in range(4)]

    for i in (0,2):
        J_PSI_ARRAY[i] = orb.apply_potential(4*np.pi , J_tree, Psi_array[i], prec/10)
        J_PSI_ARRAY[i+1] = J_PSI_ARRAY[i].ktrs()

    return J_PSI_ARRAY


def K_Psi(Psi_array, spinor_array, mra, prec, ignore_thr = 0.0):
    # Psi_array is the one it acts upon
    # spinor_array is the one that generates the density for the K operator

    if (len(Psi_array) != len(spinor_array)):
        raise ValueError("Psi_array and spinor_array must have the same length.")
    
    if (len(Psi_array) != 4):
        raise ValueError("Psi_array and spinor_array must have length 4.")
    

    
    P = vp.PoissonOperator(mra, prec/10)        

    K_PSI_ARRAY = [orb.orbital4c() for i in range(4)]
    
    for i in (0,2):
        # Apply poisson - remember that it is automultiplied by 4\pi

        overlap_0i = spinor_array[0].overlap_density(Psi_array[i], prec/10)
        if (np.sqrt(overlap_0i.squaredNorm()) > ignore_thr):
            contr_0 = cf.apply_poisson(overlap_0i, mra, P, prec/10) 
            K_PSI_ARRAY[i] += orb.apply_complex_potential(1.0, contr_0, spinor_array[0], prec) 
        overlap_2i = spinor_array[2].overlap_density(Psi_array[i], prec/10)
        if (np.sqrt(overlap_2i.squaredNorm()) > ignore_thr):
            contr_2 = cf.apply_poisson(overlap_2i, mra, P, prec/10)
            K_PSI_ARRAY[i] += orb.apply_complex_potential(1.0, contr_2, spinor_array[2], prec) 

        K_PSI_ARRAY[i].crop(prec/10)
        # Define the next as the ktrs of the current one
        K_PSI_ARRAY[i+1] = K_PSI_ARRAY[i].ktrs()

    return K_PSI_ARRAY