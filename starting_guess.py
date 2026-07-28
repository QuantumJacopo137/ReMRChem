from vampyr import vampyr3d as vp
import numpy as np
from scipy.special import eval_genlaguerre
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import orbital_2c as orb2c
import one_electron as oneel

def make_starting_guess(mra, prec):
    gauss_tree_tot = vp.FunctionTree(mra)
    gauss_tree_tot.setZero()
    a_coeff = 3.0
    b_coeff = np.sqrt(a_coeff/np.pi)**3
    AO_list = []
    for atom in coordinates:
        gauss = vp.GaussFunc(b_coeff, a_coeff, [atom[2], atom[3], atom[4]])
        gauss_tree = vp.FunctionTree(mra)
        vp.advanced.build_grid(out=gauss_tree, inp=gauss)
        vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
        AO_list.append(gauss_tree)
    gauss_tree_sum = vp.sum(AO_list)

    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real = gauss_tree_sum)
    spinorb1.copy_components(La = La_comp)
    spinorb1.init_small_components(prec/10)
    spinorb1.normalize()
    spinorb1.cropLargeSmall(prec)
    return spinorb1

def make_NR_starting_guess(position, charge, mra, prec, comp = 4, n=1, l=0):
    nr_wf_tree = vp.FunctionTree(mra)
    nr_wf_tree.setZero()
    print("Generated the non-relativistic starting guess for n =", n, "and l =", l, "with charge =", charge)
    Peps = vp.ScalingProjector(mra, prec)
    guess = lambda x : wf_hydrogenionic_atom(n,l,[x[0]-position[0], x[1]-position[1], x[2]-position[2]],charge)
    nr_wf_tree = Peps(guess)
    
    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real = nr_wf_tree)

    Sa_comp = cf.complex_fcn()
    
    if (comp == 2):
        spinorb1 = orb2c.orbital2c()
        spinorb1.setZero()
        spinorb1.copy_component(La_comp, "alpha")
        spinorb1.normalize()
        
        print(f"Initial norm of the 2-component spinor: {spinorb1}")
        
    else:
        spinorb1 = orb.orbital4c()

        spinorb1.copy_components(La = La_comp)
        #spinorb1.copy_components(Lb = La_comp)

        spinorb1.copy_components(Sa = La_comp)
        #spinorb1.copy_components(Sb = Sa_comp)
        #spinorb1 = init_Right_components(spinorb1, charge, potential)
        light_speed = orb.orbital4c.light_speed 
        spinorb1 = spinorb1 + (0.5/light_speed) * spinorb1.alpha_p(prec*10)
        spinorb1.normalize()
        spinorb1.crop(prec/10)
    return spinorb1


def make_NR_starting_guess_1s_weyl(position, charge, mra, prec):
    n, l = 1, 0
    Peps = vp.ScalingProjector(mra, prec)
    light_speed = orb.orbital4c.light_speed
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    guess = lambda x: wf_hydrogenionic_atom(
        n, l, [x[0] - position[0], x[1] - position[1], x[2] - position[2]], charge
    )
    nr_wf_tree = Peps(guess)

    def get_components(x):
        rx, ry, rz = x[0] - position[0], x[1] - position[1], x[2] - position[2]
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        if r < 1e-14:
            return 0.0, 0.0, 0.0
        factor = charge / (2.0 * light_speed * r)
        return factor * rx, factor * ry, factor * rz

    x_comp_tree = Peps(lambda x: get_components(x)[0]) * nr_wf_tree
    y_comp_tree = Peps(lambda x: get_components(x)[1]) * nr_wf_tree
    z_comp_tree = Peps(lambda x: get_components(x)[2]) * nr_wf_tree

    # Psi_L = 1/sqrt(2) * (phi - chi)
    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real=inv_sqrt2 * nr_wf_tree, imag=-inv_sqrt2 * z_comp_tree)

    Lb_comp = cf.complex_fcn()
    Lb_comp.copy_fcns(real=inv_sqrt2 * y_comp_tree, imag=-inv_sqrt2 * x_comp_tree)

    # Psi_R = 1/sqrt(2) * (phi + chi)
    Sa_comp = cf.complex_fcn() 
    Sa_comp.copy_fcns(real=inv_sqrt2 * nr_wf_tree, imag=inv_sqrt2 * z_comp_tree)
    
    Sb_comp = cf.complex_fcn()
    Sb_comp.copy_fcns(real=-inv_sqrt2 * y_comp_tree, imag=inv_sqrt2 * x_comp_tree)

    spinorb1 = orb.orbital4c()
    spinorb1.copy_components(La=La_comp, Lb=Lb_comp, Sa=Sa_comp, Sb=Sb_comp)
    spinorb1.normalize()
    spinorb1.crop(prec)
    
    return spinorb1


def make_NR_starting_guess_with_pot(position, charge, mra, prec, potential, comp = 4, n=1, l=0):
    nr_wf_tree = vp.FunctionTree(mra)
    nr_wf_tree.setZero()
    print("Generated the non-relativistic starting guess for n =", n, "and l =", l, "with charge =", charge)
    Peps = vp.ScalingProjector(mra, prec)
    guess = lambda x : wf_hydrogenionic_atom(n,l,[x[0]-position[0], x[1]-position[1], x[2]-position[2]],charge)
    nr_wf_tree = Peps(guess)
    c2 = orb.orbital4c.light_speed**2
    pot_inverse_term = lambda x : charge/((2*c2-3450 -potential(x))*())
    pot_inverse_term_tree = vp.FunctionTree(mra)
    pot_inverse_term_tree = Peps(pot_inverse_term)


    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real = nr_wf_tree)

    Sa_comp = cf.complex_fcn()
    
    if (comp == 2):
        print("2-component spinor not implemented yet")
        exit(-1)
        
    else:
        spinorb1 = orb.orbital4c()
        spinorb1.copy_components(La = La_comp)
        spinorb1.copy_components(Sa = La_comp)
        
    
        light_speed = orb.orbital4c.light_speed 
        one_over_2mc2 = 1.0/(2*light_speed * light_speed)

        ap_initial = spinorb1.alpha_p(prec/10)

        spinorb1 = spinorb1 + light_speed * one_over_2mc2 * pot_inverse_term_tree * ap_initial


        spinorb1.normalize()
        spinorb1.crop(prec)
    return spinorb1


def init_Right_components(spinorb, charge, potential):
    light_speed = orb.orbital4c.light_speed
    energy_guess = oneel.analytic_1s(light_speed, 1, -1, charge)
    V_psi_alpha = potential * spinorb.comp_array[0]
    V_psi_beta = potential * spinorb.comp_array[1]
    psiL_alpha = spinorb.comp_array[0]
    psiL_beta = spinorb.comp_array[1]
    psiL_alpha_grad = psiL_alpha.gradient('ABGV')
    psiL_beta_grad = psiL_beta.gradient('ABGV')

    psiR_alpha = psiL_alpha_grad[0] + (psiL_beta_grad[1]- 1j*psiL_beta_grad[2])
    psiR_beta = (psiL_alpha_grad[1] + 1j*psiL_alpha_grad[2]) + psiL_beta_grad[0]

    psiR_alpha = -(1/light_speed**2)*(-1j*light_speed*psiR_alpha + V_psi_alpha - energy_guess * psiL_alpha)
    psiR_beta = -(1/light_speed**2)*(-1j*light_speed*psiR_beta + V_psi_beta - energy_guess * psiL_beta)

    
    spinorb.copy_components(Sa = psiR_alpha)
    spinorb.copy_components(Sb = psiR_beta)
    return spinorb


#returns the value of the radial WF in the point r
# 1. the nucleus is assumed infintely heavy (mass of electron and Bohr radius used)
# 2. the nucleus is placed in the origin
# 3. atomic units are assumed a0 = 1  hbar = 1  me = 1  4pie0 = 1
def radial_wf_hydrogenionic_atom(n,l,r,Z):
    rho = 2 * Z * r / n # I had to add the n at the denominator as it was missing.
    slater = np.exp(-rho/2)
    polynomial = eval_genlaguerre(n-l-1, 2*l+1, rho)
    # Note: NumPy does not have a factorial function, this is a workaround
    f1 = np.prod(np.arange(1,n-l)) # factorial(n-l-1)
    f2 = np.prod(np.arange(1,n+l+1))    # factorial(n+l)
    norm = np.sqrt((2*Z/n)**3 * f1 / (2 * n * f2))
    value = norm * rho**l * polynomial * slater
    return value

def wf_hydrogenionic_atom(n,l,position,Z):
    if(l != 0):
        print("only s orbitals for now")
        exit(-1)
    distance = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
    value = radial_wf_hydrogenionic_atom(n, l, distance, Z)
    return value


