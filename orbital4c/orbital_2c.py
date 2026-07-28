from vampyr import vampyr3d as vp
import numpy as np
import copy as cp
from scipy.special import gamma
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb

class orbital2c:
    """Two components orbital."""
    mra = None
    light_speed = -1.0
    comp_dict = {'alpha': 0, 'beta': 1}
    def __init__(self):
        self.comp_array = np.array([cf.complex_fcn(),
                                    cf.complex_fcn()])
        
    def __getitem__(self, key):
        return self.comp_array[self.comp_dict[key]]
    
    def __setitem__(self, key, val):
        self.comp_array[self.comp_dict[key]] = val
        
    def __len__(self):
        return 2

    def __str__(self):
        return ('> ALPHA\n{} BETA\n{}'.format(self["alpha"],
                                  self["beta"]))
    
    def __add__(self, other):
        output = orbital2c()
        output.comp_array = self.comp_array + other.comp_array
        return output

    def __sub__(self, other):
        output = orbital2c()
        output.comp_array = self.comp_array - other.comp_array
        return output

    def __call__(self, position):
        return [x(position) for x in self.comp_array]

    def save(self, name):
        self.comp_array[0].save(f"{name}_top")
        self.comp_array[1].save(f"{name}_bottom")

    def read(self, name):
        self.comp_array[0].read(f"{name}_top")
        self.comp_array[1].read(f"{name}_bottom")

    def __rmul__(self, factor):
        output = orbital2c()
        output.comp_array =  factor * self.comp_array
        return output

    def __mul__(self, factor):
        output = orbital2c()
        output.comp_array =  factor * self.comp_array
        return output   

    def norm(self):
        out = 0
        for comp in self.comp_dict.keys():
            comp_norm = self[comp].squaredNorm()
            out += comp_norm
        out = np.sqrt(out)
        return out

    def squaredNorm(self):
        out = 0
        for comp in self.comp_dict.keys():
            comp_norm = self[comp].squaredNorm()
            out += comp_norm
        return out

    def squaredAlphaNorm(self):
        alpha_ns = self.squaredNormComp('alpha')
        return alpha_ns 

    def squaredBetaNorm(self):
        beta_ns = self.squaredNormComp('beta')
        return beta_ns

    def squaredNormComp(self, comp):
        return self[comp].squaredNorm()

    def crop(self, prec):
        for func in self.comp_array:
            func.crop(prec)

    
    def setZero(self):
        for func in self.comp_array:
            func.setZero()

    def rescale(self, factor):
        for comp in self.comp_array:
            comp.real *= factor
            comp.imag *= factor



    def copy_component(self, func, component='alpha'):
        self[component].copy_fcns(func.real, func.imag)
        
    def normalize(self):
        norm_sq = 0
        for comp in self.comp_array:
            norm_sq += comp.squaredNorm()
        norm = np.sqrt(norm_sq)
        self.rescale(1.0/norm)

    def copy_components(self, alpha=None, beta=None):
        nr_of_functions = 0
        if(alpha != None):
            nr_of_functions += 1
            self.copy_component(alpha, 'alpha')
        if(beta != None):
            nr_of_functions += 1
            self.copy_component(beta, 'beta')
        if(nr_of_functions == 0):
            print("WARNING: No component copied!")
    
    def single_func_multiply(self, function, prec):
        out_orb = orbital2c()
        for comp in self.comp_dict.keys():
            out_orb[comp] = cf.multiply(prec, self[comp], function)
        return out_orb


    # THIS IS THE SIMPLIFIED VERSION OF THE RKB CONDITION, THE OFF DIAGONAL TERMS OF THE F MATRIX ARE NEGLECTED
    def apply_R(self, V_Psi, energy, derivative, prec, L_to_R = True):
    # initalize the small components based on the kinetic balance

        sigma_p_weyl = orbital2c()  

        sigma_p_weyl = self.sigma_p(prec/10, derivative)
        c2 = (orbital2c.light_speed)**2
        prefactor = 1.0 / c2

        RKB_spinor = orbital2c()
        RKB_spinor.setZero()
        epsilon = energy - c2
        RKB_spinor = self + (epsilon * prefactor) * self - prefactor * V_Psi 
      
        if L_to_R:
            RKB_spinor = RKB_spinor - (prefactor * orbital2c.light_speed) * sigma_p_weyl
        else:
            RKB_spinor = RKB_spinor + (prefactor * orbital2c.light_speed) * sigma_p_weyl

        RKB_spinor.crop(prec)
        return RKB_spinor
        
    def apply_R_full(spinor_array, el_id,  F_ij, V_Psi, energy, derivative, prec, L_to_R = True):
        free_R_spinor = spinor_array[el_id].apply_R(V_Psi, energy, derivative, prec, L_to_R)
        light_speed = spinor_array[el_id].light_speed
        prefactor = -1.0/(light_speed**2)
        for i in range(4):
            if i != el_id:
                coupling_term = (F_ij[el_id, i] * spinor_array[i])
                free_R_spinor = free_R_spinor -  prefactor  * coupling_term

        return free_R_spinor

    def derivative(self, dir = 0, der = 'BS'):
        orb_der = orbital2c()
        for key in self.comp_dict:
            orb_der[key] = self[key].derivative(dir, der) 
        return orb_der
    
    def gradient(self, der = 'BS'):
        orb_grad = {}
        for key in self.comp_dict.keys():
            orb_grad[key] = self[key].gradient(der)
        grad = []
        for i in range(3):
            comp = orbital2c()
            comp.copy_components(alpha = orb_grad['alpha'][i], 
                          beta = orb_grad['beta'][i])
            grad.append(comp)
        return grad
    
    def complex_conj(self):
        orb_out = orbital2c()
        for key in self.comp_dict.keys():
            orb_out[key] = self[key].complex_conj() 
        return orb_out

    def density(self, prec):
        density = vp.FunctionTree(self.mra)
        add_vector = []
        for comp in self.comp_array:
            temp = comp.density(prec).crop(prec)
            if(temp.squaredNorm() > 0):
                add_vector.append((1.0,temp))
        vp.advanced.add(prec, density, add_vector)
        return density    

#    def exchange(self, other, prec):
#        exchange = vp.FunctionTree(self.mra)
#        add_vector = []
#        for comp in self.comp_dict.keys():
#            func_i = self[comp]
#            func_j = other[comp]
#            temp = func_i.exchange(func_j, prec)
#            if(temp.squaredNorm() > 0):
#                add_vector.append((1.0,temp))    
#        vp.advanced.add(prec, exchange, add_vector)
#        return exchange
#
#    def alpha_exchange(self, other, prec):
#        alpha_exchange = vp.FunctionTree(self.mra)
#        add_vector = []
#        for comp in self.comp_dict.keys():
#            func_i = self[comp]
#            func_j = other[comp]
#            temp = func_i.alpha_exchange(func_j, prec)
#            if(temp.squaredNorm() > 0):
#                add_vector.append((1.0,temp))    
#        vp.advanced.add(prec, alpha_exchange, add_vector)
#        return alpha_exchange    

    def overlap_density(self, other, prec):
        density = cf.complex_fcn()
        add_vector_real = []
        add_vector_imag = []
        for comp in self.comp_dict.keys():
            func_i = self[comp]
            func_j = other[comp]
            temp = cf.multiply(prec, func_i.complex_conj(), func_j)
            if(temp.real.squaredNorm() > 0):
                add_vector_real.append((1.0,temp.real))
            if(temp.imag.squaredNorm() > 0):
                add_vector_imag.append((1.0,temp.imag))
        vp.advanced.add(prec, density.real, add_vector_real)
        vp.advanced.add(prec, density.imag, add_vector_imag)
        return density
    


    def sigma(self, direction, prec):
        # I degined sigma as -sigma to have nice sign conventions

        out_orb = orbital2c()

        sigma_order = np.array([[1, 0],
                                [1, 0],
                                [0, 1]])
        


        sigma_coeff = np.array([[ -1,  -1],
                                [+1j, -1j],
                                [ -1, +1]])
        
        
        
        

        for idx in range(2):
            coeff = sigma_coeff[direction][idx]
            comp = sigma_order[direction][idx]
            out_orb.comp_array[idx] = coeff * self.comp_array[comp]
            out_orb.comp_array[idx].crop(prec)
        return out_orb

    def sigma_p(self, prec, der = "BS"):
        out_orb = orbital2c()
        orb_grad = self.gradient(der)
        apx = orb_grad[0].sigma(0, prec)
        apy = orb_grad[1].sigma(1, prec)
        apz = orb_grad[2].sigma(2, prec)
        result =  (apx + apy + apz)
        out_orb = -1j * result
        return out_orb

    def classicT(self, der = 'BS'):
        orb_grad = self.gradient(der)
        val = 0
        for i in range(3):
            val += 0.5 * orb_grad[i].squaredNorm()
        return val

    def sigma_vector(self, prec):
        return [self.sigma(0, prec), self.sigma(1, prec), self.sigma(2, prec)]
    
    def ktrs(self, prec = 0.001):   #Kramers´ Time Reversal Symmetry
        out_orb = orbital2c()
        tmp = self.complex_conj()
        ktrs_order = np.array([1, 0,])
        ktrs_coeff = np.array([-1,  1])
        for idx in range(2):
            coeff = ktrs_coeff[idx]
            comp = ktrs_order[idx]
            out_orb.comp_array[idx] = tmp.comp_array[comp].real_mul(coeff)
        return out_orb

""" #Beta c**2
    def beta(self, shift = 0):
        out_orb = orbital4c()
#        beta = np.array([[orbital4c.light_speed**2 + shift, 0, 0, 0  ],
#                         [0, orbital4c.light_speed**2 + shift, 0, 0  ],
#                         [0, 0, -orbital4c.light_speed**2 + shift, 0 ],
#                         [0, 0,  0, -orbital4c.light_speed**2 + shift]])
#        out_orb.comp_array = beta@self.comp_array
        beta = np.array([orbital2c.light_speed**2 + shift,
                         orbital2c.light_speed**2 + shift,
                         orbital2c.light_speed**2 + shift,
                         orbital2c.light_speed**2 + shift])
        

        mc2 = orbital2c.light_speed**2
        

        out_orb.comp_array[0] = (mc2) * self.comp_array[2] + shift * self.comp_array[0] 
        out_orb.comp_array[1] = (mc2) * self.comp_array[3] + shift * self.comp_array[1]
        out_orb.comp_array[2] = (mc2) * self.comp_array[0] + shift * self.comp_array[2]
        out_orb.comp_array[3] = (mc2) * self.comp_array[1] + shift * self.comp_array[3]

        return out_orb
    
    def beta2(self):
        out_orb = orbital4c()
        beta = np.array([1.0,
                         1.0,
                         1.0,
                         1.0])
        
        out_orb.comp_array[0] = beta[0] * self.comp_array[2]
        out_orb.comp_array[1] = beta[1] * self.comp_array[3]
        out_orb.comp_array[2] = beta[2] * self.comp_array[0]
        out_orb.comp_array[3] = beta[3] * self.comp_array[1]
        return out_orb
     """
    
def dot(self, other):
    result = 0
    for comp in self.comp_dict.keys():
#        factor = 1
#           if('S' in comp) factor = c**2
        component = self[comp].dot(other[comp])
        result += component
    return result



""" def apply_dirac_hamiltonian(orbital, prec, shift = 0.0, der = 'BS'):
    beta_phi = orbital.beta(shift)
    grad_phi = orbital.gradient(der)
    alpx_phi = -1j * orbital4c.light_speed * grad_phi[0].alpha(0, prec)
    alpy_phi = -1j * orbital4c.light_speed * grad_phi[1].alpha(1, prec)
    alpz_phi = -1j * orbital4c.light_speed * grad_phi[2].alpha(2, prec)
    return beta_phi + alpx_phi + alpy_phi + alpz_phi
 """




def apply_potential(factor, potential, orbital, prec):
    out_orbital = orbital2c()
    for comp in orbital.comp_dict:
        if orbital[comp].squaredNorm() > 0:
            out_orbital[comp] = cf.apply_potential(factor, potential, orbital[comp], prec)
    return out_orbital

def apply_complex_potential(factor, potential, orbital, prec):
    out_orbital = orbital2c()
    for comp in orbital.comp_dict:
        if orbital[comp].squaredNorm() > 0:
            out_orbital[comp] = potential * orbital[comp] 
    return out_orbital

#
# Keep this for now to maybe enable precise addition later
#
#def add_orbitals(a, orb_a, b, orb_b, prec):
#    out_orb = orbital4c("a_plus_b",orb_a.mra)
#    for comp, func in out_orb.components.items():        
#        func_a = orb_a[comp]
#        func_b = orb_b[comp]
#        if (func_a.squaredNorm() > 0 and func_b.squaredNorm() > 0):
#            vp.advanced.add(prec/10, func, a, func_a, b, func_b)
#        elif(func_a.squaredNorm() > 0):
#            out_orb.init_function(func_a, comp)
#            func *= a
#        elif(func_b.squaredNorm() > 0):
#            out_orb.init_function(func_b, comp)
#            func *= b
#        else:
#            print('Warning: adding two empty trees')
#    return out_orb

def add_vector(orbital_array, coeff_array, prec):
    output = orbital2c()
    for comp in output.comp_dict:
        func_array = []
        for orbital in orbital_array:
            func_array.append(orbital[comp])
        output[comp] = cf.add_vector(func_array, coeff_array, prec)
    return output
        

def apply_helmholtz(orbital, mu, prec):
    out_orbital = orbital2c()
    for comp in orbital.comp_dict.keys():
        out_orbital[comp] = cf.apply_helmholtz(orbital[comp], mu, orbital2c.light_speed, prec)
    # The raw VAMPyR HelmholtzOperator output already equals the full Green's
    # function convolution integral( exp(-mu*r)/(4*pi*r) * f(r') dr' ), i.e. the
    # 4*pi is already baked into the kernel itself (confirmed against the
    # official MRCPP docs and README Sec. 2.1: G^mu * f = int exp(-mu|r-r'|)/
    # (4*pi|r-r'|) f(r') dr'). No extra rescaling should be applied here.
    return (1.0/(4*np.pi))*out_orbital

def init_1s_orbital(orbital,k,Z,n,alpha,origin,prec):
    gamma_factor = compute_gamma(k,Z,alpha)
    norm_const = compute_norm_const(n, gamma_factor)
    idx = 0
    for comp in orbital.comp_array:
        func_real = lambda x: one_s_alpha_comp([x[0]-origin[0], x[1]-origin[1], x[2]-origin[2]],
                                                Z, alpha, gamma_factor, norm_const, idx)
        func_imag = lambda x: one_s_alpha_comp([x[0]-origin[0], x[1]-origin[1], x[2]-origin[2]],
                                                Z, alpha, gamma_factor, norm_const, idx+1 )
        vp.advanced.project(prec, comp.real, func_real)
        vp.advanced.project(prec, comp.imag, func_imag)
        idx += 2
    orbital.normalize()
    return orbital

def compute_gamma(k,Z,alpha):
    return np.sqrt(k**2 - Z**2 * alpha**2)

def compute_norm_const(n, gamma_factor):
# THIS NORMALIZATION CONSTANT IS FROM WIKIPEDIA BUT IT DOES NOT AGREE WITH Bethe&Salpeter
# and most importantly, it is wrong :-)
    tmp1 = 2 * n * (n + gamma_factor)
    tmp2 = 1 / (gamma_factor * gamma(2 * gamma_factor))
    return np.sqrt(tmp2/tmp1)

def one_s_alpha(x,Z,alpha,gamma_factor):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    tmp1 = 1.0 + gamma_factor
    tmp4 = Z * alpha
    u = x/r
    lar =   tmp1
    sai =   tmp4 * u[2]
    sbr = - tmp4 * u[1]
    sbi =   tmp4 * u[0]
    return lar, 0, 0, 0, 0, sai, sbr, sbi

def one_s_alpha_comp(x,Z,alpha,gamma_factor,norm_const,comp):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    tmp2 = r ** (gamma_factor - 1)
    tmp3 = np.exp(-Z*r)
    values = one_s_alpha(x,Z,alpha,gamma_factor)
    return values[comp] * tmp2 * tmp3 * norm_const / np.sqrt(2*np.pi)

def sigma_gradient(orbital, derivative, prec):
    out = orbital2c()
    grad_vec = orbital.gradient(der = derivative)
    sigma_vec = {}
    for i in range(3):
        sigma_vec[i] = grad_vec[i].sigma(i, prec)
    out = sigma_vec[0] + sigma_vec[1] + sigma_vec[2]
    return out

def Diagonal_Term_Weyl_Hamiltonian(Weyl_Spinor, prec, potential, derivative, chirality_L = True):
    # Will compute V \pm c * sigma . p, depending on the chirality of the Weyl spinor
    light_speed = orbital2c.light_speed
    VPsi = apply_potential(-1.0, potential, Weyl_Spinor, prec)
    sigma_p_Weyl = Weyl_Spinor.sigma_p(prec, derivative)
    
    if chirality_L:
        result = VPsi + light_speed * sigma_p_Weyl
    else:
        result = VPsi - light_speed * sigma_p_Weyl
    return result

def calc_energy_Weyl_2c(Psi_L, Psi_R, potential, prec):
    light_speed = orbital2c.light_speed
    #print("     Light speed in calc_energy_Weyl_2c:", light_speed)

    braket_LL = Psi_L.squaredNorm()
    #print("     braket_LL", braket_LL)

    braket_LR = dot(Psi_L,Psi_R).real
    #print("     braket_LR", braket_LR)

    braket_RR = Psi_R.squaredNorm()
    #print("     braket_RR", braket_RR)

    Dirac_Sq_Norm = braket_LL + braket_RR 

    tmp = Diagonal_Term_Weyl_Hamiltonian(Psi_L, prec, potential, 'BS', chirality_L = True)
    tmp2 = Diagonal_Term_Weyl_Hamiltonian(Psi_R, prec, potential, 'BS', chirality_L = False)

    exp_val_L = dot(Psi_L,tmp).real
    #print("     exp_val_L", exp_val_L/(Dirac_Sq_Norm))
    exp_val_R = dot(Psi_R,tmp2).real
    #print("     exp_val_R", exp_val_R/(Dirac_Sq_Norm))

    energy = exp_val_R + exp_val_L + 2 * braket_LR * (light_speed**2)
    #print("     energy numerator", energy)
    energy *= (1.0/(Dirac_Sq_Norm))
    
    return energy

#def overlap_density_balanced(spinor_array, V_psi_array, pi_psi_array, prec):


def calc_dirac_mu(energy, light_speed, verbose = False):
    val = (light_speed**4-energy**2)/light_speed**2
    if val < 0:
        raise ValueError("Negative value under square root in calc_dirac_mu:", val)   
        

    mu = np.sqrt(val)
    if verbose:
        print("-> \mu = ", mu, "| E =",  energy - light_speed**2, "| using c =", light_speed)
    return mu

def calc_kutzelnigg_mu(energy_sq, light_speed):
    c2 = light_speed**2
    val = energy_sq/c2 - c2
    return np.sqrt(-val)

def calc_non_rel_mu(energy):
    if energy < 0:
        return np.sqrt(-2.0 * energy)
    else:
        print("Positive energy")
        exit(-1)

def print_norm_debug(orbital):
    print("Component norms:")
    for comp in orbital.comp_dict.keys():
        print(f" - {comp}: {orbital[comp].squaredNorm()}")


def Dirac_to_Weyl(dirac_spinor, weyl_L = True):
    # Take the 4-component Dirac spinor and return the 2-component Weyl spinor. If weyl_L is True, return the left-handed Weyl spinor, otherwise return the right-handed Weyl spinor.
    weyl_spinor = orbital2c()
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
