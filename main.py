import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import quad

class Physical_quantities:
    def __init__(self):
        self.q = 1.6e-19 # заряд в СИ
        self.k = 8.617e-5 # постоянная Больцмана в эВ
        self.h = 4.135e-15 
        self.m = 9.1e-31 # масса электрона в СИ
        self.k_si = 1.38e-23 # постоянная Больцмана в СИ
        self.epsilond_0 = 8.85E-14 # Ф / см

class Initial_param:
    def __init__(self):
        self.I_kbo = 3 # mkA
        self.h_21e = 50
        self.f_gr = 800 # MHz
        self.U_ke_max = 15 # V
        self.I_k_max = 15 # mA
        self.t_j_max = 40 # oC
        self.Eg = 1.12 # eV
        self.n_i = 1.5E10 # cm^(-3)
        self.epsilond = 11.7
        self.r = 0.1 # Ом*см определяем из графика после того, как нашли Nd_vk
        self.C_k = 2.5E-12
        self.C_e = 2.5E-12
        self.fi = 0.9 # eV
        self.Na_xjz = 5E18 # cm^(-3)
        self.Na_xje = 5E18 # cm^(-3)
        self.D_n_xe = 7 # cm^4 / c
        self.Nd_0 = 2E21 # cm^(-3)
        self.Na_0 = 5E20 # cm^(-3)
        self.mu_p_Na_x_j_e = 200 # cm^2 / (В * с)
        self.mu_n_Nd_bk = 500 # cm^2 / (В * с)
        self.vs = 2E7 # cm / c
        self.po_vk = 0.2
        self.po_nk = 4E-4

class Culculation_param():
    def __init__(self):
        Physical_quantities.__init__(self)
        Initial_param.__init__(self)

    def find_U_kb_prob(self):
        return 2 * self.U_ke_max
    
    def find_Nd_bk(self):
        return 1E16 * ((self.find_U_kb_prob() / 60) * ((self.Eg / 1.1) ** (-3 / 2))) ** (-4 / 3)
    
    def find_W_opz_k(self):
        return (2 * self.epsilond_0 * self.epsilond * self.U_ke_max / self.q / self.find_Nd_bk()) ** 0.5
    
    def find_A_k(self):
        return (self.C_k * self.find_W_opz_k()) / (self.epsilond_0 * self.epsilond)

    def find_W_opz_k_max(self):
        return (2 * self.epsilond_0 * self.epsilond * self.find_U_kb_prob() / self.q / self.find_Nd_bk()) ** 0.5

    def find_h_k(self):
        return self.find_W_opz_k_max() * 10000 + 5

    def find_W_opz_e(self):
        return (2 * self.epsilond_0 * self.epsilond * self.fi * 0.5 / self.q / self.Na_xjz) ** 0.5

    def find_A_e(self):
        return (self.C_e * self.find_W_opz_e()) / (self.epsilond_0 * self.epsilond)
    
    def find_t_dreif_pr_b(self):
        return 1 / (2 * 3.14 * self.f_gr * 1E6)

    def find_koef_perapada(self):
        return np.log(self.Na_xje / self.find_Nd_bk())

    def find_W_bo(self):
        return (self.find_koef_perapada() * self.find_t_dreif_pr_b() * self.D_n_xe) ** 0.5

    def find_Ld(self, x_j_e, Na_x_j_e):
        return x_j_e / (np.log(self.Nd_0 / Na_x_j_e))

    def find_La(self, x_j_e, Na_x_j_e):
        return self.find_W_bo() / (np.log(Na_x_j_e / self.find_Nd_bk()))

    def find_Nd_x(self, x, Ld):
        return self.Nd_0 * np.exp(- x / Ld)

    def find_Na_x(self, x, x_j_e, La, Na_x_j_e):
        return Na_x_j_e * np.exp( - (x - x_j_e) / (La))
    
    # Рассчитаем интеграл для эффективной концентрации доноров в эмиттере с учетом компенсации и найдем это значение
    def find_Nd_eff(self, x_j_e, La, Ld, Na_x_j_e):
        def Nd_eff(x):
            def Nd_x(x):
                return self.Nd_0 * np.exp(- x / Ld)

            def Na_x(x):
                return Na_x_j_e * np.exp( - (x - x_j_e) / (La))
            
            return Nd_x(x) - Na_x(x) + self.find_Nd_bk()
        
        return (quad(Nd_eff, 0, x_j_e)[0]) / x_j_e

    # Рассчитаем интеграл для эффективной концентрации акцепторов в базе с учетом компенсации и найдем это значение
    def find_Na_eff(self, x_j_e, La, Ld, Na_x_j_e):
        def Nd_eff(x):
            def Nd_x(x):
                return self.Nd_0 * np.exp(- x / Ld)

            def Na_x(x):
                return Na_x_j_e * np.exp( - (x - x_j_e) / (La))
            
            return Na_x(x) - Nd_x(x) - self.find_Nd_bk()
        
        return (quad(Nd_eff, x_j_e, x_j_e + self.find_W_bo())[0]) / self.find_W_bo()
    
    def find_fi(self, Ndonor, Nacteptor):
        return 0.026 * np.log(Ndonor * Nacteptor / (self.n_i ** 2))
    
    def find_x_n(self, fi, Ndonor, Nacteptor):
        return (2 * self.epsilond_0 * self.epsilond * fi * Nacteptor / (self.q * Ndonor * (Ndonor + Nacteptor))) ** 0.5
    
    def find_x_p(self, fi, Ndonor, Nacteptor):
        return (2 * self.epsilond_0 * self.epsilond * fi * Ndonor / (self.q * Nacteptor * (Ndonor + Nacteptor))) ** 0.5
    
    def find_W(self, xn, xp):
        return xn + xp
    
    def find_x_n_work(self, U, fi, Ndonor, Nacteptor):
        return (2 * self.epsilond_0 * self.epsilond * (fi + U) * Nacteptor / (self.q * Ndonor * (Ndonor + Nacteptor))) ** 0.5
    
    def find_x_p_work(self, U, fi, Ndonor, Nacteptor):
        return (2 * self.epsilond_0 * self.epsilond * (fi + U) * Ndonor / (self.q * Nacteptor * (Ndonor + Nacteptor))) ** 0.5
    
    def find_y(self, x_j_k, W_bc):
        return x_j_k / W_bc
    
    def find_Ucb_prob_sfer(self, y):
        n = 2
        return 60 * (self.Eg / 1.1) ** 1.5 * (self.find_Nd_bk() / 1E16) ** (-3 / 4) * (((n + 1 + y) * y ** n) ** (1 / (n + 1)) - y)

    def find_La_p_plus(self, x, Na_eff):
        return x / (np.log(self.Na_0 / Na_eff))
    
    def find_I_e(self, U_e, x_e, x_k, x_j_e, La, Ld, Na_x_j_e):
        def integral(x):
            def Nd_x(x):
                return self.Nd_0 * np.exp(- x / Ld)

            def Na_x(x):
                return Na_x_j_e * np.exp( - (x - x_j_e) / (La))

            Dp = 0.026 * self.mu_p_Na_x_j_e
            Dn = 0.026 * self.mu_n_Nd_bk
            return (Na_x(x) - Nd_x(x)) / (Dn - Dp)
        
        solution = (quad(integral, x_e, x_k))[0]

        return self.find_A_e() * 1.6E-19 * self.n_i ** 2 * np.exp(U_e / 0.026) / solution

    def find_r_e(self, I_e):
        return 0.026 / I_e
    
    def find_talu_e(self, r_e):
        return r_e * self.C_e

    def find_E_usk(self):
        return 0.026 * self.find_koef_perapada() / self.find_W_bo()
    
    def find_t_pr_k(self):
        return self.find_W_opz_k() / self.vs

    def find_y_0(self, La, Ld, talu_p):
        Dn = 0.026 * self.mu_n_Nd_bk
        
        return 1 - 1.7 * La * Ld / (Dn * talu_p)
    
    def find_y_omega(self, omega, y0, talu_e):
        omega_y = 1 / talu_e

        return y0 / ((1 + (omega / omega_y) ** 2) ** 0.5)
    
    def find_betta_0(self, W_b, W_b0, Ln_x_e):
        return 1 - W_b * W_b0 / (Ln_x_e ** 2 * self.find_koef_perapada())
    
    def find_betta_omega(self, omega, betta_0):
        omega_betta = 1 / (betta_0 * self.find_t_dreif_pr_b())

        return betta_0 / ((1 + (omega / omega_betta) ** 2) ** 0.5)
    
    def find_h_21_b_0(self, y_0, betta_0):
        return y_0 * betta_0
    
    def find_h_21_b_omega(self, omega, h_21_b_0):
        omega_h_21_b_0 = 1 / self.find_t_dreif_pr_b()

        return h_21_b_0 / ((1 + (omega / omega_h_21_b_0) ** 2) ** 0.5)
    
    def find_h_21_e_0(self, h_21_b_0):
        return h_21_b_0 / (1 - h_21_b_0)
    
    def find_h_21_e_omega(self, omega, h_21_e_o):
        omega_h_21_b_0 = 1 / self.find_t_dreif_pr_b()
        omega_h_21_e_0 = omega_h_21_b_0 / h_21_e_o

        return h_21_e_o / ((1 + (omega / omega_h_21_e_0) ** 2) ** 0.5)
    
    def find_C_e_d(self, I_e):
        return 0.026 * I_e * self.find_t_dreif_pr_b()
    
    def find_I_gen(self, I_e, h_21b):
        return I_e * h_21b

    def find_C(self, A, W):
        return self.epsilond * self.epsilond_0 * A / W

    def find_r_n_minus(self, W_epi, x_j_k, W_opz_k, I_e, A_e):
        r = (A_e / 3.14) ** 0.5

        return self.po_vk * (W_epi - x_j_k - W_opz_k) / (I_e * r)

    def find_r_n_plus(self, W_podl, I_e, A_e):
        r = (A_e / 3.14) ** 0.5

        return self.po_nk * W_podl / (I_e * r)
    
    def find_r_ka(self, r_n_minus, r_n_plus, R_kn_plus):
        return r_n_minus + r_n_plus + R_kn_plus
    

class Draw_graph(Culculation_param):
    def draw_N_x(self, x_j_e, x_j_k, Na_x_j_e, Nd_bk, Ld, La, hk):
        x_list = list()
        Nd_list = list()
        Na_list = list()
        Nd_bk_list = list()
        koef = 1.125 # просто коэффициент, насколько отложить ось х после того, как она достигнет значения hk

        for x in np.arange(0, hk, 1E-7):
            x_list.append(x)
            Nd_list.append(self.find_Nd_x(x, Ld))
            Na_list.append(self.find_Na_x(x, x_j_e, La, Na_x_j_e))
            Nd_bk_list.append(Nd_bk)
        
        for x in np.arange(hk, koef * hk, 1E-7):
            x_list.append(x)
            Nd_list.append(self.find_Nd_x(x, Ld))
            Na_list.append(self.find_Na_x(x, x_j_e, La, Na_x_j_e))
            Nd_bk_list.append(self.Nd_0)
            
        fig, axes = plt.subplots()

        axes.plot(x_list, np.log10(Nd_list), color='b', label='Nd(x)', linewidth=1.5)
        axes.plot(x_list, np.log10(Nd_bk_list), color='y', label='Nd_bk(x)', linewidth=1.5)
        axes.plot(x_list, np.log10(Na_list), color='r', label='Na(x)', linewidth=1.5)

        axes.plot([x_j_e, x_j_e], [0, self.Nd_0], color='g', linestyle = '--', linewidth=1)
        axes.plot([x_j_k, x_j_k], [0, self.Nd_0], color='g', linestyle = '--', linewidth=1)   

        axes.set(xlim=(0, koef * hk))
        axes.set(ylim=(15, 22))

        plt.title('Распределение примеси')
        plt.xlabel('х, см')
        plt.ylabel('log10(N)')
        plt.legend(loc=5)

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_N_complex_x(self, x_j_e, x_j_k, Na_x_j_e, Nd_bk, Ld, La, hk):
        x_list = list()
        N_complex_list = list()
        koef = 1.125 # просто коэффициент, насколько отложить ось х после того, как она достигнет значения hk

        for x in np.arange(0, x_j_e, 1E-7):
            x_list.append(x)
            N_complex_list.append(self.find_Nd_x(x, Ld) - self.find_Na_x(x, x_j_e, La, Na_x_j_e) + Nd_bk)

        for x in np.arange(x_j_e, x_j_k, 1E-7):
            x_list.append(x)
            N_complex_list.append(self.find_Na_x(x, x_j_e, La, Na_x_j_e) - Nd_bk - self.find_Nd_x(x, Ld))

        for x in np.arange(x_j_k, hk, 1E-7):
            x_list.append(x)
            N_complex_list.append(Nd_bk + self.find_Nd_x(x, Ld) - self.find_Na_x(x, x_j_e, La, Na_x_j_e))

        for x in np.arange(hk, hk * koef, 1E-7):
            x_list.append(x)
            N_complex_list.append(self.Nd_0 + self.find_Nd_x(x, Ld) - self.find_Na_x(x, x_j_e, La, Na_x_j_e))

        fig, axes = plt.subplots()

        axes.plot(x_list, np.log10(N_complex_list), color='b', label='N(x)', linewidth=1.5)

        axes.plot([x_j_e, x_j_e], [0, self.Nd_0], color='g', linestyle = '--', linewidth=1)
        axes.plot([x_j_k, x_j_k], [0, self.Nd_0], color='g', linestyle = '--', linewidth=1)   

        axes.set(xlim=(0, koef * hk))
        axes.set(ylim=(15, 22))

        plt.title('Результурующее распределение примеси')
        plt.xlabel('х, см')
        plt.ylabel('log10(N)')
        plt.legend(loc=5)

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_y_omega(self, y_0, talu_e):
        y_list = list()
        omega_list = list()
        omega_lim = 1E5

        for omega in np.arange(0, omega_lim, 10):
            omega_list.append(omega)
            y_list.append(self.find_y_omega(omega, y_0, talu_e))

        fig, axes = plt.subplots()

        axes.plot(omega_list, y_list, color='b', label='y(omega)', linewidth=1.5)

        axes.set(xlim=(0, omega_lim))

        plt.title('Частотная зависимость коэффициента инжекции эмиттера')
        plt.xlabel('omega, Гц')
        plt.ylabel('y')
        plt.legend(loc=5)

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_betta_omega(self, betta_0):
        y_list = list()
        omega_list = list()
        omega_lim = 1E8

        for omega in np.arange(0, omega_lim, 1000):
            omega_list.append(omega)
            y_list.append(self.find_betta_omega(omega, betta_0))

        fig, axes = plt.subplots()

        axes.plot(omega_list, y_list, color='b', label='y(omega)', linewidth=1.5)

        axes.set(xlim=(0, omega_lim))

        plt.title('Частотная зависимость коэффициента переноса')
        plt.xlabel('omega, Гц')
        plt.ylabel('betta')
        plt.legend(loc=5)

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_h_21_b_omega(self, h_21_b_0):
        y_list = list()
        omega_list = list()
        omega_lim = 1E8

        for omega in np.arange(0, omega_lim, 1000):
            omega_list.append(omega)
            y_list.append(self.find_h_21_b_omega(omega, h_21_b_0))

        fig, axes = plt.subplots()

        axes.plot(omega_list, y_list, color='b', label='h_21_b(omega)', linewidth=1.5)

        axes.set(xlim=(0, omega_lim))

        plt.title('Частотная зависимость h21б')
        plt.xlabel('omega, Гц')
        plt.ylabel('h21b')
        plt.legend(loc=5)

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

    def draw_h_21_e_omega(self, h_21_e_0):
        y_list = list()
        omega_list = list()
        omega_lim = 1E7

        for omega in np.arange(0, omega_lim, 1000):
            omega_list.append(omega)
            y_list.append(self.find_h_21_e_omega(omega, h_21_e_0))

        fig, axes = plt.subplots()

        axes.plot(omega_list, y_list, color='b', label='h_21_b(omega)', linewidth=1.5)

        axes.set(xlim=(0, omega_lim))
        axes.set(ylim=(0))

        plt.title('Частотная зависимость h21e')
        plt.xlabel('omega, Гц')
        plt.ylabel('h21e')
        plt.legend(loc=5)

        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

def main():
    transistor = Culculation_param()

    # задаем вручную
    x_j_e = 4.1E-5
    x_j_k = x_j_e + transistor.find_W_bo()
    Na_x_j_e = 5E17

    Nd_bk = transistor.find_Nd_bk()
    Ld = transistor.find_Ld(x_j_e, Na_x_j_e)
    La = transistor.find_La(x_j_e, Na_x_j_e)
    hk = transistor.find_h_k() * 1E-4 # см

    draw_class = Draw_graph()
    print('=======================PART-1==========================')
    print(f'Uкб проб = {transistor.find_U_kb_prob()} В')
    print(f'Ndвк = {transistor.find_Nd_bk()} см^(-3)')

    W_opz_k = transistor.find_W_opz_k()
    print(f'Wопз к = {transistor.find_W_opz_k()} см')
    A_k = transistor.find_A_k()
    print(f'Aк = {A_k} см^2')
    print(f'Wопз к макс = {transistor.find_W_opz_k_max()} см')

    h_k = transistor.find_h_k() * 1E-4
    print(f'hk = {transistor.find_h_k()} мкм = {transistor.find_h_k() * 1E-4} см')
    print(f'Wопз э = {transistor.find_W_opz_e()} см')

    A_e = transistor.find_A_e()

    print(f'Aэ = {A_e} см^2')
    print(f't_дрейф_пр_б = {transistor.find_t_dreif_pr_b()} с')
    print(f'Коэффициент перепада концентрации = {transistor.find_koef_perapada()}')
    print(f'Wбо = {transistor.find_W_bo()} см')

    print('=======================PART-2==========================')

    print(f'mu_p_Na_x_j_e = {transistor.mu_p_Na_x_j_e} cм^2 / (В * с)')
    print(f'mu_n_Nd_bk = {transistor.mu_n_Nd_bk} cм^2 / (В * с)')
    print(f'Ld = {Ld} см')
    print(f'La = {La} см')
    print(f'La / Ld = {La / Ld}')

    Nd_eff = transistor.find_Nd_eff(x_j_e, La, Ld, Na_x_j_e)
    Na_eff = transistor.find_Na_eff(x_j_e, La, Ld, Na_x_j_e)

    print(f'Nd_eff = {Nd_eff}')
    print(f'Na_eff = {Na_eff}')

    fi_eb = transistor.find_fi(Nd_eff, Na_eff)
    fi_bc = transistor.find_fi(Nd_bk, Na_eff)

    print(f'fi_eb = {fi_eb} В')
    print(f'fi_bc = {fi_bc} В')

    x_n_eb = transistor.find_x_n(fi_eb, Nd_eff, Na_eff)
    x_p_eb = transistor.find_x_p(fi_eb, Nd_eff, Na_eff)

    print(f'x_n_eb = {x_n_eb} см')
    print(f'x_p_eb = {x_p_eb} см')

    x_n_bc = transistor.find_x_n(fi_bc, Nd_bk, Na_eff)
    x_p_bc = transistor.find_x_p(fi_bc, Nd_bk, Na_eff)

    print(f'x_n_bc = {x_n_bc} см')
    print(f'x_p_bc = {x_p_bc} см')

    W_eb = x_n_eb + x_p_eb
    W_bc = x_n_bc + x_p_bc

    print(f'W_be = {W_eb} см')
    print(f'W_bc = {W_bc} см')

    x_n_bc_work = transistor.find_x_n_work(0.5 * fi_bc, fi_bc, Nd_bk, Na_eff)
    x_p_bc_work = transistor.find_x_p_work(0.5 * fi_bc, fi_bc, Nd_bk, Na_eff)

    print(f'x_n_bc_work = {x_n_bc_work} см')
    print(f'x_p_bc_work = {x_p_bc_work} см')

    W_bc_work = x_n_bc_work + x_p_bc_work

    print(f'W_bc_work = {W_bc_work} см')

    y = transistor.find_y(x_j_e + transistor.find_W_bo(), W_bc)

    print(f'y = {y}')
    
    Ucb_prob_sfer = transistor.find_Ucb_prob_sfer(y)

    print(f'Ucb_prob_sfer = {Ucb_prob_sfer} В')

    La_p_plus = transistor.find_La_p_plus(4E-4, Na_eff)

    print(f'La_p_+ = {La_p_plus} см')

    I_e = transistor.find_I_e(0.5 * fi_eb, x_j_e, x_j_e + transistor.find_W_bo(), x_j_e, La, Ld, Na_x_j_e)

    print(f'Iэ = {I_e} A')

    r_e = transistor.find_r_e(I_e)

    print(f'rэ\' = {r_e}')

    talu_e = transistor.find_talu_e(r_e)

    print(f'talu_э = {talu_e}')

    E_usk = transistor.find_E_usk()

    print(f'Eуск = {E_usk}')

    t_pr_k = transistor.find_t_pr_k()

    print(f't пролета ОПЗ коллектора = {t_pr_k} c')

    talu_p = 1E-6
    y_0 = transistor.find_y_0(La, Ld, talu_p)

    print(f'y_0 = {y_0}')

    Ln_x_e = 5E-2
    betta_0 = transistor.find_betta_0(x_j_k - x_j_e, transistor.find_W_bo(), Ln_x_e)

    print(f'betta_0 = {betta_0}')

    h_21_b_0 = transistor.find_h_21_b_0(y_0, betta_0)
    
    print(f'h_21_b_0 = {h_21_b_0}')

    h_21_e_0 = transistor.find_h_21_e_0(h_21_b_0)
    
    print(f'h_21_e_0 = {h_21_e_0}')

    print('=======================PART-3==========================')

    print(f'Cэб = {transistor.C_e} пФ')
    print(f'rэ\' = {r_e}')

    C_e_d = transistor.find_C_e_d(I_e)

    print(f'Ced = {C_e_d} Ф')
    
    h_21b = transistor.h_21e / (1 + transistor.h_21e)

    print(f'h21b = {h_21b}')

    I_gen = transistor.find_I_gen(I_e, h_21b)

    print(f'I_gen = {I_gen} A')

    A_ca = A_e
    A_cp = A_k - A_e
    C_ca = transistor.find_C(A_ca, W_bc)
    C_cp = transistor.find_C(A_cp, W_bc)
    print(f'A_ca = {A_ca} см^2')
    print(f'A_cp = {A_cp} см^2')
    print(f'C_ca = {C_ca} Ф')
    print(f'C_cp = {C_cp} Ф')
    
    W_epi = h_k + W_bc
    print(f'W_epi = {W_epi} см')

    r_n_minus = transistor.find_r_n_minus(W_epi, x_j_k, W_opz_k, I_e, A_e)
    r_n_plus = transistor.find_r_n_plus(h_k, I_e, A_e)
    print(f'r_n_minus = {r_n_minus}')
    print(f'r_n_plus = {r_n_plus}')


    draw_class.draw_N_x(x_j_e, x_j_k, Na_x_j_e, Nd_bk, Ld, La, hk)
    draw_class.draw_N_complex_x(x_j_e, x_j_k, Na_x_j_e, Nd_bk, Ld, La, hk)
    draw_class.draw_y_omega(y_0, talu_e)
    draw_class.draw_betta_omega(betta_0)
    draw_class.draw_h_21_b_omega(h_21_b_0)
    draw_class.draw_h_21_e_omega(h_21_e_0)

if __name__ == "__main__":
    main()
