import numpy as np

class MinimumGlacierModel():
    """ Base class for minimum glacier models """
    def __init__(self, calving=True, alpha=3., beta=0.007, nu=10.):
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.W = 1. # meters

        self.L_last = 1. # meters # initial value
        self.t_last = 0. # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1/20.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0, 1e5, 101)
        self.y = self.x


    def mean_ice_thickness(self):
        s_mean = self.mean_slope()
        return self.alpha / (1+self.nu*s_mean) * np.power(self.L_last, 1/2.)


    def water_depth(self):
        return -1. * np.min([0, self.bed(self.L_last)])


    def calving_flux(self):
        d = self.water_depth()
        H_mean = self.mean_ice_thickness()
        Hf = np.max([self.kappa * H_mean, self.rho_water / self.rho_ice * d])
        F = -self.c * d * Hf * self.W
        return F


    def mean_balance(self):
        b_mean = self.mean_bed()
        H_mean = self.mean_ice_thickness()
        return self.beta * (b_mean + H_mean - self.E) * self.W * self.L_last


    def change_L(self):
        s_mean = self.mean_slope()
        ds = self.slope_gradient(self.L_last)
        Bs = self.mean_balance()
        if self.calving:
            F = self.calving_flux()
        else: 
            F = 0.
        dldt_1 = 3. * self.alpha / (2*(1+self.nu*s_mean)) * np.power(self.L_last, 1/2.)
        dldt_2 = - self.alpha * self.nu / np.power(1+self.nu*s_mean, 2.) * np.power(self.L_last, 3/2.) * ds
        dldt_3 = Bs + F
        return np.power(dldt_1 + dldt_2, -1.) * dldt_3


    def integrate(self, dt, time, E=None, E_new=None, show=False):
        if E is None:
            E = self.E
        else:
            self.E = E

        i_max = np.int(np.round(time/dt))

        if E and E_new:
            E_list = np.linspace(E, E_new, i_max)
        else:
            E_list = None

        t = np.zeros(i_max+1, dtype=np.float)
        x = np.zeros(i_max+1, dtype=np.float)

        self.t_last = self.t[-1]
        self.L_last = self.L[-1]

        t[0] = self.t_last
        x[0] = self.L_last

        if show: print("Start integration for t from {:0.2f} to {:0.2f} with dt = {:0.2f}, E0 = {:0.2f} and L0 = {:0.2f}".format(t[0], t[0]+time, dt, self.E, x[0]))

        for i in range(i_max):
            if E_list is not None:
                self.E = E_list[i]

            t[i+1] = t[i] + dt
            x[i+1] = np.max([x[i] + dt * self.change_L(), 0.])
            self.L_last = x[i+1]

        if show: print("Integration finished")

        self.t = np.append(self.t, t[1:])
        self.L = np.append(self.L, x[1:])

        self.t_last = self.t[-1]
        self.L_last = self.L[-1]

        if E_list is not None:
            self.E_data = np.append(self.E_data, E_list)
        else:
            self.E_data = np.append(self.E_data, np.array([self.E]*i_max))



class LinearBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a linear bed """
    def __init__(self, b0=3900, s=0.1, calving=True, alpha=3., beta=0.007, nu=10.):
        self.b0 = b0
        self.s = s
        
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.W = 1.  # meters

        self.L_last = 1.  # meters # initial value
        self.t_last = 0.  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1/20.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0, 1e5, 101)
        self.y = self.bed(self.x)


    def __str__(self):
        return "Minimum Glacier Model for a linear bed."


    def bed(self, x):
        return self.b0 - self.s * x


    def mean_bed(self):
        return self.b0 - self.s * self.L_last / 2


    def slope_gradient(self, x):
        return 0.


    def mean_slope(self):
        return self.s
