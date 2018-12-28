import matplotlib.pyplot as plt
import numpy as np

__all__ = ["LinearBedModel", "ConcaveBedModel", "CustomBedModel"]


class MinimumGlacierModel():
    """ Base class for minimum glacier models """

    def __init__(self, calving=True, L0=1., t0=0.):
        # should not use this object to create a glacier
        self.alpha = 3.
        self.beta = 0.007
        self.nu = 10.
        self.W = 1. # meters

        self.L_last = L0  # meters # initial value
        self.t_last = t0  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1/200.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0., 1e5)
        self.y = -1.*self.x + np.max(self.x)/2.


    def __str__(self):
        return "Base class for a minimum Glacier Model."

    def bed(self, x):
        """ Returns the elevation of the bed at point x """
        return 0.

    def mean_bed(self, x=None):
        """ Returns the mean elevation of the bed from 0 to x """
        if x is None:
            x = self.L_last
        return 0.

    def slope(self, x):
        """ Returns the slope of the bed at point x """
        return 0.

    def mean_slope(self, x=None):
        """ Returns the mean slope of the bed from 0 to x """
        if x is None:
            x = self.L_last
        return 0.

    def d_slope_d_L(self, L=None):
        """ Returns the change of the mean slope if the value of L changes """
        if L is None:
            L = self.L_last
        return 0.

    def mean_ice_thickness(self, L=None):
        """ Calculates the mean ice thickness Hm with equation 1.2.2 """
        if L is None:
            L = self.L_last
        s_mean = self.mean_slope(L)
        return self.alpha / (1.+self.nu*s_mean) * np.power(L, 1./2.)

    def water_depth(self, x=None):
        """ Determines the water depth at x by checking if the elevation of the bed is negative """
        if x is None:
            x = self.L_last
        return -1. * np.min([0., self.bed(x)])

    def calving_flux(self, L=None):
        """ Calculates the calving flux using equations 1.5.1, 1.5.2 and 1.5.3 """
        if L is None:
            L = self.L_last
        d = self.water_depth(L)
        H_mean = self.mean_ice_thickness(L)
        Hf = np.max([self.kappa * H_mean, self.rho_water / self.rho_ice * d])
        F = -self.c * d * Hf * self.W
        return F

    def mass_balance(self, L=None):
        """ Returns the mass balance from equation 1.3.0 """
        if L is None:
            L = self.L_last
        b_mean = self.mean_bed(L)
        H_mean = self.mean_ice_thickness(L)
        return self.beta * (b_mean + H_mean - self.E) * self.W * L

    def change_L(self, L=None):
        """ Determines dL/dt from equation 1.2.5 """
        if L is None:
            L = self.L_last
        s_mean = self.mean_slope(L)
        ds = self.d_slope_d_L(L)
        Bs = self.mass_balance(L)
        if self.calving:
            F = self.calving_flux(L)
        else:
            F = 0.
        dldt_1 = 3. * self.alpha / (2*(1+self.nu*s_mean)) * np.power(L, 1./2.)
        dldt_2 = - self.alpha * self.nu / np.power(1+self.nu*s_mean, 2.) * np.power(L, 3./2.) * ds
        dldt_3 = Bs + F
        return np.power(dldt_1 + dldt_2, -1.) * dldt_3

    def integrate(self, dt, time, E=None, E_new=None, show=False):
        """ Function to integrate in time using a forward Euler scheme """
        if E is None:
            E = self.E
        else:
            self.E = E

        if np.size(self.E_data) == 1:
            self.E_data = np.array([self.E], dtype=np.float)

        if dt > time:
            dt, time = time, dt
            print("Switching dt and time, so dt={} and time={}".format(dt, time))

        i_max = np.int(np.round(time/dt))

        if E_new:
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


    def plot(self, E=False, C=False, V=False, show=True, filename=None):
        """ Create a figure with glacier length and other data """
        p = 1 # count number of subplots
        c = 0 # current count
        if E is True:
            p += 1
        if C is True:
            p += 1
        if V is True:
            p += 1
        
        # create figure
        fig = plt.figure()
        # fig.set_title("Glacier length")
        fig.set_size_inches(10, 4*p)
        # fig.set_dpi(300)
        fig.set_tight_layout(True)

        subplots = []

        # length
        c += 1
        print("L: {}/{}".format(c, p))
        sub1 = plt.subplot(p, 1, c)
        subplots.append(sub1)
        sub1.plot(self.t, self.L)
        if c == p: 
            sub1.set_xlabel("Time [y]")
        sub1.set_ylabel("Length [m]")
        sub1.grid()

        if E is True:
            c += 1
            print("E: {}/{}".format(c, p))
            sub_E = plt.subplot(p, 1, c)
            subplots.append(sub_E)
            sub_E.plot(self.t, self.E_data, label="E")
            sub_E.plot(self.t, np.max(self.y)*np.ones(np.size(self.t)), label="Top of mountain")
            sub_E.plot(self.t, self.bed(self.L), label="End of glacier")
            if c == p:
                sub_E.set_xlabel("Time [y]")
            sub_E.set_ylabel("Height [m]")
            sub_E.grid()
            sub_E.legend()

        if C is True:
            c += 1
            print("C: {}/{}".format(c, p))
            sub_C = plt.subplot(p, 1, c)
            subplots.append(sub_C)
            sub_C.plot(self.t, [self.calving_flux(Li) for Li in self.L])
            if c == p:
                sub_C.set_xlabel("Time [y]")
            sub_C.set_ylabel("Calving Flux []")
            sub_C.grid()

        if V is True:
            c += 1
            print("V: {}/{}".format(c, p))
            sub_V = plt.subplot(p, 1, c)
            subplots.append(sub_V)
            sub_V.plot(self.t, [self.mean_ice_thickness(Li) * Li for Li in self.L])
            if c == p:
                sub_V.set_xlabel("Time [y]")
            sub_V.set_ylabel("Ice volume [m$^3$]")
            sub_V.grid()


        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()


class LinearBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a linear bed """

    def __init__(self, b0=3900., s=0.1, calving=True, L0=1., t0=0.):
        self.b0 = b0
        self.s = s

        self.alpha = 3.
        self.beta = 0.007
        self.nu = 10.
        self.W = 1. # meters

        self.L_last = L0  # meters # initial value
        self.t_last = t0  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1./200.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0., 1e5, 101)
        self.y = self.bed(self.x)

    def __str__(self):
        return "Minimum Glacier Model for a linear bed."

    def bed(self, x):
        """ Returns the elevation of the bed at point x """
        return self.b0 - self.s * x

    def mean_bed(self, x=None):
        """ Returns the mean elevation of the bed from 0 to x """
        if x is None:
            x = self.L_last
        return self.b0 - self.s * x / 2.

    def slope(self, x):
        """ Returns the slope of the bed at point x """
        return self.s

    def mean_slope(self, x=None):
        """ Returns the mean slope of the bed from 0 to x """
        if x is None:
            x = self.L_last
        return self.s

    def d_slope_d_L(self, L=None):
        """ Returns the change of the mean slope if the value of L changes """
        if L is None:
            L = self.L_last
        return 0.


class ConcaveBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a concave bed """

    def __init__(self, b0=3900., ba=-100., xl=7000., calving=True, L0=1., t0=0.):
        self.b0 = b0
        self.ba = ba
        self.xl = xl

        self.alpha = 3.
        self.beta = 0.007
        self.nu = 10.
        self.W = 1. # meters

        self.L_last = L0  # meters # initial value
        self.t_last = t0  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1/200.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0., 1e5, 101)
        self.y = self.bed(self.x)

    def __str__(self):
        return "Minimum Glacier Model for a concave bed."

    def bed(self, x):
        """ Returns the elevation of the bed at point x """
        return self.ba + self.b0 * np.exp(-x/self.xl)

    def mean_bed(self, x=None):
        """ Returns the mean elevation of the bed from 0 to x """
        if x is None:
            x = self.L_last
        if np.isclose(x, 0.):             
            return self.bed(x)
        return self.ba + self.xl * self.b0 / x * (1. - np.exp(-1.*x/self.xl))

    def slope(self, x):
        """ Returns the slope of the bed at point x """
        return self.b0 / self.xl * np.exp(-1.*x / self.xl)

    def mean_slope(self, x=None):
        """ Returns the mean slope of the bed from 0 to x """
        if x is None:
            x = self.L_last
        if np.isclose(x, 0.):             
            return self.slope(x)
        return self.b0 * (1. - np.exp(-1.*x/self.xl)) / x

    def d_slope_d_L(self, L=None):
        """ Returns the change of the mean slope if the value of L changes """
        if L is None:
            L = self.L_last
        if np.isclose(L, 0.):             
            return 0.
        dsdl_1 = -self.b0 * (1. - np.exp(-1.*L/self.xl)) / np.power(L, 2.)
        dsdl_2 = self.b0 * np.exp(-1.*L/self.xl) / self.xl / L
        return dsdl_1 + dsdl_2


class CustomBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a custom bed """

    def __init__(self, x, y, calving=True, L0=1., t0=0.):
        self.alpha = 3.
        self.beta = 0.007
        self.nu = 10.
        self.W = 1. # meters

        self.L_last = L0  # meters # initial value
        self.t_last = t0  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1/200.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.array(x, dtype=np.float)
        self.y = np.array(y, dtype=np.float)
        self.s = self.slope(self.x)
        self.mb = self.mean_bed_init()
        self.ms = self.mean_slope_init()

    def __str__(self):
        return "Minimum Glacier Model for a custom bed."

    def bed(self, x):
        """ Returns the elevation of the bed at point x """
        return np.interp(x, self.x, self.y)

    def mean_bed_init(self):
        """ Returns the mean elevation of the bed from 0 to x for all x in self.x """
        # calculate distance between two points
        delta_x = self.x[1:] - self.x[:-1]
        # calculate y at midpoint
        y_midpoint = (self.y[1:] + self.y[:-1])/2.
        # calculate integrated elevation per delta_x
        total_elevation = delta_x * y_midpoint
        # calculate mean elevation
        means = np.zeros(np.size(self.x), dtype=np.float)
        means[0] = self.bed(self.x[0])
        for i in range(1, np.size(means)):
            means[i] = np.sum(total_elevation[:i]) / (np.max(self.x[:i+1]) - np.min(self.x[:i+1]))
        return means

    def mean_bed(self, x=None):
        """ Returns the mean elevation of the bed from 0 to x """
        if x is None:
            x = self.L_last
        return np.interp(x, self.x, self.mb)

    def slope(self, x):
        """ Returns the slope of the bed at point x """
        return -1. * np.interp(x, self.x, np.gradient(self.y, self.x))

    def mean_slope_init(self):
        """ Returns the mean slope of the bed from 0 to x for all x in self.x """
        # calculate distance between two points
        delta_x = self.x[1:] - self.x[:-1]
        # calculate y at midpoint
        y_midpoint = (self.s[1:] + self.s[:-1])/2.
        # calculate integrated slope per delta_x
        total_slopes = delta_x * y_midpoint
        # calculate mean slope
        means = np.zeros(np.size(self.x), dtype=np.float)
        means[0] = self.slope(self.x[0])
        for i in range(1, np.size(means)):
            means[i] = np.sum(total_slopes[:i]) / (np.max(self.x[:i+1]) - np.min(self.x[:i+1]))
        return means

    def mean_slope(self, x=None):
        """ Returns the mean slope of the bed from 0 to x """
        if x is None:
            x = self.L_last
        return np.interp(x, self.x, self.ms)

    def d_slope_d_L(self, L=None, dL=10.):
        """ Returns the change of the mean slope if the value of L changes """
        ### Method: central finite differences with second-order accuracy
        if L is None:
            L = self.L_last
        if L < dL:
            # fix for left border
            dL = 0.9*L
        [a, b] = [L-dL, L+dL]
        A = self.mean_slope(a)
        B = self.mean_slope(b)
        return (-1.*A/2. + B/2.) / dL



if __name__ == "__main__":

    glacier_l = LinearBedModel()
    glacier_l.integrate(0.1, 400., E=900)
    glacier_l.integrate(0.1, 400., E=800)
    glacier_l.integrate(0.1, 400., E=700)
    glacier_l.plot(E=True, C=True, V=True, show=False, filename=None)

    glacier_c = ConcaveBedModel()
    glacier_c.integrate(0.1, 400., E=1200)
    glacier_c.integrate(0.1, 400., E=1100)
    glacier_c.integrate(0.1, 400., E=1000)
    glacier_c.plot(E=True, C=True, V=True, show=False, filename=None)

    plt.show()
