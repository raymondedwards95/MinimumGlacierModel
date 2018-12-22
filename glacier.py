import numpy as np

__all__ = ["LinearBedModel", "ConcaveBedModel", "CustomBedModel"]


class MinimumGlacierModel():
    """ Base class for minimum glacier models """
    def __init__(self, calving=True):
        # should not use this object to create a glacier
        self.alpha = 3.
        self.beta = 0.007
        self.nu = 10.
        self.W = 1. # meters

        self.L_last = 1.  # meters # initial value
        self.t_last = 0.  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1/200.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0, 1e5)
        self.y = -self.x + np.max(self.x)/2


    def __str__(self):
        return "Base class for a minimum Glacier Model."


    def bed(self, x):
        return 0


    def mean_bed(self, x=None):
        if x is None:
            x = self.L_last
        return 0


    def slope(self, x):
        return 0


    def mean_slope(self, x=None):
        if x is None:
            x = self.L_last
        return 0


    def d_slope_d_L(self, L=None):
        if L is None:
            L = self.L_last
        return 0


    def mean_ice_thickness(self):
        """ Calculates the mean ice thickness Hm (eq 1.2.2) """
        s_mean = self.mean_slope(x=self.L_last)
        return self.alpha / (1+self.nu*s_mean) * np.power(self.L_last, 1/2.)


    def water_depth(self):
        """ Determine the water depth at the end of the glacier by comparing the height of the bed with zero """
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
        s_mean = self.mean_slope(x=self.L_last)
        ds = self.d_slope_d_L(self.L_last)
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
    def __init__(self, b0=3900, s=0.1, calving=True):
        self.b0 = b0
        self.s = s

        self.alpha = 3.
        self.beta = 0.007
        self.nu = 10.
        self.W = 1. # meters

        self.L_last = 1.  # meters # initial value
        self.t_last = 0.  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1/200.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0, 1e5, 101)
        self.y = self.bed(self.x)


    def __str__(self):
        return "Minimum Glacier Model for a linear bed."


    def bed(self, x):
        return self.b0 - self.s * x


    def mean_bed(self, x=None):
        if x is None:
            x = self.L_last
        return self.b0 - self.s * x / 2


    def slope(self, x):
        return self.s


    def mean_slope(self, x=None):
        if x is None:
            x = self.L_last
        return self.s


    def d_slope_d_L(self, L=None):
        if L is None:
            L = self.L_last
        return 0.


class ConcaveBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a concave bed """

    def __init__(self, b0=3900., ba=-100., xl=7000., calving=True):
        self.b0 = b0
        self.ba = ba
        self.xl = xl

        self.alpha = 3.
        self.beta = 0.007
        self.nu = 10.
        self.W = 1. # meters

        self.L_last = 1.  # meters # initial value
        self.t_last = 0.  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = 1.
        self.kappa = 1/200.

        self.E = 2900.
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0, 1e5, 101)
        self.y = self.bed(self.x)


    def __str__(self):
        return "Minimum Glacier Model for a concave bed."


    def bed(self, x):
        return self.ba + self.b0 * np.exp(-x/self.xl)


    def mean_bed(self, x=None):
        if x is None:
            x = self.L_last
        if np.isclose(x, 0):             
            return 0
        return self.ba + self.xl * self.b0 / x * (1 - np.exp(-x/self.xl))

    
    def slope(self, x):
        return self.b0 / self.xl * np.exp(-x / self.xl)


    def mean_slope(self, x=None):
        if x is None:
            x = self.L_last
        if np.isclose(x, 0):             
            return 0
        return self.b0 * (1 - np.exp(-x/self.xl)) / x


    def d_slope_d_L(self, L=None):
        if L is None:
            L = self.L_last
        if np.isclose(L, 0):             
            return 0
        dsdl_1 = -self.b0 * (1 - np.exp(-L/self.xl)) / np.power(L, 2.)
        dsdl_2 = self.b0 * np.exp(-L/self.xl) / self.xl / L
        return dsdl_1 + dsdl_2


class CustomBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a custom bed """

    def __init__(self, x, y, calving=True):
        self.alpha = 3.
        self.beta = 0.007
        self.nu = 10.
        self.W = 1. # meters

        self.L_last = 1.  # meters # initial value
        self.t_last = 0.  # years

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


    def __str__(self):
        return "Minimum Glacier Model for a custom bed."


    def bed(self, x):
        return np.interp(x, self.x, self.y)


    def mean_bed(self, x=None):
        ### NOTE: this may be not the best way to calculate the mean bed elevation
        if x is None:
            x = self.L_last
        # find all indices where x_bed < argument x
        indices = np.where(self.x < x)[0]
        # create a list of x and y for x <= argument x
        x_list = np.append(self.x[indices], x)
        y_list = np.append(self.y[indices], self.bed(x))
        # calculate distance between two points
        delta_x = x_list[1:] - x_list[:-1]
        # calculate y at midpoint
        y_midpoint = (y_list[1:] + y_list[:-1])/2.
        # calculate integrated elevation per delta_x
        elevation = delta_x * y_midpoint
        # return mean elevation
        return np.sum(elevation) / (np.max(x_list) - np.min(x_list))


    def slope(self, x):
        return -1 * np.interp(x, self.x, np.gradient(self.y))


    def mean_slope(self, x=None):
        ### NOTE: this is possibly not the best way to calculate the mean bed slope
        if x is None:
            x = self.L_last
        # find all indices where x_bed < argument x
        indices = np.where(self.x < x)[0]
        # create a list of x and y for x <= argument x
        x_list = np.append(self.x[indices], x)
        y_list = np.array([self.slope(xi) for xi in x_list])
        # calculate distance between two points
        delta_x = x_list[1:] - x_list[:-1]
        # calculate y at midpoint
        y_midpoint = (y_list[1:] + y_list[:-1])/2.
        # calculate integrated slope per delta_x
        slopes = delta_x * y_midpoint
        # return mean slope
        return np.sum(slopes) / (np.max(x_list) - np.min(x_list))


    def d_slope_d_L(self, L=None):
        if L is None:
            L = self.L_last
        # TODO
        return 0



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    glacier_l = LinearBedModel()
    glacier_l.integrate(0.1, 500.)

    plt.figure()
    plt.title("Linear bed glacier length")
    plt.plot(glacier_l.t, glacier_l.L)
    plt.grid()
    plt.xlabel("time [years]")
    plt.ylabel("length [m]")
    plt.tight_layout()
    plt.show()
