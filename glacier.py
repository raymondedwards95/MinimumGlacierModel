import matplotlib.pyplot as plt
import numpy as np

from tributary import BucketTributary

__all__ = ["LinearBedModel", "ConcaveBedModel", "CustomBedModel"]


class MinimumGlacierModel():
    """ Base class for minimum glacier models 
    
    Glacier options:
        name: name of object
        L0: initial length
        t0: initial time
        E: initial equilibrium line altitude

    Bed parameters:

    Ice parameters:
        alpha:
        beta:
        nu:

    Calving parameters:
        calving: include calving?
        c: calving parameter ~1
        kappa: fraction of mean ice thickness that is being calved
    """

    def __init__(self, calving: bool=True, L0: float=1., t0: float=0., E: float=2900.,
                 alpha: float=3., beta: float=0.007, nu: float=10., 
                 c: float=1., kappa: float=1/2.,
                 w: float=1., name: str="Glacier"):
        # should not use this object to create a glacier
        self.name = name

        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.w = w # used for mass balance of tributaries

        self.L_last = L0  # meters # initial value
        self.t_last = t0  # years

        self.L = np.array([self.L_last], dtype=np.float)
        self.t = np.array([self.t_last], dtype=np.float)

        self.calving = calving
        self.rho_water = 1000.
        self.rho_ice = 917.
        self.c = c
        self.kappa = kappa

        self.E = E
        self.E_data = np.array([self.E], dtype=np.float)

        self.x = np.linspace(0., 1e5)
        self.y = -1.*self.x + np.max(self.x)/2.

        self.tributary = []
        self.tributary_number = 0

    def __str__(self):
        return "Base class for a minimum Glacier Model: {}".format(self.name)

    def bed(self, x):
        """ Returns the elevation of the bed at point x 
        
        b(x) = 0
        """
        return 0.

    def mean_bed(self, x=None):
        """ Returns the mean elevation of the bed from 0 to x 
        
        [b](x) = 0
        """
        if x is None:
            x = self.L_last
        return 0.

    def slope(self, x):
        """ Returns the slope of the bed at point x 
        
        db/dx = 0
        """
        return 0.

    def mean_slope(self, x=None):
        """ Returns the mean slope of the bed from 0 to x 
        
        [db/dx] = 0
        """
        if x is None:
            x = self.L_last
        return 0.

    def d_slope_d_L(self, L=None):
        """ Returns the change of the mean slope if the value of L changes 
        
        d[db/dx]/dL = 0
        """
        if L is None:
            L = self.L_last
        return 0.

    def add_bucket_tributary(self, L=1e3, w0=5, w1=15, h0=0, h1=1e2, show=False):
        """ Adds a bucket tributary glacier to the main glacier 
        
        parameters:
        L - Length
        w0 - width at bottom
        w1 - width at top
        h0 - elevation of bottom
        h1 - elevation of top
        """
        self.tributary.append(BucketTributary(L=L, w0=w0, w1=w1, h0=h0, h1=h1, number=self.tributary_number))
        self.tributary_number += 1
        if show: 
            print("Created tributary at y={:0.2f}".format(h0))

    def mean_ice_thickness(self, L=None):
        """ Calculates the mean ice thickness Hm with equation 1.2.2 
        
        Hm(L) = a/(1+nu*[s]) * L^(1/2)
        """
        if L is None:
            L = self.L_last
        s_mean = self.mean_slope(L)
        return self.alpha / (1.+self.nu*s_mean) * np.power(L, 1./2.)

    def water_depth(self, x=None):
        """ Determines the water depth at x by checking if the elevation of the bed is negative 
        
        d(x) = min(0, b(x))
        """
        if x is None:
            x = self.L_last
        return -1. * np.min([0., self.bed(x)])

    def calving_flux(self, L=None):
        """ Calculates the calving flux using equations 1.5.1, 1.5.2 and 1.5.3 
        
        Hf = max(K*Hm, d*rw/ri);
        F = - c*d*Hf*w
        """
        w = 1. # note that most equations use w = 1
        if L is None:
            L = self.L_last
        d = self.water_depth(L)
        H_mean = self.mean_ice_thickness(L)
        Hf = np.max([self.kappa * H_mean, self.rho_water / self.rho_ice * d])
        F = -self.c * d * Hf * w
        return F

    def mass_balance(self, L=None):
        """ Returns the mass balance from equation 1.3.0 
        
        Bs = beta * ([b] + Hm - E) * w * L
        """
        w = 1. # note that most equations use w = 1
        if L is None:
            L = self.L_last
        b_mean = self.mean_bed(L)
        H_mean = self.mean_ice_thickness(L)
        return self.beta * (b_mean + H_mean - self.E) * w * L

    def mass_budget(self, L=None, E=None):
        """ Returns the mass budget from equation 1.6.2 
        
        B_tot = dV/dt = F + Bm + sum(Bi)
        """
        if L is None:
            L = self.L_last
        if E is None:
            E = self.E
        if self.calving:
            F = self.calving_flux(L)
        else:
            F = 0.
        low = self.bed(L)
        high = self.bed(0)
        Bm = self.mass_balance(L)
        budget = F + Bm
        for subglacier in self.tributary:
            if high >= subglacier.h0 >= low:
                budget += subglacier.net_effect(E, beta=self.beta) / (self.w * self.mean_ice_thickness())
        return budget

    def change_L(self, L=None):
        """ Determines dL/dt from equation 1.2.5 
        
        dL/dt = {3a/2(1+nu*[s])*L^(1/2) - a*nu*(1+nu*[s])^(-2)*L^(3/2)*d[s]/dL}^(-1) * B_tot
        """
        if L is None:
            L = self.L_last
        s_mean = self.mean_slope(L)
        ds = self.d_slope_d_L(L)
        dldt_1 = 3. * self.alpha / (2*(1+self.nu*s_mean)) * np.power(L, 1./2.)
        dldt_2 = - self.alpha * self.nu / np.power(1+self.nu*s_mean, 2.) * np.power(L, 3./2.) * ds
        dldt_3 = self.mass_budget(L)
        return np.power(dldt_1 + dldt_2, -1.) * dldt_3

    def integrate(self, dt, time, E=None, E_new=None, show=False):
        """ Function to integrate in time using a forward Euler scheme 
        
        L(t+dt) = L(t) + dt * dL/dt
        """
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
        if show:
            print("Creating plot:")
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
        if show:
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
            if show:
                print("E: {}/{}".format(c, p))
            sub_E = plt.subplot(p, 1, c)
            subplots.append(sub_E)
            sub_E.plot(self.t, self.E_data, "--", label="E")
            sub_E.plot(self.t, np.max(self.y)*np.ones(np.size(self.t)), label="Top of mountain")
            sub_E.plot(self.t, self.bed(self.L), label="End of glacier")
            sub_E.plot(self.t, [self.mean_bed(Li) for Li in self.L], label="Mean bed height")
            if c == p:
                sub_E.set_xlabel("Time [y]")
            sub_E.set_ylabel("Height [m]")
            sub_E.grid()
            sub_E.legend()

        if C is True:
            w = 1. # note that most equations use w = 1
            c += 1
            if show:
                print("C: {}/{}".format(c, p))
            sub_C = plt.subplot(p, 1, c)
            subplots.append(sub_C)
            sub_C.plot(self.t, [self.calving_flux(Li) for Li in self.L], label="Total", c="black")
            sub_C.plot(self.t, [-self.c * self.water_depth(Li) * w * self.kappa * self.mean_ice_thickness(Li) for Li in self.L], "--", label="Front")
            sub_C.plot(self.t, [-self.c * self.water_depth(Li)**2. * w * self.rho_water / self.rho_ice for Li in self.L], "--", label="Float")
            if c == p:
                sub_C.set_xlabel("Time [y]")
            sub_C.set_ylabel("Calving Flux []")
            sub_C.grid()
            sub_C.legend()

        if V is True:
            c += 1
            if show:
                print("V: {}/{}".format(c, p))
            sub_V = plt.subplot(p, 1, c)
            subplots.append(sub_V)
            sub_V.plot(self.t, [self.mean_ice_thickness(Li) * Li for Li in self.L])
            if c == p:
                sub_V.set_xlabel("Time [y]")
            sub_V.set_ylabel("Ice volume [m$^3$]")
            sub_V.grid()


        if filename is not None:
            plt.savefig(filename+".png")

        if show:
            plt.show()
        else:
            # plt.clf()
            # plt.close()
            pass

    def parameters(self, header=False, show=False):
        """ Returns the parameters of ice and calving 
        
        alpha, beta, nu, c, kappa, w
        """
        if show:
            if header:
                print("|    alpha |     beta |       nu |        c |    kappa |        w |")
            print("| {:8.2f} | {:8.4f} | {:8.2f} | {:8.2f} | {:8.4f} | {:8.2f}".format(self.alpha, self.beta, self.nu, self.c, self.kappa, self.w), self.name)
        return (self.alpha, self.beta, self.nu, self.c, self.kappa, self.w)

    def data(self):
        """ Returns data:

        time, length, equilibrium line altitude 
        """
        return self.t, self.L, self.E_data


class LinearBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a linear bed 
    
    Glacier options:
        name: name of object
        L0: initial length
        t0: initial time
        E: initial equilibrium line altitude

    Bed parameters:
        b0: elevation of the top of the bed
        s: slope of the bed

    Ice parameters:
        alpha:
        beta:
        nu:

    Calving parameters:
        calving: include calving?
        c: calving parameter ~1
        kappa: fraction of mean ice thickness that is being calved
    """

    def __init__(self, b0: float=3900., s: float=0.1, 
                 calving: bool=True, L0: float=1., t0: float=0., E: float=2900.,
                 alpha: float=3., beta: float=0.007, nu: float=10.,
                 c: float=1., kappa: float=1/2.,
                 w: float=1., name: str="Linear Bed glacier"):
        # super sets parameters as in parent class, MinimumGlacierModel
        super().__init__(calving=calving, L0=L0, t0=t0, E=E,
                         alpha=alpha, beta=beta, nu=nu,
                         c=c, kappa=kappa,
                         w=w, name=name)

        self.b0 = b0
        self.s = s

        self.x = np.linspace(0., 1e5, 101)
        self.y = self.bed(self.x)

    def __str__(self):
        return "Minimum Glacier Model for a linear bed: {}".format(self.name)

    def bed(self, x):
        """ Returns the elevation of the bed at point x 
        
        b(x) = b0 - s*x
        """
        return self.b0 - self.s * x

    def mean_bed(self, x=None):
        """ Returns the mean elevation of the bed from 0 to x 
        
        [b](x) = b0 - s*x/2
        """
        if x is None:
            x = self.L_last
        return self.b0 - self.s * x / 2.

    def slope(self, x):
        """ Returns the slope of the bed at point x 
        
        s(x) = db/dx = s
        """
        return self.s

    def mean_slope(self, x=None):
        """ Returns the mean slope of the bed from 0 to x 
        
        [s](x) = [db/dx] = s
        """
        if x is None:
            x = self.L_last
        return self.s

    def d_slope_d_L(self, L=None):
        """ Returns the change of the mean slope if the value of L changes 
        
        d[s]/dL = d[db/dx]/dL = 0        
        """
        if L is None:
            L = self.L_last
        return 0.


class ConcaveBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a concave bed 
    
    Glacier options:
        name: name of object
        L0: initial length
        t0: initial time
        E: initial equilibrium line altitude

    Bed parameters:
        b0: elevation of the top of the bed
        ba: elevation of the end of the bed
        xl: length-scale of the bed

    Ice parameters:
        alpha:
        beta:
        nu:

    Calving parameters:
        calving: include calving?
        c: calving parameter ~1
        kappa: fraction of mean ice thickness that is being calved
    """

    def __init__(self, b0: float=3900., ba: float=-100., xl: float=7000., 
                 calving: bool=True, L0: float=1., t0: float=0., E: float=2900.,
                 alpha: float=3., beta: float=0.007, nu: float=10., 
                 c: float=1., kappa: float=1/2.,
                 w: float=1., name: str="Concave bed glacier"):
        # super sets parameters as in parent class, MinimumGlacierModel
        super().__init__(calving=calving, L0=L0, t0=t0, E=E,
                         alpha=alpha, beta=beta, nu=nu,
                         c=c, kappa=kappa,
                         w=w, name=name)
        
        self.b0 = b0
        self.ba = ba
        self.xl = xl

        self.x = np.linspace(0., 1e5, 101)
        self.y = self.bed(self.x)

    def __str__(self):
        return "Minimum Glacier Model for a concave bed: {}".format(self.name)

    def bed(self, x):
        """ Returns the elevation of the bed at point x 
        
        b(x) = ba + b0*exp(-x/xl)
        """
        return self.ba + self.b0 * np.exp(-x/self.xl)

    def mean_bed(self, x=None):
        """ Returns the mean elevation of the bed from 0 to x 
        
        [b](x) = ba + xl*b0/L*(1 - exp(-L/xl))
        """
        if x is None:
            x = self.L_last
        if np.isclose(x, 0.):             
            return self.bed(x)
        return self.ba + self.xl * self.b0 / x * (1. - np.exp(-1.*x/self.xl))

    def slope(self, x):
        """ Returns the slope of the bed at point x 
        
        s(x) = db/dx = -b0/xl*exp(-x/xl)
        """
        return self.b0 / self.xl * np.exp(-1.*x / self.xl)

    def mean_slope(self, x=None):
        """ Returns the mean slope of the bed from 0 to x 
        
        [s](x) = [db/dx] = b0/L*(1-exp(-L/xl))
        """
        if x is None:
            x = self.L_last
        if np.isclose(x, 0.):             
            return self.slope(x)
        return self.b0 * (1. - np.exp(-1.*x/self.xl)) / x

    def d_slope_d_L(self, L=None):
        """ Returns the change of the mean slope if the value of L changes 
        
        d[s]/dL = d[db/dx]/dL = -b0/L^2*(1-exp(-L/xl)) + b0/L/xl*exp(-L/xl)        
        """
        if L is None:
            L = self.L_last
        if np.isclose(L, 0.):             
            return 0.
        dsdl_1 = -self.b0 * (1. - np.exp(-1.*L/self.xl)) / np.power(L, 2.)
        dsdl_2 = self.b0 * np.exp(-1.*L/self.xl) / self.xl / L
        return dsdl_1 + dsdl_2


class CustomBedModel(MinimumGlacierModel):
    """ Minimum glacier model for a custom bed 
    
    Glacier options:
        name: name of object
        L0: initial length
        t0: initial time
        E: initial equilibrium line altitude

    Bed parameters:
        x: x-coordinates
        y: y-coordinates

    Ice parameters:
        alpha:
        beta:
        nu:

    Calving parameters:
        calving: include calving?
        c: calving parameter ~1
        kappa: fraction of mean ice thickness that is being calved
        
        """

    def __init__(self, x, y, 
                 calving: bool=True, L0: float=1., t0: float=0., E: float=2900.,
                 alpha: float=3., beta: float=0.007, nu: float=10., 
                 c: float=1., kappa: float=1/2.,
                 w: float=1., name: str="Custom bed glacier"):
        # super sets parameters as in parent class, MinimumGlacierModel
        super().__init__(calving=calving, L0=L0, t0=t0, E=E,
                         alpha=alpha, beta=beta, nu=nu,
                         c=c, kappa=kappa,
                         w=w, name=name)

        self.x = np.array(x, dtype=np.float)
        self.y = np.array(y, dtype=np.float)
        self.s = self.slope(self.x)
        self.mb = self.mean_bed_init()
        self.ms = self.mean_slope_init()

    def __str__(self):
        return "Minimum Glacier Model for a custom bed: {}".format(self.name)

    def bed(self, x):
        """ Returns the elevation of the bed at point x 
        
        b(x)
        """
        return np.interp(x, self.x, self.y)

    def mean_bed_init(self):
        """ Returns the mean elevation of the bed from 0 to x for all x in self.x 
        
        [b]
        """
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
        """ Returns the mean elevation of the bed from 0 to x 
        
        [b](x)
        """
        if x is None:
            x = self.L_last
        return np.interp(x, self.x, self.mb)

    def slope(self, x):
        """ Returns the slope of the bed at point x 
        
        s(x)
        """
        return -1. * np.interp(x, self.x, np.gradient(self.y, self.x))

    def mean_slope_init(self):
        """ Returns the mean slope of the bed from 0 to x for all x in self.x 
        
        [s]
        """
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
        """ Returns the mean slope of the bed from 0 to x 
        
        [s](x)
        """
        if x is None:
            x = self.L_last
        return np.interp(x, self.x, self.ms)

    def d_slope_d_L(self, L=None, dL=10.):
        """ Calculates the change of the mean slope if the value of L changes, 
        with a central finite differences method
        
        d[s]/dL = d[db/dx]/dL
        """
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

    glacier_l = LinearBedModel(name="linear_test")
    print(glacier_l)
    glacier_l.parameters(header=True, show=True)
    glacier_l.integrate(0.5, 400., E=900)
    glacier_l.integrate(0.5, 400., E=800)
    glacier_l.integrate(0.5, 400., E=700)
    glacier_l.plot(E=True, C=True, V=True, show=False)

    glacier_c = ConcaveBedModel(name="concave_test")
    print(glacier_c)
    glacier_c.parameters(show=True)
    glacier_c.integrate(0.5, 400., E=1200)
    glacier_c.integrate(0.5, 400., E=1100)
    glacier_c.integrate(0.5, 400., E=1000)
    glacier_c.plot(E=True, C=True, V=True, show=False)

    plt.show()
