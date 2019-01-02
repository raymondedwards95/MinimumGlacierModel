import matplotlib.pyplot as plt
import numpy as np

__all__ = ["BucketTributary"]

class BucketTributary():
    """ Class for bucket tributary glacier """
    def __init__(self, L=1e3, w0=5, w1=15, h0=0, h1=1e2):
        self.l = L
        self.w0 = w0
        self.w1 = w1
        self.h0 = np.min([h0, h1])
        self.h1 = np.max([h0, h1])

        self.q = (w1 - w0) / L
        self.s = (h1 - h0) / L

    def __str__(self):
        return "Tributary glacier represented by a basin with a tilted trapezoidal shape."

    def mass_budget(self, E, beta=None):
        """ Calculates the mass budget from equation 1.6.1 
        
        Bm = beta * [w0 * (h0-E) * L 
            + 1/2 * {s*w0 + (b0-E)*q} * L^2 
            + 1/3  * s * q * L^3]
        """
        if beta is None:
            beta = 0.007
        A_ = self.w0 * (self.h0 - E) * self.l
        B_ = 1/2 * (self.s * self.w0 + (self.h0 - E) * self.q) * np.power(self.l, 2.)
        C_ = 1/3 * self.s * self.q * np.power(self.l, 3.)
        return beta * (A_ + B_ + C_)

    def net_effect(self, E, beta=None):
        """ Returns mass_budget if it is positive """
        if beta is None:
            beta = 0.007
        return np.max([0, self.mass_budget(E, beta)])


if __name__ == "__main__":
    bucket = BucketTributary()
    E = np.linspace(0, 2e2)
    b = [bucket.mass_budget(Ei) for Ei in E]
    n = [bucket.net_effect(Ei) for Ei in E]

    plt.figure()
    plt.plot(E, b)
    plt.plot(E, n)
    plt.ylabel("B")
    plt.xlabel("E")
    plt.grid()
    plt.tight_layout()
    plt.show()
