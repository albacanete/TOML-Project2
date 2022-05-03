import gpkit as gp
import numpy as np
import header
import matplotlib.pyplot as plt

def Nd(d):
    if d == 0:
        return 1
    else:
        return (2 * d - 1) * header.C


def Id(d):
    if d == header.D:
        return 0
    elif d == 0:
        return header.C
    else:
        return Nd(d + 1) / Nd(d)


def Fout(d):
    if d == header.D:
        return header.Fs
    else:
        return Fin(d) + header.Fs


def Fin(d):
    if d == 0:
        return header.Fs * header.D ** 2 * header.C
    else:
        return Id(d) * Fout(d + 1)


def Fb(d):
    return (header.C - abs(Id(d))) * Fout(d)

# COMMON VARIABLES
# Calculate alphas
d = 1
alpha1 = header.Tcs + header.Tal + 3 / 2 * header.Tps * ((header.Tps + header.Tal) / 2 + header.Tack + header.Tdata) \
         * Fb(d)
alpha2 = Fout(d) / 2
alpha3 = ((header.Tps + header.Tal) / 2 + header.Tcs + header.Tal + header.Tack + header.Tdata) * Fout(d) \
         + (3 / 2 * header.Tps + header.Tack + header.Tdata) * Fin(d) + 3 / 4 * header.Tps * Fb(d)

# Calculate betas
d = header.D
beta1 = sum([1 / 2] * d)
beta2 = sum([header.Tcw / 2 + header.Tdata] * d)


# Create optimization variable
Tw = gp.Variable("Tw")

# Calculate Etxn and Ttx
# Ttx = np.ceil((Tw / (header.Tps + header.Tal))) * (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata
Ttx = (Tw / (header.Tps + header.Tal)) * (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata
Etxn = (header.Tcs + header.Tal + Ttx) * Fout(1)

# P1: ENERGY MINIMIZATION
Lmax = np.arange(100, 5001, 100)
energy_sols = []

for l in Lmax:
    # Constraint functions
    f1 = beta1 * Tw + beta2
    f2 = Tw
    f3 = abs(Id(0)) * Etxn
    constraints = [f1 <= l, f2 >= header.Tw_min, f3 <= 1 / 4]

    # Form objective
    f0 = alpha1 / Tw + alpha2 * Tw + alpha3

    # Form and solve problem.
    model = gp.Model(f0, constraints)
    solution = model.solve()
    energy_sols.append(solution["cost"])

print(energy_sols)
plt.plot(Lmax, energy_sols)
plt.xlabel('Lmax')
plt.ylabel('Minimization')
plt.title("Energy minimization")
plt.show()

# P2: DELAY MINIMIZATION
Ebudget = np.arange(0.5, 5.1, 0.5)
delay_sols = []

for e in Ebudget:
    # Constraint functions
    f1 = alpha1 / Tw + alpha2 * Tw + alpha3
    f2 = Tw
    f3 = abs(Id(0)) * Etxn
    constraints = [f1 <= e, f2 >= header.Tw_min, f3 <= 1 / 4]

    # Form objective
    f0 = beta1 * Tw + beta2

    # Form and solve problem.
    model = gp.Model(f0, constraints)
    solution = model.solve()
    delay_sols.append(solution["cost"])

plt.plot(Ebudget, delay_sols)
plt.xlabel('Ebudget')
plt.ylabel('Minimization')
plt.title("Delay minimization")
plt.show()