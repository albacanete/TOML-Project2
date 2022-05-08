import numpy as np
import header
import matplotlib.pyplot as plt
from cvxpy import Variable, Minimize, Problem, log, SolverError, SCS, power

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


def energy(Tw):
    return alpha1 * power(Tw, -1) + alpha2 * Tw + alpha3


def delay(Tw):
    return beta1 * Tw + beta2


def energy_plt(Tw):
    return alpha1 / Tw + alpha2 * Tw + alpha3


def delay_plt(Tw):
    return beta1 * Tw + beta2


Lmax = np.linspace(500, 5000, 7)   # No solution for Lmax < 500
nbs_sols = []

# Plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
Tw_plt = np.linspace(header.Tw_min, header.Tw_max)

for l in Lmax:
    E_worst = 0.05
    L_worst = l

    # Create optimization variable
    Tw = Variable(1, name='Tw')
    E1 = Variable(1, name='E1')
    L1 = Variable(1, name='L1')

    # Calculate Etxn and Ttx
    Ttx = (Tw / (header.Tps + header.Tal)) * (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata
    Etxn = (header.Tcs + header.Tal + Ttx) * Fout(1)

    # Calculate alphas and betas
    d = 1
    alpha1 = header.Tcs + header.Tal + 3 / 2 * header.Tps * (
            (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata) \
             * Fb(d)
    alpha2 = Fout(d) / 2
    alpha3 = ((header.Tps + header.Tal) / 2 + header.Tcs + header.Tal + header.Tack + header.Tdata) * Fout(d) \
             + (3 / 2 * header.Tps + header.Tack + header.Tdata) * Fin(d) + 3 / 4 * header.Tps * Fb(d)

    d = header.D
    beta1 = sum([1 / 2] * d)
    beta2 = sum([header.Tcw / 2 + header.Tdata] * d)

    # Constraint functions
    f1 = E_worst
    f2 = E1
    f3 = L_worst
    f4 = L1
    f5 = Tw
    f6 = abs(Id(0)) * Etxn
    constraints = [f1 >= energy(Tw),
                   f2 >= energy(Tw),
                   f3 >= delay(Tw),
                   f4 >= delay(Tw),
                   f5 >= header.Tw_min,
                   f6 <= 1/4]

    # Form objective
    f0 = -log(E_worst - E1) - log(L_worst - L1)
    obj = Minimize(f0)

    # Form and solve problem.
    problem = Problem(obj, constraints)
    solution = problem.solve()

    print("optimal value p* = ", problem.value)
    print("optimal var: Tw = ", Tw.value)
    print("optimal var: E1 = ", E1.value)
    print("optimal var: L1 = ", L1.value)

    # Add to plot
    ax.scatter(E1.value, L1.value, label='Lmax=' + str(l))

plt.plot(energy_plt(Tw_plt), delay_plt(Tw_plt), color='b')
plt.xlabel("E(Tw)")
plt.ylabel("L(Tw)")
plt.legend(loc="upper right")
# plt.show()
plt.savefig("img/NBS")
