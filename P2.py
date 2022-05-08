import numpy as np
import header
import matplotlib.pyplot as plt
from cvxpy import Variable, Minimize, Problem, log, power
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)


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
# Lmax = [750, 1500, 2000]
nbs_sols = []

# Tw_plt = np.linspace(header.Tw_min, header.Tw_max)
Tw_plt = np.linspace(100, 1200)

# To plot all Lmax in the same plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for l in Lmax:
    E_worst = 0.0457
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

    print("Lmax = ", str(l))
    print("optimal value p* = ", problem.value)
    print("optimal var: Tw = ", Tw.value)
    print("optimal var: E1 = ", E1.value)
    print("optimal var: L1 = ", L1.value)

    # Add to plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Worst E and L point
    x_values = [E_worst, E1.value]
    y_values = [L_worst, L1.value]
    ax.scatter(E_worst, L_worst, color="green", label = "[Eworst, Lworst] = [" + str(E_worst) + ", " + str(L_worst) + "]")
    plt.text(E_worst - 0.0025, L_worst + 50, "Eworst, Lworst", color="red", fontsize="medium")
    plt.plot(x_values, y_values, linestyle="--")

    # Best E and L points
    E_best = 0.0162
    L_best = 452.0477
    x_values = [E_worst, E_best]
    y_values = [L_worst, L_best]
    ax.scatter(E_best, L_best, color="orange", label="[Ebest, Lbest] = [" + str(E_best) + ", " + str(L_best) + "]")
    plt.text(E_best - 0.0015, L_best - 80, "Ebest, Lbest", fontsize="medium")
    plt.plot(x_values, y_values, linestyle="--")

    # NBS point
    # ax.scatter(E1.value, L1.value, label="Lmax = " + str(l)) # To plot all Lmax in the same plot
    ax.scatter(E1.value, L1.value, label="[E1, L1] = [" + str(E1.value[0]) + ", " + str(L1.value[0]) + "]")
    plt.text(E1.value - 0.001, L1.value + 100, "E1, L1", color="blue", fontsize="medium")

    plt.plot(energy_plt(Tw_plt), delay_plt(Tw_plt))
    plt.xlabel("E(Tw)")
    plt.ylabel("L(Tw)")
    plt.title("NBS for Lmax = " + str(l))
    plt.legend(loc="upper center", fontsize="medium")
    # plt.show()
    label = "img/NBS_Lmax" + str(l) + ".png"
    plt.savefig(label)

# plt.plot(energy_plt(Tw_plt), delay_plt(Tw_plt), color='b')
# plt.xlabel("E(Tw)")
# plt.ylabel("L(Tw)")
# plt.legend(loc="upper right")
# plt.title("NBS for different Lmax")
# plt.show()
# plt.savefig("img/NBS")
