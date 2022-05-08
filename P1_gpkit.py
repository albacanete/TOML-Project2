import gpkit as gp
import numpy as np
import header
import matplotlib.pyplot as plt
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
        return Fs
    else:
        return Fin(d) + Fs


def Fin(d):
    if d == 0:
        return Fs * header.D ** 2 * header.C
    else:
        return Id(d) * Fout(d + 1)


def Fb(d):
    return (header.C - abs(Id(d))) * Fout(d)


def energy(Tw):
    return alpha1/Tw + alpha2*Tw + alpha3


def delay(Tw):
    return beta1 * Tw + beta2


def alphas(d):
    alpha1 = header.Tcs + header.Tal + 3 / 2 * header.Tps * (
            (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata) \
             * Fb(d)
    alpha2 = Fout(d) / 2
    alpha3 = ((header.Tps + header.Tal) / 2 + header.Tcs + header.Tal + header.Tack + header.Tdata) * Fout(d) \
             + (3 / 2 * header.Tps + header.Tack + header.Tdata) * Fin(d) + 3 / 4 * header.Tps * Fb(d)
    return alpha1, alpha2, alpha3


def betas(d):
    beta1 = sum([1 / 2] * d)
    beta2 = sum([header.Tcw / 2 + header.Tdata] * d)
    return beta1, beta2


# P1: ENERGY MINIMIZATION
Lmax = [500, 1200, 2000, 3500, 5000]   # No solution for Lmax < 500
energy_sols = []
minutes = [5, 10, 15, 20, 25, 30]      # No solution for 1 minute
Tw_plt = np.linspace(header.Tw_min, header.Tw_max)

for l in Lmax:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for m in minutes:
        # Calculate Fs
        Fs = 1.0 / (60 * m * 1000)

        # Create optimization variable
        Tw = gp.Variable("Tw")

        # Calculate alphas
        alpha1, alpha2, alpha3 = alphas(1)

        # Calculate betas
        beta1, beta2 = betas(header.D)

        # Calculate Etxn and Ttx
        # Ttx = np.ceil((Tw / (header.Tps + header.Tal))) * (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata
        Ttx = (Tw / (header.Tps + header.Tal)) * (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata
        Etxn = (header.Tcs + header.Tal + Ttx) * Fout(1)

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

        # Plot
        lab = "Fs = " + str(m)
        plt.plot(Tw_plt, energy(Tw_plt), label=lab)
        ax.scatter(solution['variables'][Tw], solution['cost'], color="red")

    plt.xlabel('Tw')
    plt.ylabel('E(Tw)')
    plt.title("Energy minimization for L_max="+str(l))
    plt.legend(loc='upper right')
    #plt.show()
    name = "Energy_Lmax_" + str(l)
    plt.savefig("img/"+name)

# P2: DELAY MINIMIZATION
Ebudget = np.arange(1, 5.1, 0.5)
delay_sols = []

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for e in Ebudget:
    # Calculate Fs
    Fs = 1.0 / (60 * 5 * 1000)

    # Create optimization variable
    Tw = gp.Variable("Tw")

    # Calculate alphas
    alpha1, alpha2, alpha3 = alphas(1)

    # Calculate betas
    beta1, beta2 = betas(header.D)

    # Calculate Etxn and Ttx
    # Ttx = np.ceil((Tw / (header.Tps + header.Tal))) * (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata
    Ttx = (Tw / (header.Tps + header.Tal)) * (header.Tps + header.Tal) / 2 + header.Tack + header.Tdata
    Etxn = (header.Tcs + header.Tal + Ttx) * Fout(1)

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
    # print(solution.table())
    delay_sols.append(solution["cost"])

    # Plot
    plt.plot(Tw_plt, delay(Tw_plt))
    ax.scatter(solution['variables'][Tw], solution['cost'], color="red")

plt.xlabel('Tw')
plt.ylabel('L(Tw)')
plt.title("Delay minimization for Ebudget")
# plt.show()
plt.savefig("img/Delay_Ebudget")
