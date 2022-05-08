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


# E = alpha1/Tw + alpha2*Tw + alpha3
# Fout = Fs*(D^2-d^2+2d-1)/(2d-1) when d = 1
# Fin = Fs*(D^2-d^2)/(2d-1) when d = 1
# Fb = (C-Id)*Fout = (C-(2d+1)/(2d-1))*Fout when d = 1
def energy(Tw):
    d = 1
    alpha1 = header.Tcs + header.Tal + 3/2 * header.Tps * ((header.Tps + header.Tal)/2 + header.Tack + header.Tdata) \
             * Fb(d)
    alpha2 = Fout(d)/2
    alpha3 = ((header.Tps + header.Tal)/2 + header.Tcs + header.Tal + header.Tack + header.Tdata) * Fout(d) \
             + (3/2 * header.Tps + header.Tack + header.Tdata) * Fin(d) + 3/4 * header.Tps * Fb(d)
    return alpha1/Tw + alpha2*Tw + alpha3


# L = beta1*Tw + beta2
# Fout = Fs when d = D
# Fin = Fs*(D^2-d^2)/(2d-1) when d = D
# Fb = (C-Id)*Fout = C*Fout when d = D
def delay(Tw):
    d = header.D
    beta1 = sum([1/2] * d)
    beta2 = sum([header.Tcw/2 + header.Tdata] * d)
    return beta1 * Tw + beta2


# Energy plot
minutes = [1, 5, 10, 15, 20, 25, 30]
plt.figure()
for m in minutes:
    Fs = 1.0 / (60 * m * 1000)
    Tw = np.linspace(header.Tw_min, header.Tw_max)
    lab = "Fs = " + str(m)
    plt.plot(Tw, energy(Tw), label=lab)
    plt.xlabel("Tw")
    plt.ylabel("Energy")
    plt.title("Energy function for Fs = " + str(m))
    #plt.legend()
    # plt.show()
    plt.savefig("img/Energy_Fs_" + str(m))
    plt.clf()

# Delay plot
plt.figure()
Tw = np.linspace(header.Tw_min, header.Tw_max)
plt.plot(Tw, delay(Tw))
plt.xlabel("Tw")
plt.ylabel("Delay")
plt.title("Delay function")
# plt.show()
plt.savefig("img/Delay")

# Energy-delay curve plot
minutes = [1, 5, 10, 15, 20, 25, 30]
Tw = np.linspace(10, header.Tw_max)
plt.figure()
for m in minutes:
    Fs = 1.0 / (60 * m * 1000)
    lab = "Fs = " + str(m)
    plt.plot(energy(Tw), delay(Tw), label=lab)
    plt.xlabel("Energy")
    plt.ylabel("Delay")
    plt.title("Energy-delay curve for different Fs")
plt.legend()
# plt.show()
name = "Energy_Delay_Fs"
plt.savefig("img/" + name)
