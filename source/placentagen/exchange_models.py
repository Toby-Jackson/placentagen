import numpy as np
def no_resistance_model(Q_m, Q_f, C_ma, C_fa, N_co, N_sa, verbose=False):
    """
    Parameters
    ----------
    Q_m Maternal flow
    Q_f Fetal flow
    C_ma Concentration in the maternal arterial blood supply
    C_fa Concentration in fetal arterial blood
    N_co Number of cotyledons
    N_sa Number of Spiral arteries

    Returns
    -------
    C_fv Concentration in fetal venous circulation
    C_mv Concentration in maternal venous circulation

    """
    N_tv = N_co*N_sa
    
    D_t = 2 # Diffusion coefficient for terminal villous tissue in m^2/s
    L_tv = 16.5*10**-6 # Length scale for TV geom, roughly exchange area divided by thickness, 15 - 18 mm
    F = lambda theta: 1 - np.e**(-1*theta)

    Damkohler_fetal = (D_t*L_tv*N_tv)/Q_f
    # L = ((D_t*L_tv)/Damkohler_fetal)*F(Damkohler_fetal)
    Damkohler_maternal = ((D_t*L_tv*N_tv)/Q_m) * (F(Damkohler_fetal)/Damkohler_fetal)

    N_tot = (F(Damkohler_fetal)/Damkohler_fetal) * (F(Damkohler_maternal)/Damkohler_maternal) * D_t * L_tv * N_tv * (C_ma - C_fa)
    if verbose:
        print(f"{N_tot} oxygen transferred, with fetal Damkohler: {Damkohler_fetal}, maternal Damkohler: "
              f"{Damkohler_maternal} and {N_tot} terminal villi")
    C_fv = (N_tot + Q_f*C_fa)/Q_f
    C_mv = (Q_m*C_ma - N_tot)/Q_m
    return C_fv, C_mv

def resistance_model(P_m, P_f, C_ma, C_fa, N_co, N_sa, R_uta, R_utv, R_co, R_fpa, R_fpv, R_tv, verbose=False):
    """
    Parameters
    ----------
    P_m Pressure drop across maternal circulation from maternal arterial to maternal venous
    P_f Pressure drop across fetal circulation from fetal arterial to fetal venous
    C_ma Concentration in the maternal arterial blood supply
    C_fa Concentration in fetal arterial blood
    N_co Number of cotyledons
    N_sa Number of Spiral arteries
    R_uta Resistance to flow in a typical uterine arterial network
    R_utv Resistance to flow in a typical uterine venous network
    R_co Resistance to flow across the intervillous space in a single cotyledon from spiral artery to decidual vein
    R_fpa Resistance to flow in a typical fetal arterial network
    R_fpv Resistance to flow in a typical fetal venous network
    R_tv Resistance to flow of a single capillary network in a terminal villae

    Returns
    -------
    C_fv Concentration in fetal venous circulation
    C_mv Concentration in maternal venous circulation

    """
    N_tv = N_co*N_sa

    Q_m = P_m/(R_uta + R_co + R_utv)
    Q_f = P_f/(R_fpa + R_fpv + R_tv)

    D_t = 2 # Diffusion coefficient for terminal villous tissue in m^2/s
    L_tv = 16.5*10**-6 # Length scale for TV geom, roughly exchange area divided by thickness, 15 - 18 mm
    F = lambda theta: 1 - np.e**(-1*theta)

    Damkohler_fetal = (D_t*L_tv*N_tv)/Q_f
    # L = ((D_t*L_tv)/Damkohler_fetal)*F(Damkohler_fetal)
    Damkohler_maternal = ((D_t*L_tv*N_tv)/Q_m) * (F(Damkohler_fetal)/Damkohler_fetal)

    N_tot = (F(Damkohler_fetal)/Damkohler_fetal) * (F(Damkohler_maternal)/Damkohler_maternal) * D_t * L_tv * N_tv * (C_ma - C_fa)
    if verbose:
        print(f"{N_tot} oxygen transferred, with fetal Damkohler: {Damkohler_fetal}, maternal Damkohler: "
              f"{Damkohler_maternal} and {N_tot} terminal villi")
    C_fv = (N_tot + Q_f*C_fa)/Q_f
    C_mv = (Q_m*C_ma - N_tot)/Q_m
    return C_fv, C_mv

def consumption(t, C):
    """
    :param t: time
    :param C: Concentration of Oxygen in Fetal circulation
    :return: dodt - rate of change in fetal oxygen conctration with respect to time
    This function models the change in fetal oxygen concentration as a function of time using Michaelis-Menten mechanics
    TODD: properly parameterise Max_consumption and K_m
    """
    Max_consumption = 0.1
    K_m = 0.4
    dodt = -Max_consumption/(K_m + C)
    return dodt

def oygen_consumption(C_fetal, cardiac_cyle_time):
    """
    :param C_fetal: Initial concentration of fetal oxygen
    :param cardiac_cyle_time: time it takes for the fetus to complete one cardiac cycle
    :return: two array-like objects with indexed values representing the time course of fetal oxygen concentration, the
    first array is time, and the second is oxygen concentration
    """
    sol= sp.integrate.solve_ivp(consumption, [0, cardiac_cyle_time], [C_fetal,])
    return sol.t, sol.y