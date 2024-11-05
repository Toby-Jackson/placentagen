import numpy as np
def no_resistance_model(Q_m, Q_f, C_ma, C_fa, N_co, N_sa):
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

    D_t = 1 # Diffusion coefficient for terminal villous tissue
    L_tv = 1 # Length scale for TV geom, roughly exchange area divided by thickness
    F = lambda theta: 1 - np.e**(-1*theta)

    Damkohler_fetal = (D_t*L_tv*N_tv)/Q_f
    # L = ((D_t*L_tv)/Damkohler_fetal)*F(Damkohler_fetal)
    Damkohler_maternal = ((D_t*L_tv*N_tv)/Q_m) * (F(Damkohler_fetal)/Damkohler_fetal)

    N_tot = (F(Damkohler_fetal)/Damkohler_fetal) * (F(Damkohler_maternal)/Damkohler_maternal) * D_t * L_tv * (C_ma - C_fa)
    C_fv = (N_tot + Q_f*C_fa)/Q_f
    C_mv = (Q_m*C_ma - N_tot)/Q_m
    return C_fv, C_mv

def resistance_model(P_m, P_f, C_ma, C_fa, N_co, N_sa, R_uta, R_utv, R_co, R_fpa, R_fpv, R_tv):
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

    D_t = 1 # Diffusion coefficient for terminal villous tissue
    L_tv = 1 # Length scale for TV geom, roughly exchange area divided by thickness
    F = lambda theta: 1 - np.e**(-1*theta)

    Damkohler_fetal = (D_t*L_tv*N_tv)/Q_f
    # L = ((D_t*L_tv)/Damkohler_fetal)*F(Damkohler_fetal)
    Damkohler_maternal = ((D_t*L_tv*N_tv)/Q_m) * (F(Damkohler_fetal)/Damkohler_fetal)

    N_tot = (F(Damkohler_fetal)/Damkohler_fetal) * (F(Damkohler_maternal)/Damkohler_maternal) * D_t * L_tv * (C_ma - C_fa)
    C_fv = (N_tot + Q_f*C_fa)/Q_f
    C_mv = (Q_m*C_ma - N_tot)/Q_m
    return C_fv, C_mv