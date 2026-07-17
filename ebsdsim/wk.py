"""Weickenmeier-Kohl atomic scattering factors."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from ebsdsim.wk_params import WK_A_PARAM, WK_B_PARAM

PI = np.pi
EULER = 0.5772156649015329

ComplexPair = tuple[float, float]
_WK_I_IDX = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.intp)
_WK_J_IDX = np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3], dtype=np.intp)


class ScatterFactorWKOptions(TypedDict, total=False):
    include_core: bool
    include_phonon: bool


def _poly(coeffs: list[float], x: float) -> float:
    result = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        result = result * x + coeffs[i]
    return result


def _poly_vec(coeffs: list[float], x: NDArray[np.floating]) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    result = np.full_like(x, coeffs[-1], dtype=np.float64)
    for i in range(len(coeffs) - 2, -1, -1):
        result = result * x + coeffs[i]
    return result


def exp1(x: float) -> float:
    if x == 0:
        return float("inf")
    if x < 0:
        return float("nan")
    if x <= 1:
        y = 0.6637353897094727
        p = [
            0.0865197248079398, 0.03209136653035592, -0.2450882166397615,
            -0.03680317362579437, -0.003991671060811133, -0.00011150779292119786,
        ]
        q = [
            1, 0.37091387659397013, 0.05677067710420753, 0.004273476000171037,
            0.00013104990079843468, -0.5286110295202171e-6,
        ]
        return _poly(p, x) / _poly(q, x) + x - np.log(x) - y
    p = [
        -0.12101319065772557e-18, -0.9999999999999988, -43.305866081181795,
        -724.5814827914625, -6046.825011271104, -27182.625446673397,
        -66598.26523454186, -86273.15677116495, -54844.45872264021,
        -14751.489578612845, -1185.4572031520103,
    ]
    q = [
        1, 45.30586608118015, 809.1932149545503, 7417.3762445468955,
        38129.55944848185, 113057.05869159631, 192104.04779022798,
        180329.49838050182, 86722.34034673347, 18455.412473772205,
        1229.2078418240305, -0.776491285282331,
    ]
    r = 1.0 / x
    return (1 + _poly(p, r) / _poly(q, r)) * np.exp(-x) * r


def expi(x: float) -> float:
    if x == 0:
        return float("-inf")
    if x < 0:
        return -exp1(-x)
    if x <= 6:
        p = [
            2.986772243435986, 0.3563436187693774, 0.7808360762837308,
            0.114670926327032, 0.04994347735765153, 0.007262245933412282,
            0.001154782372278043, 0.0001164195236097652, 0.7982963656792697e-5,
            0.27770562544020087e-6,
        ]
        q = [
            1, -1.1709041236541391, 0.6221510984601675, -0.1951147820694954,
            0.039152343139296724, -0.005048001586637057, 0.0003890340074360654,
            -0.1389725896017817e-4,
        ]
        root = 0.37250741078136663
        t1 = x / 3 - 1
        t = x - root
        correction = np.log1p(t / root) if abs(t) < 0.1 else np.log(x / root)
        return (_poly(p, t1) / _poly(q, t1)) * t + correction
    if x <= 10:
        y = 1.1589851379394531
        p = [
            0.001393240861994028, -0.034992122182388874, -0.026409552075413485,
            -0.007612240030054764, -0.0024749620959214363, -0.00037488591794210026,
            -0.5540862720248818e-4, -0.3964876489248045e-5,
        ]
        q = [
            1, 0.7446255668232721, 0.32906109501176706, 0.10012862497731387,
            0.022385109912850635, 0.0036533419074231665, 0.00040245340851247684,
            0.2636496307202557e-4,
        ]
        t = x / 2 - 4
        return (y + _poly(p, t) / _poly(q, t)) * np.exp(x) / x + x
    if x <= 20:
        y = 1.0869731903076172
        p = [
            -0.008938910943569457, -0.048460773012713405, -0.06528104442222369,
            -0.04784475726473097, -0.02260592189237771, -0.007206036369174821,
            -0.0015594194703597203, -0.0002097500226602009, -0.1386522003491826e-4,
        ]
        q = [
            1, 1.970172140390612, 1.8623246504307316, 1.0960143709033752,
            0.43887328577308887, 0.1225377319796861, 0.02334584782757693,
            0.0027817076916330367, 0.00015915028116610876,
        ]
        t = x / 5 - 3
        return (y + _poly(p, t) / _poly(q, t)) * np.exp(x) / x + x
    y = 1.0130653381347656
    p = [
        -0.013065338134765624, 0.19029710559486577, 94.73650945371972,
        -2516.3532367984426, 18932.0850014926, -38703.14313620567,
    ]
    q = [
        1, 61.97335928494399, -2354.562113234202, 22329.145948989308,
        -70126.24514039657, 54738.28331477755, 8297.162963565184,
    ]
    t = 1.0 / x
    return (y + _poly(p, t) / _poly(q, t)) * np.exp(x) / x + x


def exp1_vec(x: NDArray[np.floating]) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[x == 0] = np.inf
    pos = x > 0
    xp = x[pos]
    if xp.size == 0:
        return out
    chunk = np.empty_like(xp)
    small = xp <= 1
    if np.any(small):
        xs = xp[small]
        y = 0.6637353897094727
        p = [
            0.0865197248079398, 0.03209136653035592, -0.2450882166397615,
            -0.03680317362579437, -0.003991671060811133, -0.00011150779292119786,
        ]
        q = [
            1, 0.37091387659397013, 0.05677067710420753, 0.004273476000171037,
            0.00013104990079843468, -0.5286110295202171e-6,
        ]
        chunk[small] = _poly_vec(p, xs) / _poly_vec(q, xs) + xs - np.log(xs) - y
    large = xp > 1
    if np.any(large):
        xl = xp[large]
        p = [
            -0.12101319065772557e-18, -0.9999999999999988, -43.305866081181795,
            -724.5814827914625, -6046.825011271104, -27182.625446673397,
            -66598.26523454186, -86273.15677116495, -54844.45872264021,
            -14751.489578612845, -1185.4572031520103,
        ]
        q = [
            1, 45.30586608118015, 809.1932149545503, 7417.3762445468955,
            38129.55944848185, 113057.05869159631, 192104.04779022798,
            180329.49838050182, 86722.34034673347, 18455.412473772205,
            1229.2078418240305, -0.776491285282331,
        ]
        r = 1.0 / xl
        chunk[large] = (1 + _poly_vec(p, r) / _poly_vec(q, r)) * np.exp(-xl) * r
    out[pos] = chunk
    return out


def expi_vec(x: NDArray[np.floating]) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    out[x == 0] = -np.inf
    neg = x < 0
    if np.any(neg):
        out[neg] = -exp1_vec(-x[neg])
    pos = x > 0
    if not np.any(pos):
        return out
    xp = x[pos]
    chunk = np.zeros_like(xp)
    m6 = xp <= 6
    if np.any(m6):
        xs = xp[m6]
        p = [
            2.986772243435986, 0.3563436187693774, 0.7808360762837308,
            0.114670926327032, 0.04994347735765153, 0.007262245933412282,
            0.001154782372278043, 0.0001164195236097652, 0.7982963656792697e-5,
            0.27770562544020087e-6,
        ]
        q = [
            1, -1.1709041236541391, 0.6221510984601675, -0.1951147820694954,
            0.039152343139296724, -0.005048001586637057, 0.0003890340074360654,
            -0.1389725896017817e-4,
        ]
        root = 0.37250741078136663
        t1 = xs / 3 - 1
        t = xs - root
        # Masked branches: np.where would evaluate log1p(t/root) for xs≈0 where
        # t/root≈-1, triggering a spurious divide-by-zero warning.
        correction = np.empty_like(xs)
        near = np.abs(t) < 0.1
        if np.any(near):
            correction[near] = np.log1p(t[near] / root)
        far = ~near
        if np.any(far):
            with np.errstate(divide="ignore", invalid="ignore"):
                correction[far] = np.log(xs[far] / root)
        chunk[m6] = (_poly_vec(p, t1) / _poly_vec(q, t1)) * t + correction
    m10 = (xp > 6) & (xp <= 10)
    if np.any(m10):
        xs = xp[m10]
        y = 1.1589851379394531
        p = [
            0.001393240861994028, -0.034992122182388874, -0.026409552075413485,
            -0.007612240030054764, -0.0024749620959214363, -0.00037488591794210026,
            -0.5540862720248818e-4, -0.3964876489248045e-5,
        ]
        q = [
            1, 0.7446255668232721, 0.32906109501176706, 0.10012862497731387,
            0.022385109912850635, 0.0036533419074231665, 0.00040245340851247684,
            0.2636496307202557e-4,
        ]
        t = xs / 2 - 4
        chunk[m10] = (y + _poly_vec(p, t) / _poly_vec(q, t)) * np.exp(xs) / xs + xs
    m20 = (xp > 10) & (xp <= 20)
    if np.any(m20):
        xs = xp[m20]
        y = 1.0869731903076172
        p = [
            -0.008938910943569457, -0.048460773012713405, -0.06528104442222369,
            -0.04784475726473097, -0.02260592189237771, -0.007206036369174821,
            -0.0015594194703597203, -0.0002097500226602009, -0.1386522003491826e-4,
        ]
        q = [
            1, 1.970172140390612, 1.8623246504307316, 1.0960143709033752,
            0.43887328577308887, 0.1225377319796861, 0.02334584782757693,
            0.0027817076916330367, 0.00015915028116610876,
        ]
        t = xs / 5 - 3
        chunk[m20] = (y + _poly_vec(p, t) / _poly_vec(q, t)) * np.exp(xs) / xs + xs
    mbig = xp > 20
    if np.any(mbig):
        xs = xp[mbig]
        y = 1.0130653381347656
        p = [
            -0.013065338134765624, 0.19029710559486577, 94.73650945371972,
            -2516.3532367984426, 18932.0850014926, -38703.14313620567,
        ]
        q = [
            1, 61.97335928494399, -2354.562113234202, 22329.145948989308,
            -70126.24514039657, 54738.28331477755, 8297.162963565184,
        ]
        t = 1.0 / xs
        chunk[mbig] = (y + _poly_vec(p, t) / _poly_vec(q, t)) * np.exp(xs) / xs + xs
    out[pos] = chunk
    return out


_RIH2_TABLE = np.array([
    1, 1.005051, 1.010206, 1.015472, 1.020852, 1.026355, 1.031985, 1.037751,
    1.043662, 1.049726, 1.055956, 1.062364, 1.068965, 1.07578, 1.08283, 1.09014,
    1.097737, 1.105647, 1.113894, 1.122497, 1.13147,
], dtype=np.float64)


def _rih2(x: float) -> float:
    idx = max(0, min(len(_RIH2_TABLE) - 2, int(np.floor(200 / x))))
    return (
        _RIH2_TABLE[idx]
        + 200 * (_RIH2_TABLE[idx + 1] - _RIH2_TABLE[idx]) * (1 / x - 0.5e-3 * idx)
    )


def _rih2_vec(x: NDArray[np.floating]) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    idx = np.floor(200 / x).astype(np.intp)
    idx = np.clip(idx, 0, len(_RIH2_TABLE) - 2)
    inv_x = 1.0 / x
    return _RIH2_TABLE[idx] + 200 * (_RIH2_TABLE[idx + 1] - _RIH2_TABLE[idx]) * (inv_x - 0.5e-3 * idx)


def _rih1_vec(x1: NDArray[np.floating], x2: NDArray[np.floating], x3: NDArray[np.floating]) -> NDArray[np.float64]:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    x3 = np.asarray(x3, dtype=np.float64)
    out = np.empty_like(x1, dtype=np.float64)
    m00 = (x2 <= 20) & (x3 <= 20)
    if np.any(m00):
        out[m00] = np.exp(-x1[m00]) * (expi_vec(x2[m00]) - expi_vec(x3[m00]))
    m10 = (x2 > 20) & (x3 <= 20)
    if np.any(m10):
        out[m10] = (
            np.exp(x2[m10] - x1[m10]) * _rih2_vec(x2[m10]) / x2[m10]
            - np.exp(-x1[m10]) * expi_vec(x3[m10])
        )
    m01 = (x2 <= 20) & (x3 > 20)
    if np.any(m01):
        out[m01] = (
            np.exp(-x1[m01]) * expi_vec(x2[m01])
            - np.exp(x3[m01] - x1[m01]) * _rih2_vec(x3[m01]) / x3[m01]
        )
    m11 = (x2 > 20) & (x3 > 20)
    if np.any(m11):
        out[m11] = (
            np.exp(x2[m11] - x1[m11]) * _rih2_vec(x2[m11]) / x2[m11]
            - np.exp(x3[m11] - x1[m11]) * _rih2_vec(x3[m11]) / x3[m11]
        )
    return out


def _rih1(x1: float, x2: float, x3: float) -> float:
    if x2 <= 20 and x3 <= 20:
        return np.exp(-x1) * (expi(x2) - expi(x3))
    if x2 > 20 and x3 <= 20:
        return np.exp(x2 - x1) * _rih2(x2) / x2 - np.exp(-x1) * expi(x3)
    if x2 <= 20 and x3 > 20:
        return np.exp(-x1) * expi(x2) - np.exp(x3 - x1) * _rih2(x3) / x3
    return np.exp(x2 - x1) * _rih2(x2) / x2 - np.exp(x3 - x1) * _rih2(x3) / x3


def scatter_factor_wk_array(
    g: NDArray[np.floating],
    z: int,
    thermal_sigma: float,
    voltage_kv: float,
    opts: ScatterFactorWKOptions | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not (isinstance(z, int) and 1 <= z <= WK_A_PARAM.shape[0]):
        raise ValueError(f"WK atomic number must be 1..{WK_A_PARAM.shape[0]} (got {z})")
    g_val = np.asarray(g, dtype=np.float64)
    options = opts or {}
    include_core = options.get("include_core", True)
    include_phonon = options.get("include_phonon", True)
    a = WK_A_PARAM[z - 1]
    b = WK_B_PARAM[z - 1]
    s = g_val / (4 * PI)
    s2 = s * s
    f_real = np.zeros_like(g_val)
    for i in range(4):
        argu = b[i] * s2
        term1 = a[i] * b[i] * (1 - 0.5 * argu)
        safe_s2 = np.where(s2 == 0, 1.0, s2)
        term2 = np.where(s2 == 0, a[i] * b[i], a[i] * (1 - np.exp(-argu)) / safe_s2)
        term3 = np.where(s2 == 0, a[i] * b[i], a[i] / safe_s2)
        f_real += np.where(argu < 0.1, term1, np.where(argu <= 20, term2, term3))
    dwf = np.exp(-0.5 * thermal_sigma * thermal_sigma * g_val * g_val)
    k0 = 0.5068 * np.sqrt(1022 * voltage_kv + voltage_kv * voltage_kv)
    f_core = np.zeros_like(g_val)
    if include_core:
        a0 = 0.5289
        de = 6e-3 * z
        theta_e = de / (2 * voltage_kv) * (2 * voltage_kv + 1022) / (voltage_kv + 1022)
        r_val = 0.885 * a0 / (z ** (1 / 3))
        ta = 1 / (k0 * r_val)
        tb = g_val / (2 * k0)
        omega = 2 * tb / ta
        kappa = theta_e / ta
        k2 = kappa * kappa
        o2 = omega * omega
        sqrt_o = np.sqrt(o2 + 4 * k2)
        x1 = np.where(
            omega == 0,
            0.0,
            omega / ((1 + o2) * sqrt_o) * np.log((omega + sqrt_o) / (2 * kappa)),
        )
        x2 = (
            1
            / np.sqrt((1 + o2) * (1 + o2) + 4 * k2 * o2)
            * np.log(
                (1 + 2 * k2 + o2 + np.sqrt((1 + o2) * (1 + o2) + 4 * k2 * o2))
                / (2 * kappa * np.sqrt(1 + k2))
            )
        )
        denom_x3 = omega * np.sqrt(o2 + 4 * (1 + k2))
        x3 = np.where(
            omega > 1e-2,
            np.divide(
                np.log((omega + np.sqrt(o2 + 4 * (1 + k2))) / (2 * np.sqrt(1 + k2))),
                denom_x3,
                out=np.zeros_like(omega),
                where=denom_x3 != 0,
            ),
            1 / (4 * (1 + k2)),
        )
        hi = 2 * z / (ta * ta) * (-x1 + x2 - x3)
        f_core = 4 / (a0 * a0) * 2 * PI / (k0 * k0) * hi
    f_phon = np.zeros_like(g_val)
    if include_phonon:
        m_val = 0.5 * thermal_sigma * thermal_sigma
        g2 = g_val * g_val
        exp_dw = np.exp(-m_val * g2)
        g0 = g_val == 0
        for p in range(10):
            ii = int(_WK_I_IDX[p])
            jj = int(_WK_J_IDX[p])
            ai = a[ii] * (4 * PI) ** 2
            aj = a[jj] * (4 * PI) ** 2
            bi = b[ii] / (4 * PI) ** 2
            bj = b[jj] / (4 * PI) ** 2
            mult = 1 if ii == jj else 2
            i1 = np.zeros_like(g_val)
            i2 = np.zeros_like(g_val)
            if np.any(g0):
                i1[g0] = PI * (bi * np.log((bi + bj) / bi) + bj * np.log((bi + bj) / bj))
                i2[g0] = PI * (
                    (bi + 2 * m_val) * np.log((bi + bj + 2 * m_val) / (bi + 2 * m_val))
                    + bj * np.log((bi + bj + 2 * m_val) / (bj + 2 * m_val))
                    + 2 * m_val * np.log((2 * m_val) / (bj + 2 * m_val))
                )
            gnz = ~g0
            if np.any(gnz):
                gv = g_val[gnz]
                g2v = g2[gnz]
                pi_by_g2 = PI / g2v
                bi_bj = bi + bj
                bi_bj_term = -bi * bj * g2v / bi_bj
                bi2 = bi * bi * g2v / bi_bj
                bj2 = bj * bj * g2v / bi_bj
                i1[gnz] = pi_by_g2 * (
                    2 * EULER + np.log(bi * g2v) + np.log(bj * g2v) - 2 * expi_vec(bi_bj_term)
                    + _rih1_vec(bi * g2v, bi2, bi * g2v)
                    + _rih1_vec(bj * g2v, bj2, bj * g2v)
                )
                bi_m = bi + m_val
                bj_m = bj + m_val
                bij_m = bi + bj + 2 * m_val
                i2[gnz] = pi_by_g2 * (
                    2
                    * (
                        expi_vec(-m_val * bi_m * g2v / (bi + 2 * m_val))
                        + expi_vec(-m_val * bj_m * g2v / (bj + 2 * m_val))
                        - expi_vec(-bi_m * bj_m * g2v / bij_m)
                        - expi_vec(-0.5 * m_val * g2v)
                    )
                    + _rih1_vec(m_val * g2v, 0.5 * m_val * g2v, m_val * m_val * g2v / (bi + 2 * m_val))
                    + _rih1_vec(m_val * g2v, 0.5 * m_val * g2v, m_val * m_val * g2v / (bj + 2 * m_val))
                    + _rih1_vec(bi_m * g2v, bi_m * bi_m * g2v / bij_m, bi_m * bi_m * g2v / (bi + 2 * m_val))
                    + _rih1_vec(bj_m * g2v, bj_m * bj_m * g2v / bij_m, bj_m * bj_m * g2v / (bj + 2 * m_val))
                )
            f_phon += mult * ai * aj * (exp_dw * i1 - i2)
    gamma = (voltage_kv + 511) / 511
    real = f_real * dwf * gamma
    imag = np.maximum(0.0, (f_core * dwf + f_phon) * gamma * gamma / (4 * PI * k0))
    return real, imag


def scatter_factor_wk(
    g: float,
    z: int,
    thermal_sigma: float,
    voltage_kv: float,
    opts: ScatterFactorWKOptions | None = None,
) -> ComplexPair:
    real, imag = scatter_factor_wk_array(np.asarray([g], dtype=np.float64), z, thermal_sigma, voltage_kv, opts)
    return float(real[0]), float(imag[0])
