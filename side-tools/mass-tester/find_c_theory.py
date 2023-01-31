from math import pi, e, erf
sqrt2 = 2**0.5

def part(Z, z, beta, c):
    group_1 = z / (c * sqrt2)
    group_2 = group_1 * beta
    coeff = Z*z * e**(group_2**2)
    part_1 = 0.5 * (1 + erf(-group_2))
    part_2 = e**(0.5 * (1 - beta**2) * z**2 / c**2) \
             * 0.5 * (1 + erf(-group_1))
    return coeff * (part_1 - part_2)

def v_th_minus_I(A, a, beta, c):
    value = (beta/(1 - beta)) * ((2*pi)**0.5 / c) * part(A, a, beta, c)
    return value

def find_c(A, a, beta, target):
    lower_bound = 1
    upper_bound = 1
    breakout = 0
    while v_th_minus_I(A, a, beta, lower_bound) < target:
        lower_bound *= 0.5
        breakout += 1
        if breakout > 20:
            raise RuntimeError(
                "Couldn't find a small enough lower bound. A={}, a={}, beta={}"\
                .format(A, a, beta))
    breakout = 0
    while v_th_minus_I(A, a, beta, upper_bound) > target:
        upper_bound *= 2
        breakout += 1
        if breakout > 20:
            raise RuntimeError(
                "Couldn't find a big enough upper bound. A={}, a={}, beta={}"\
                .format(A, a, beta))

    while upper_bound - lower_bound > 0.005:
        new_bound = 0.5*(upper_bound + lower_bound)
        if v_th_minus_I(A, a, beta, new_bound) > target:
            lower_bound = new_bound
        else:
            upper_bound = new_bound

    segments = str(lower_bound).split(".")
    truncated = float(segments[0] + "." + segments[1][:2])
    if v_th_minus_I(A, a, beta, truncated + 0.01) > target:
        return truncated + 0.01
    else:
        return truncated
        
    

    
    
    
