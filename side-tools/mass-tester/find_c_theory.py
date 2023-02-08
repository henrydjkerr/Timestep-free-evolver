from math import pi, e, erf
sqrt2 = 2**0.5

#Now adjusted to 1D equations


##def v_th_minus_I(A, a, beta, c):
##    gamma = a / (c * sqrt2)
##    #See eqn 7.21 on overleaf running review
##    try:
##        value = (beta/(1 - beta)) * (A/c) * 0.5 * e**(gamma**2) \
##                * (1 + erf(-beta*gamma) \
##                   - e**((1 - beta**2)*(gamma**2)) * (1 + erf(-gamma)))
##    except OverflowError:
##        print(A, a, beta, c)
##        raise
##    return value

def v_th_minus_I(A, a, beta, c):
    value = (beta/(1 - beta)) * A * e**((a*beta/(c*sqrt2))**2) * 0.5 \
            * (1 + erf(-a*beta/(c*sqrt2))
               - e**(0.5*(1 - beta**2)*((a/c)**2)) * (1 + erf(-a/(c*sqrt2))))
    return value            

##def find_bounds(A, a, beta, target):
##    test_1 = v_th_minus_I(A, a, beta, 1)
##    test_2 = v_th_minus_I(A, a, beta, 1.01)
##    if (test_2 - test_1) * (test_1 - target) > 0:
##        #1 is an upper bound on the interval containing the solution
##        factor = 0.5
##    else:
##        #1 is a lower bound on the interval containing the solution
##        factor = 2
##    bound = 1
##    breakout = 0
##    while (v_th_minus_I(A, a, beta, bound) - target) * (test_1 - target) > 0:
##        bound *= factor
##        breakout += 1
##        if breakout > 10:
##            print("Couldn't find a bound. A={}, a={}, beta={}"\
##                  .format(A, a, beta))
##            raise AssertionError
##    if bound > 1:
##        return 1, bound
##    else:
##        return bound, 1
  

def find_c(A, a, beta, target):
    lower_bound = 1
    upper_bound = 1
    breakout = 0
    while v_th_minus_I(A, a, beta, lower_bound) < target:
        lower_bound *= 0.5
        breakout += 1
        if breakout > 4:
            print("Couldn't find a small enough lower bound. A={}, a={}, beta={}"\
                  .format(A, a, beta))
            raise AssertionError                
    breakout = 0
    while v_th_minus_I(A, a, beta, upper_bound) > target:
        upper_bound *= 2
        breakout += 1
        if breakout > 10:
            print("Couldn't find a big enough upper bound. A={}, a={}, beta={}"\
                  .format(A, a, beta))
            raise AssertionError
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


#De-verified by older interval seeker
##def find_c(A, a, beta, target):
##    lower_bound, upper_bound = find_bounds(A, a, beta, target)
##    gradient_sign =   v_th_minus_I(A, a, beta, upper_bound) \
##                    - v_th_minus_I(A, a, beta, lower_bound)
##
##    while abs(upper_bound - lower_bound) > 0.005:
##        new_bound = 0.5*(upper_bound + lower_bound)
##        if (v_th_minus_I(A, a, beta, new_bound) - target) * gradient_sign > 0:
##            upper_bound = new_bound
##        else:
##            lower_bound = new_bound
##
##    segments = str(lower_bound).split(".")
##    truncated = float(segments[0] + "." + segments[1][:2])
##    if v_th_minus_I(A, a, beta, truncated + 0.01) > target:
##        return truncated + 0.01
##    else:
##        return truncated   
##        

    
    
    
