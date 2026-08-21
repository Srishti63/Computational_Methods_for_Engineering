import math 

def f(T):
    Ta = T + 273.15

    return(
        -139.34411
        +(1.57571e5/Ta)
        -(6.642308e7 / Ta**2)
        + (1.243800e10 / Ta**3)
        - (8.621949e11 / Ta**4)
        - math.log(10)
    )

def bisection(xl,xu, es = 0.00001,max_iter = 1000):

    if f(xl)*f(xu) >0:
        return None
    
    xr_old = 0

    for i in range(1, max_iter+1):
        xr = (xl + xu)/2

        if xr != 0:
            ea = abs((xr - xr_old) / xr) * 100
        else:
            ea = 100
        
        print(f"Iteration {i}: xr = {xr}")

        if f(xr) == 0:
            break

        if f(xl) * f(xr) < 0:
            xu = xr
        else:
            xl = xr

        if ea < es:
            break

        xr_old = xr

        if ea<es:
            break

        xr_old = xr

    return xr
    
xl = 0
xu = 40

root = bisection(xl,xu)

print("Temperature = ", root ,"degree Celcius" )
 
    
    
