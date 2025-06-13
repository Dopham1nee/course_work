import sympy as sp 

def solve() -> str:
    s = sp.Symbol('s')
    v0, r = sp.symbols('v0 r', positive=True)
    v = sp.Function('v')(s)
    
    eq = sp.Eq(sp.Derivative(v, s), -v / r) 
    
    solution = sp.dsolve(eq, v, ics={v.subs(s, 0): v0})

    res = sp.simplify(solution.rhs)
    return f'v(s) = {res}'
