import cmath
import math

def _err(a,p):
    """Calcula el error absoluto, relativo y porcentual entre dos valores."""
    if p is None:
        return None, None, None
    ea = abs(a - p)
    er = ea / abs(a) if a != 0 else float('inf')
    return ea, er, er * 100


def _cumple_tolerancia(ea, er, erp, tolerancia, error_type):
    """Indica si el error calculado cumple el tipo de tolerancia elegido."""
    if tolerancia is None:
        return False
    if error_type == 'absolute':
        return ea < tolerancia
    if error_type == 'relative':
        return er < tolerancia
    if error_type == 'relative_percent':
        return erp < tolerancia
    raise ValueError(f'Tipo de error desconocido: {error_type}')

def imprimir_iteraciones(rows):
    """Muestra las iteraciones recibidas en forma de tabla."""
    if not rows:
        return

    headers = list(rows[0])
    print(' | '.join(headers))
    print('-' * 100)
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if value is None:
                text = '-'
            elif isinstance(value, complex):
                text = f'{value.real:.12g}{value.imag:+.12g}j'
            elif isinstance(value, float):
                text = f'{value:.12g}'
            else:
                text = str(value)
            values.append(text)
        print(' | '.join(values))

def biseccion(f,a,b,iteraciones=None,tolerancia=None,max_iter=1000,
              error_type='absolute'):
    """Busca una raíz dividiendo sucesivamente un intervalo que la contiene."""
    fa, fb = f(a), f(b)
    if not (math.isfinite(float(fa)) and math.isfinite(float(fb))):
        raise ValueError('La función no es finita en uno de los extremos.')
    if fa * fb > 0:
        raise ValueError('El intervalo no encierra una raíz.')
    rows = []
    previous = None
    number_of_iterations = (
        iteraciones if iteraciones is not None else max_iter
    )
    for iteration in range(1, number_of_iterations + 1):
        x = (a + b) / 2
        fx = f(x)
        ea, er, ep = _err(x, previous)
        rows.append({'i': iteration, 'x': x, 'f(x)': fx,
                     'Ea': ea, 'Er': er, 'Er%': ep})
        if _cumple_tolerancia(ea, er, ep, tolerancia, error_type):
            break
        if fa * fx < 0:
            b, fb = x, fx
        elif fx * fb < 0:
            a, fa = x, fx
        else:
            break
        previous = x
    return x, rows

def regla_falsa(f,a,b,iteraciones=None,tolerancia=None,max_iter=1000,
                error_type='absolute'):
    """Busca una raíz usando la intersección de la secante con el eje x."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError('El intervalo no encierra una raíz.')
    rows = []
    prev = None
    n = iteraciones if iteraciones is not None else max_iter
    for i in range(1, n + 1):
        x = b - fb * (a - b) / (fa - fb)
        fx = f(x)
        ea, er, ep = _err(x, prev)
        rows.append({'i': i, 'x': x, 'f(x)': fx,
                     'Ea': ea, 'Er': er, 'Er%': ep})
        if _cumple_tolerancia(ea, er, ep, tolerancia, error_type):
            break
        if fa * fx < 0:
            b, fb = x, fx
        elif fa * fx > 0:
            a, fa = x, fx
        else:
            break
        prev = x
    return x, rows

def punto_fijo(g,x0,iteraciones=None,tolerancia=None,max_iter=1000,
               error_type='absolute'):
    """Obtiene un punto fijo aplicando repetidamente la función g."""
    rows = []
    x = x0
    n = iteraciones if iteraciones is not None else max_iter
    for i in range(1, n + 1):
        xn = g(x)
        ea, er, ep = _err(xn, x)
        rows.append({'i': i, 'x': xn, 'Ea': ea,
                     'Er': er, 'Er%': ep})
        x = xn
        if _cumple_tolerancia(ea, er, ep, tolerancia, error_type):
            break
    return x, rows

def newton(f,df,x0,iteraciones=None,tolerancia=None,max_iter=1000,
           error_type='absolute'):
    """Busca una raíz mediante el método de Newton-Raphson."""
    rows = []
    x = x0
    n = iteraciones if iteraciones is not None else max_iter
    for i in range(1, n + 1):
        derivative = df(x)
        if abs(derivative) < 1e-15:
            raise ZeroDivisionError('Derivada ~0.')
        xn = x - f(x) / derivative
        ea, er, ep = _err(xn, x)
        rows.append({'i': i, 'x': xn, 'f(x)': f(xn),
                     'Ea': ea, 'Er': er, 'Er%': ep})
        x = xn
        if _cumple_tolerancia(ea, er, ep, tolerancia, error_type):
            break
    return x, rows

def secante(f,x0,x1,iteraciones=None,tolerancia=None,max_iter=1000,
            error_type='absolute'):
    """Busca una raíz aproximando la derivada con dos puntos consecutivos."""
    rows = []
    n = iteraciones if iteraciones is not None else max_iter
    for i in range(1, n + 1):
        f0, f1 = f(x0), f(x1)
        denominator = f1 - f0
        if abs(denominator) < 1e-15:
            raise ZeroDivisionError('Denominador ~0.')
        x2 = x1 - f1 * (x1 - x0) / denominator
        ea, er, ep = _err(x2, x1)
        rows.append({'i': i, 'x': x2, 'f(x)': f(x2),
                     'Ea': ea, 'Er': er, 'Er%': ep})
        x0, x1 = x1, x2
        if _cumple_tolerancia(ea, er, ep, tolerancia, error_type):
            break
    return x1, rows

def muller(f,x0,x1,x2,iteraciones=None,tolerancia=None,max_iter=1000,
           error_type='absolute'):
    """Busca una raíz ajustando una parábola a tres aproximaciones."""
    x0, x1, x2 = complex(x0), complex(x1), complex(x2)
    rows = []
    prev = x2
    n = iteraciones if iteraciones is not None else max_iter
    for i in range(1, n + 1):
        h0, h1 = x1 - x0, x2 - x1
        if h0 == 0 or h1 == 0:
            raise ZeroDivisionError(
                'Dos aproximaciones consecutivas son iguales.'
            )
        d0 = (f(x1) - f(x0)) / h0
        d1 = (f(x2) - f(x1)) / h1
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = f(x2)
        disc = cmath.sqrt(b * b - 4 * a * c)
        d1_ = b + disc
        d2_ = b - disc
        den = d1_ if abs(d1_) > abs(d2_) else d2_
        if den == 0:
            raise ZeroDivisionError('Denominador cero en Müller.')
        x3 = x2 + (-2 * c) / den
        ea, er, ep = _err(x3, prev)
        rows.append({'i': i, 'x': x3, 'f(x)': f(x3),
                     'Ea': ea, 'Er': er, 'Er%': ep})
        x0, x1, x2 = x1, x2, x3
        prev = x3
        if _cumple_tolerancia(ea, er, ep, tolerancia, error_type):
            break
    return (x2.real if abs(x2.imag) < 1e-12 else x2), rows

def sustitucion_adelante(L,b):
    """Resuelve un sistema triangular inferior hacia adelante."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def sustitucion_atras(U,b):
    """Resuelve un sistema triangular superior hacia atrás."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def gauss_parcial(A,b):
    """Resuelve un sistema mediante Gauss con pivoteo parcial."""
    A = [list(map(float, row)) for row in A]
    b = list(map(float, b))
    n = len(A)
    steps = []
    for k in range(n - 1):
        p = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[p][k]) < 1e-15:
            raise ValueError('Matriz singular o pivote nulo.')
        if p != k:
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
            steps.append(f'F{k + 1}<->F{p + 1}')
        for i in range(k + 1, n):
            multiplier = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= multiplier * A[k][j]
            b[i] -= multiplier * b[k]
    if n and abs(A[-1][-1]) < 1e-15:
        raise ValueError('Matriz singular o pivote nulo.')
    return sustitucion_atras(A,b),A,b,steps

def gauss_total(A,b):
    """Resuelve un sistema mediante Gauss con pivoteo total."""
    A = [list(map(float, row)) for row in A]
    b = list(map(float, b))
    n = len(A)
    perm = list(range(n))
    steps = []
    for k in range(n - 1):
        p, q = max(
            ((i, j) for i in range(k, n) for j in range(k, n)),
            key=lambda position: abs(A[position[0]][position[1]]),
        )
        if abs(A[p][q]) < 1e-15:
            raise ValueError('Matriz singular.')
        if p != k:
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
            steps.append(f'F{k + 1}<->F{p + 1}')
        if q != k:
            for row in A:
                row[k], row[q] = row[q], row[k]
            perm[k], perm[q] = perm[q], perm[k]
            steps.append(f'C{k + 1}<->C{q + 1}')
        for i in range(k + 1, n):
            multiplier = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= multiplier * A[k][j]
            b[i] -= multiplier * b[k]
    if n and abs(A[-1][-1]) < 1e-15:
        raise ValueError('Matriz singular.')
    xp = sustitucion_atras(A, b)
    x = [0.0] * n
    for i, original_index in enumerate(perm):
        x[original_index] = xp[i]
    return x, A, b, steps, perm

def doolittle(A):
    """Descompone una matriz en L y U con el método de Doolittle."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
        if abs(U[i][i]) < 1e-15:
            raise ZeroDivisionError('Pivote cero en Doolittle.')
        for k in range(i + 1, n):
            L[k][i] = (
                A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))
            ) / U[i][i]
    return L, U

def crout(A):
    """Descompone una matriz en L y U con el método de Crout."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        U[i][i] = 1.0
    for j in range(n):
        for i in range(j, n):
            L[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(j))
        if abs(L[j][j]) < 1e-15:
            raise ZeroDivisionError('Pivote cero en Crout.')
        for i in range(j + 1, n):
            U[j][i] = (
                A[j][i] - sum(L[j][k] * U[k][i] for k in range(j))
            ) / L[j][j]
    return L, U

def resolver_lu(A,b,metodo='doolittle'):
    """Descompone una matriz y resuelve el sistema triangular resultante."""
    L, U = doolittle(A) if metodo == 'doolittle' else crout(A)
    y = sustitucion_adelante(L, list(map(float, b)))
    x = sustitucion_atras(U, y)
    return x, L, U, y

def cholesky(A):
    """Descompone una matriz definida positiva como L por L traspuesta."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                value = A[i][i] - s
                if value <= 0:
                    raise ValueError('La matriz no es definida positiva.')
                L[i][j] = math.sqrt(value)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def resolver_cholesky(A,b):
    """Resuelve un sistema usando la descomposición de Cholesky."""
    L = cholesky(A)
    transposed_l = [list(row) for row in zip(*L)]
    y = sustitucion_adelante(L, list(map(float, b)))
    x = sustitucion_atras(transposed_l, y)
    return x, L, transposed_l, y

def gauss_seidel(A,b,x0=None,iteraciones=None,tolerancia=None,max_iter=1000,
                 error_type='absolute'):
    """Aproxima la solución de un sistema mediante Gauss-Seidel."""
    A = [list(map(float, row)) for row in A]
    b = list(map(float, b))
    n = len(A)
    x = [0.0] * n if x0 is None else list(map(float, x0))
    rows = []
    nit = iteraciones if iteraciones is not None else max_iter
    for iteration in range(1, nit + 1):
        old = x.copy()
        for i in range(n):
            if abs(A[i][i]) < 1e-15:
                raise ZeroDivisionError(f'Diagonal cero en fila {i+1}.')
            x[i] = (
                b[i]
                - sum(A[i][j] * x[j] for j in range(i))
                - sum(A[i][j] * old[j] for j in range(i + 1, n))
            ) / A[i][i]
        ea = max(abs(x[i] - old[i]) for i in range(n))
        maximum = max(abs(value) for value in x)
        er = ea / maximum if maximum != 0 else float('inf')
        row = {
            'i': iteration,
            **{f'x{j + 1}': x[j] for j in range(n)},
            'Ea': ea,
            'Er': er,
        }
        rows.append(row)
        if _cumple_tolerancia(ea, er, er * 100, tolerancia, error_type):
            break
    return x, rows

def newton_multivariable_2d(f1,f2,j11,j12,j21,j22,x0,y0,
                            iteraciones=None,tolerancia=None,max_iter=1000,
                            error_type='absolute'):
    """Resuelve un sistema de dos ecuaciones con Newton multivariable."""
    x, y = float(x0), float(y0)
    rows = []
    nit = iteraciones if iteraciones is not None else max_iter
    for iteration in range(1, nit + 1):
        a, b = j11(x, y), j12(x, y)
        c, d = j21(x, y), j22(x, y)
        determinant = a * d - b * c
        if abs(determinant) < 1e-15:
            raise ZeroDivisionError('Jacobiano singular.')
        r1, r2 = -f1(x, y), -f2(x, y)
        dx = (r1 * d - b * r2) / determinant
        dy = (a * r2 - r1 * c) / determinant
        x += dx
        y += dy
        ea = max(abs(dx), abs(dy))
        maximum = max(abs(x), abs(y))
        er = ea / maximum if maximum else float('inf')
        rows.append({
            'i': iteration,
            'x': x,
            'y': y,
            'f1': f1(x, y),
            'f2': f2(x, y),
            'Ea': ea,
            'Er': er,
        })
        if _cumple_tolerancia(ea, er, er * 100, tolerancia, error_type):
            break
    return (x, y), rows
