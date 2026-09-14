import ast
import cmath
import json
import math
import sys

from core import (
    biseccion,
    regla_falsa,
    punto_fijo,
    newton,
    secante,
    muller,
    gauss_parcial,
    gauss_total,
    resolver_lu,
    resolver_cholesky,
    gauss_seidel,
    newton_multivariable_2d,
    imprimir_iteraciones,
)

REAL_ALLOWED = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'exp': math.exp,
    'log': math.log,
    'ln': math.log,
    'sqrt': math.sqrt,
    'pi': math.pi,
    'e': math.e,
    'abs': abs,
}

COMPLEX_ALLOWED = {
    'sin': cmath.sin,
    'cos': cmath.cos,
    'tan': cmath.tan,
    'exp': cmath.exp,
    'log': cmath.log,
    'ln': cmath.log,
    'sqrt': cmath.sqrt,
    'pi': math.pi,
    'e': math.e,
    'abs': abs,
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def make_func(expr, vars=('x',), complex_mode=False):
    """Convierte una expresión de texto en una función matemática segura."""
    allowed = COMPLEX_ALLOWED if complex_mode else REAL_ALLOWED
    expression = expr.replace('^', '**')
    tree = ast.parse(expression, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(
                f'Elemento no permitido: {type(node).__name__}'
            )
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in allowed
        ):
            raise ValueError('Función no permitida')
        if isinstance(node, ast.Name) and (
            node.id not in vars and node.id not in allowed
        ):
            raise ValueError(f'Nombre no permitido: {node.id}')
        if isinstance(node, ast.Constant) and (
            not isinstance(node.value, (int, float, complex))
            or isinstance(node.value, bool)
        ):
            raise ValueError('Constante no permitida')
    code = compile(tree, '<expr>', 'eval')
    return lambda *args: eval(
        code,
        {'__builtins__': {}},
        {**allowed, **dict(zip(vars, args))},
    )


def pm(M, n):
    """Imprime una matriz con un formato numérico uniforme."""
    print(f'\n{n}=')
    for row in M:
        values = ', '.join(f'{value:.12g}' for value in row)
        print(f'  [{values}]')


def run(s):
    """Selecciona y ejecuta el método descrito en la especificación JSON."""
    method = s['method']
    iterations = s.get('iterations')
    tolerance = s.get('tolerance')
    error_type = s.get('error_type', 'absolute')
    if method == 'biseccion':
        ans, rows = biseccion(
            make_func(s['f']), s['a'], s['b'],
            iteraciones=iterations,
            tolerancia=tolerance,
            error_type=error_type,
        )
    elif method == 'regla_falsa':
        ans, rows = regla_falsa(
            make_func(s['f']), s['a'], s['b'],
            iteraciones=iterations,
            tolerancia=tolerance,
            error_type=error_type,
        )
    elif method == 'punto_fijo':
        ans, rows = punto_fijo(
            make_func(s['g']), s['x0'],
            iteraciones=iterations,
            tolerancia=tolerance,
            error_type=error_type,
        )
    elif method == 'newton':
        ans, rows = newton(
            make_func(s['f']), make_func(s['df']), s['x0'],
            iteraciones=iterations,
            tolerancia=tolerance,
            error_type=error_type,
        )
    elif method == 'secante':
        ans, rows = secante(
            make_func(s['f']), s['x0'], s['x1'],
            iteraciones=iterations,
            tolerancia=tolerance,
            error_type=error_type,
        )
    elif method == 'muller':
        ans, rows = muller(
            make_func(s['f'], complex_mode=True),
            s['x0'], s['x1'], s['x2'],
            iteraciones=iterations,
            tolerancia=tolerance,
            error_type=error_type,
        )
    elif method == 'gauss_parcial':
        x,U,b,st=gauss_parcial(s['A'],s['b']); print('Intercambios:',st); pm(U,'U'); print('\nx=',x); return
    elif method == 'gauss_total':
        x,U,b,st,p=gauss_total(s['A'],s['b']); print('Intercambios:',st); print('Permutación:',p); pm(U,'U'); print('\nx=',x); return
    elif method in ('doolittle', 'crout'):
        x, L, U, y = resolver_lu(s['A'], s['b'], method)
        pm(L, 'L')
        pm(U, 'U')
        print('\ny=', y, '\nx=', x)
        return
    elif method == 'cholesky':
        x,L,LT,y=resolver_cholesky(s['A'],s['b']); pm(L,'L'); pm(LT,'L^T'); print('\ny=',y,'\nx=',x); return
    elif method == 'gauss_seidel':
        ans, rows = gauss_seidel(
            s['A'], s['b'], s.get('x0'),
            iteraciones=iterations,
            tolerancia=tolerance,
            error_type=error_type,
        )
    elif method == 'newton_multivariable_2d':
        variables = ('x', 'y')
        ans, rows = newton_multivariable_2d(
            make_func(s['f1'], variables), make_func(s['f2'], variables),
            make_func(s['j11'], variables), make_func(s['j12'], variables),
            make_func(s['j21'], variables), make_func(s['j22'], variables),
            s['x0'], s['y0'],
            iteraciones=iterations,
            tolerancia=tolerance,
            error_type=error_type,
        )
    else:
        raise ValueError('Método no soportado')
    imprimir_iteraciones(rows)
    print('\nResultado:', ans)
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Uso: python runner.py problema.json')

    with open(sys.argv[1], encoding='utf-8') as file:
        run(json.load(file))
