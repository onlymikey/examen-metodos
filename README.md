# Kit offline de Métodos Numéricos

El LLM no hace las cuentas: solo convierte el enunciado en JSON. `runner.py` ejecuta algoritmos verificados de `core.py`.

## Prueba sin IA
```bash
python runner.py problema_ejemplo.json
```

## Uso con Ollama
```bash
export OLLAMA_MODEL="qwen2.5-coder:7b"   # cambia por el modelo que tengas
python agent.py
```

Pega algo como:
```text
Newton-Raphson, cinco iteraciones:
x^4 + 2x - 1 = 0
x0 = 0
```

y termina con una línea vacía.

## Métodos incluidos
bisección, punto fijo, Newton, secante, regla falsa, Müller, Gauss parcial/total, Doolittle, Crout, Cholesky, Gauss-Seidel y Newton multivariable 2D.

## Plan B
Si el modelo interpreta mal un signo o exponente, edita `problema.json` y ejecuta:
```bash
python runner.py problema.json
```

No toques `core.py` durante el examen salvo que de verdad aparezca un método nuevo.
