import json
import os
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")

SYSTEM = r'''
Eres un parser LOCAL para un examen de Métodos Numéricos.

Tu única tarea es convertir el enunciado del usuario a UN objeto JSON
compatible con un programa Python existente.

NO resuelvas el ejercicio.
NO inventes datos.
NO cambies nombres de claves.
NO agregues claves no solicitadas.
NO uses sinónimos de las claves indicadas.
Devuelve SOLO JSON válido, sin Markdown.

Métodos permitidos y formatos EXACTOS:

Newton 1D:
{
  "method": "newton",
  "f": "expresion",
  "df": "derivada",
  "x0": numero,
  "iterations": entero
}
o, si el ejercicio usa tolerancia:
{
  "method": "newton",
  "f": "expresion",
  "df": "derivada",
  "x0": numero,
  "tolerance": numero
}

Bisección:
{
  "method": "biseccion",
  "f": "expresion",
  "a": numero,
  "b": numero,
  "iterations": entero
}

Secante:
{
  "method": "secante",
  "f": "expresion",
  "x0": numero,
  "x1": numero,
  "iterations": entero
}
o con "tolerance".

Regla falsa:
{
  "method": "regla_falsa",
  "f": "expresion",
  "a": numero,
  "b": numero,
  "iterations": entero
}
o con "tolerance".

Punto fijo:
{
  "method": "punto_fijo",
  "g": "expresion",
  "x0": numero,
  "iterations": entero
}
No inventes g si no está dada.

Müller:
{
  "method": "muller",
  "f": "expresion",
  "x0": numero,
  "x1": numero,
  "x2": numero,
  "iterations": entero
}

Gauss parcial:
{
  "method": "gauss_parcial",
  "A": [[...], [...]],
  "b": [...]
}

Gauss total:
{
  "method": "gauss_total",
  "A": [[...], [...]],
  "b": [...]
}

Si el enunciado dice "pivoteo total", usa exactamente
"method": "gauss_total".
Si dice "pivoteo parcial", usa exactamente
"method": "gauss_parcial".
Nunca intercambies ambos métodos.

Doolittle:
{
  "method": "doolittle",
  "A": [[...], [...]],
  "b": [...]
}

Crout:
{
  "method": "crout",
  "A": [[...], [...]],
  "b": [...]
}

Cholesky:
{
  "method": "cholesky",
  "A": [[...], [...]],
  "b": [...]
}

Gauss-Seidel:
{
  "method": "gauss_seidel",
  "A": [[...], [...]],
  "b": [...],
  "x0": [...],
  "iterations": entero
}

Newton multivariable 2D:
{
  "method": "newton_multivariable_2d",
  "f1": "expresion",
  "f2": "expresion",
  "j11": "expresion",
  "j12": "expresion",
  "j21": "expresion",
  "j22": "expresion",
  "x0": numero,
  "y0": numero,
  "iterations": entero
}

Si el problema contiene DOS ecuaciones y pide Newton para sistemas,
DEBES usar exactamente el formato newton_multivariable_2d anterior.
Nunca uses "method": "newton" para un sistema de dos ecuaciones.
Cada función debe ir en su propia clave: f1, f2, j11, j12, j21 y j22.
Nunca representes múltiples funciones como listas ni como strings que
contengan listas.

Para:
x^2 + y^2 - 4 = 0
exp(-x) + 3*y - x = 0

las derivadas parciales son:
j11 = 2*x
j12 = 2*y
j21 = -exp(-x) - 1
j22 = 3

REGLAS DE EXPRESIONES:
- Multiplicación SIEMPRE con *
- Potencias con ^
- "dos por x" significa 2*x, JAMÁS 2/x.
- "x al cuadrado" significa x^2.
- "x al cubo" significa x^3.
- "x a la cuarta" significa x^4.
- Usa sin, cos, tan, exp, log, ln, sqrt.
- Conserva exactamente los números dados por el usuario.
- Conserva exactamente todos los exponentes: y^2 nunca puede convertirse en y.

REGLAS IMPORTANTES:
- Si el usuario pide exactamente N iteraciones, usa "iterations": N.
- Si NO menciona tolerancia, NO agregues "tolerance".
- Si menciona explícitamente el tipo de error, usa "error_type" con uno de
  estos valores: "absolute", "relative" o "relative_percent".
- Si no menciona el tipo de error, NO agregues "error_type"; el ejecutor usa
  error absoluto por defecto.
- Si dice x inicial igual a cero, usa "x0": 0.
- No cambies 0 por 1.
- Newton 1D usa f, df y x0. JAMÁS uses "function", "jacobian" o "initial_guess".
- Newton para dos ecuaciones usa SIEMPRE "newton_multivariable_2d" y las seis
  claves de funciones parciales indicadas arriba.
- Si df puede derivarse inequívocamente de f, calcúlala simbólicamente.
- Si falta información esencial, responde:
  {"error":"descripcion breve"}

Ejemplo:

Entrada:
Usa Newton Raphson para x a la cuarta mas dos por x menos uno.
Empieza con x igual a cero y realiza cinco iteraciones.

Salida:
{
  "method": "newton",
  "f": "x^4 + 2*x - 1",
  "df": "4*x^3 + 2",
  "x0": 0,
  "iterations": 5
}

Entrada:
Resuelve mediante Newton para sistemas:

x^2 + y^2 - 4 = 0
e^(-x) + 3y - x = 0

Usa x0 = 1.5, y0 = 0.4 y realiza 5 iteraciones.

Salida:
{
  "method": "newton_multivariable_2d",
  "f1": "x^2 + y^2 - 4",
  "f2": "exp(-x) + 3*y - x",
  "j11": "2*x",
  "j12": "2*y",
  "j21": "-exp(-x) - 1",
  "j22": "3",
  "x0": 1.5,
  "y0": 0.4,
  "iterations": 5
}
'''


def ask(txt):
    """Envía el enunciado a Ollama y devuelve el texto de su respuesta."""
    payload = {
        "model": MODEL,
        "stream": False,

        # No necesitamos razonamiento interno:
        # el modelo solo debe convertir el enunciado a JSON.
        "think": False,

        # Obliga a Ollama a producir JSON.
        "format": "json",

        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": txt},
        ],
        "options": {
            "temperature": 0,
        },
    }

    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout):
        raise RuntimeError(
            "Qwen tardó más de 60 segundos en responder."
        ) from None
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"No se pudo conectar con Ollama: {exc.reason}"
        ) from None
    except json.JSONDecodeError:
        raise RuntimeError(
            "Ollama devolvió una respuesta HTTP que no era JSON válido."
        ) from None

    try:
        return result["message"]["content"].strip()
    except (KeyError, TypeError):
        raise RuntimeError(
            f"Respuesta inesperada de Ollama: {result}"
        ) from None


def main():
    """Lee un ejercicio, obtiene su especificación y ejecuta el método."""
    print("Modelo:", MODEL)
    print("Escribe o pega el ejercicio. Termina con END en una línea aparte:\n")

    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip().upper() == "END":
            break

        lines.append(line)

    prompt = "\n".join(lines).strip()

    if not prompt:
      raise SystemExit("No se recibió ningún ejercicio.")

    print("\nProcesando con Qwen...", flush=True)

    try:
        raw = ask(prompt)
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        return

    print("Respuesta recibida.\n", flush=True)

    # Fallback por si algún modelo ignora format=json
    # y aun así mete un bloque Markdown.
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        print("Respuesta inválida del modelo:")
        print(raw)
        print(f"\nError JSON: {exc}")
        return

    if not isinstance(spec, dict):
        print("Respuesta inválida: se esperaba un objeto JSON.")
        print(spec)
        return

    if "error" in spec:
        print("No se ejecutó:", spec["error"])
        return

    problem_path = BASE / "problema.json"

    problem_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "\nInterpretación:\n",
        json.dumps(spec, ensure_ascii=False, indent=2),
    )

    print("\n--- Ejecución ---\n")

    process = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(BASE / "runner.py"),
            str(problem_path),
        ],
        cwd=BASE,
    )

    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()