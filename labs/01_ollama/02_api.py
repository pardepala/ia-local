# -*- coding: utf-8 -*-
"""
Lab 01 · Ejercicio 2 — Hablar con Ollama desde Python.

Lo que hace `ollama run` por dentro son llamadas HTTP a http://127.0.0.1:11434.
Aquí hacemos lo mismo con la librería oficial `ollama` en 4 pasos.

Ejecuta:  python labs/01_ollama/02_api.py
Lee el código de cada paso y modifícalo para experimentar.
"""

import sys
import time

from ollama import Client

sys.stdout.reconfigure(encoding="utf-8")
client = Client(host="http://127.0.0.1:11434")
MODELO = "qwen2.5:7b"


def titulo(texto: str):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


# --- PASO 1: una petición simple (equivale a /api/generate) -----------------
titulo("PASO 1 · generate: una pregunta, una respuesta")
r = client.generate(model=MODELO, prompt="Di 'hola' en 5 idiomas, en una línea.")
print(r.response)
# La respuesta trae también las métricas que veías con /set verbose:
velocidad = r.eval_count / (r.eval_duration / 1e9)
print(f"\n[métricas] tokens generados={r.eval_count} · {velocidad:.1f} tok/s")


# --- PASO 2: chat con roles y SIN memoria ----------------------------------
titulo("PASO 2 · chat: el modelo NO recuerda nada por sí solo")
r1 = client.chat(model=MODELO, messages=[{"role": "user", "content": "Me llamo Pablo."}])
print("Respuesta 1:", r1.message.content)
r2 = client.chat(model=MODELO, messages=[{"role": "user", "content": "¿Cómo me llamo?"}])
print("Respuesta 2:", r2.message.content, "  <-- no lo sabe")


# --- PASO 3: la "memoria" es reenviar el historial --------------------------
titulo("PASO 3 · chat con historial: así funciona la memoria de un chat")
historial = [
    {"role": "system", "content": "Eres un asistente conciso. Responde en español."},
    {"role": "user", "content": "Me llamo Pablo."},
]
r = client.chat(model=MODELO, messages=historial)
historial.append({"role": "assistant", "content": r.message.content})
historial.append({"role": "user", "content": "¿Cómo me llamo?"})
r = client.chat(model=MODELO, messages=historial)
print("Respuesta:", r.message.content)
print(f"[métricas] tokens de entrada enviados: {r.prompt_eval_count}")


# --- PASO 4: streaming + parámetros -----------------------------------------
titulo("PASO 4 · streaming y temperature (0 = determinista, 1.5 = creativo)")
pregunta = [{"role": "user", "content": "Inventa un nombre para un asistente de IA personal. Solo el nombre."}]
for temp in (0, 0, 1.5, 1.5):
    print(f"temperature={temp}: ", end="", flush=True)
    t0 = time.time()
    for trozo in client.chat(model=MODELO, messages=pregunta, stream=True,
                             options={"temperature": temp, "num_ctx": 4096}):
        print(trozo.message.content, end="", flush=True)
    print(f"   ({time.time() - t0:.2f}s)")

print("\n✅ Fin. Retos: cambia MODELO a 'qwen2.5:1.5b', prueba otros 'options' "
      "(top_p, seed, num_predict) o convierte el PASO 3 en un bucle con input().")
