# -*- coding: utf-8 -*-
"""
Lab 01 · Ejercicio 3 — Parámetros críticos y Embeddings (la base de RAG)

1. Determinismo absoluto con `seed`.
2. El impacto de `num_ctx` (ventana de contexto) en la memoria.
3. Qué es un Vector / Embedding y cómo se genera con Ollama.
"""

import sys
from ollama import Client

sys.stdout.reconfigure(encoding="utf-8")
client = Client(host="http://127.0.0.1:11434")
MODELO_CHAT = "qwen2.5:7b"
MODELO_EMBED = "nomic-embed-text"


def titulo(texto: str):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


# --- PASO 1: Determinismo total con seed fija -------------------------------
titulo("PASO 1 · Determinismo absoluto con seed fija")
prompt = [{"role": "user", "content": "Genera 3 palabras aleatorias en español separadas por comas."}]

for i in range(2):
    r = client.chat(
        model=MODELO_CHAT,
        messages=prompt,
        options={"temperature": 0.0, "seed": 42}
    )
    print(f"Intento {i + 1} (seed=42): {r.message.content.strip()}")


# --- PASO 2: El contexto (num_ctx) ------------------------------------------
titulo("PASO 2 · Contexto (num_ctx)")
print(
    "El contexto determina cuántos tokens de lectura + respuesta caben a la vez.\n"
    "- Default en Ollama: suele ser 2048 o 4096.\n"
    "- Para un Agente que lee proyectos o RAG: 8192 o 16384 es lo ideal.\n"
    "- Qwen 2.5 7B soporta técnicamente hasta 32k o 128k, pero en 8GB VRAM\n"
    "  un num_ctx muy grande desbordará a la RAM del sistema (haciéndolo muy lento)."
)

# Probamos una llamada con 8k tokens de contexto explícito
r_ctx = client.chat(
    model=MODELO_CHAT,
    messages=[{"role": "user", "content": "¿Cuál es la capital de Andalucía? Responde en 1 palabra."}],
    options={"num_ctx": 8192}
)
print(f"Respuesta con num_ctx=8192: {r_ctx.message.content.strip()}")


# --- PASO 3: Embeddings (Vectorización) --------------------------------------
titulo("PASO 3 · Embeddings: convirtiendo texto en significado matemático")
print(f"Verificando si tienes el modelo de embeddings '{MODELO_EMBED}'...")

modelos_instalados = [m.model for m in client.list().models]
if not any(MODELO_EMBED in m for m in modelos_instalados):
    print(f"[*] Modelo '{MODELO_EMBED}' no encontrado. Descargando (~274 MB)...")
    client.pull(MODELO_EMBED)
    print(f"[✔] Descarga completada.")

frases = [
    "El perro corre por el parque.",
    "Un canino juega en el jardín.",
    "Instalación de paquetes en Windows usando PowerShell."
]

print("\nGenerando vectores para 3 frases:")
vectores = []
for f in frases:
    res = client.embeddings(model=MODELO_EMBED, prompt=f)
    vector = res.embedding
    vectores.append(vector)
    print(f"- '{f}' -> Vector de {len(vector)} dimensiones. Primeros 5 valores: {vector[:5]}")


# --- PASO 4: Similitud Coseno básica ----------------------------------------
titulo("PASO 4 · Comparando significados (Similitud Coseno)")

def similitud_coseno(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = sum(a * a for a in v1) ** 0.5
    norm_b = sum(b * b for b in v2) ** 0.5
    return dot / (norm_a * norm_b)

sim_1_2 = similitud_coseno(vectores[0], vectores[1])
sim_1_3 = similitud_coseno(vectores[0], vectores[2])

print(f"Similitud entre 'perro en parque' y 'canino en jardín': {sim_1_2:.4f}  (¡Muy alta!)")
print(f"Similitud entre 'perro en parque' y 'PowerShell':        {sim_1_3:.4f}  (Baja)")

print("\n💡 Conclusión: ¡Así es como un RAG encuentra tus notas y código sin hacer búsquedas de palabras exactas!")
