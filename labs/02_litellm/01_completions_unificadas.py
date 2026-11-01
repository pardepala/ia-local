# -*- coding: utf-8 -*-
"""
Lab 02 · Ejercicio 1 — LiteLLM: Una única API para gobernarlos a todos

Demuestra cómo hablar con:
- Tu modelo local en Ollama (agente-local o qwen2.5:7b)
- Google Gemini (si tienes GEMINI_API_KEY en .env)
- OpenAI ChatGPT (si tienes OPENAI_API_KEY en .env)
¡Usando exactamente la misma función y la misma estructura de datos!
"""

import os
import sys
from dotenv import load_dotenv
import litellm

# Cargar variables de entorno desde .env
load_dotenv()

# Asegurar codificación UTF-8 en consola
sys.stdout.reconfigure(encoding="utf-8")

# Desactivar telemetría y logs ruidosos de litellm
litellm.telemetry = False
litellm.suppress_debug_info = True


def titulo(texto: str):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


# --- 1. LLAMADA AL MODELO LOCAL A TRAVÉS DE LITELLM ------------------------
titulo("1. LLAMADA LOCAL (Ollama vía LiteLLM)")
prompt = [
    {"role": "system", "content": "Eres un asistente técnico conciso."},
    {"role": "user", "content": "Explica en 1 línea qué ventaja tiene usar un router de modelos como LiteLLM."}
]

print("🤖 Enviando petición a 'ollama/agente-local'...")
try:
    respuesta = litellm.completion(
        model="ollama/agente-local",
        api_base="http://localhost:11434",
        messages=prompt,
        stream=True
    )
    print("\nRespuesta (streaming): ", end="")
    for chunk in respuesta:
        contenido = chunk.choices[0].delta.content or ""
        print(contenido, end="", flush=True)
    print("\n")
except Exception as e:
    print(f"❌ Error al llamar a Ollama vía LiteLLM: {e}")


# --- 2. VERIFICACIÓN DE CLAVES DE LA NUBE (GEMINI / OPENAI) -----------------
titulo("2. PROVEEDORES EN LA NUBE (Gemini / OpenAI)")

gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

print(f"- GEMINI_API_KEY configurada: {'✅ SÍ' if gemini_key else '❌ NO (vacía o no definida en .env)'}")
print(f"- OPENAI_API_KEY configurada: {'✅ SÍ' if openai_key else '❌ NO (vacía o no definida en .env)'}")

if gemini_key:
    print("\n🌐 Probando llamada a Google Gemini (gemini/gemini-2.5-flash)...")
    try:
        r_gemini = litellm.completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Di 'Hola desde Gemini' en una frase."}]
        )
        print("Respuesta de Gemini:", r_gemini.choices[0].message.content)
    except Exception as e:
        print(f"❌ Error con Gemini: {e}")
else:
    print("\nℹ️  Para probar Gemini:")
    print("   1. Consigue una clave gratuita en: https://aistudio.google.com/apikey")
    print("   2. Añádela a tu archivo .env: GEMINI_API_KEY=tu_clave_aqui")

if openai_key:
    print("\n🌐 Probando llamada a OpenAI (gpt-4o-mini)...")
    try:
        r_openai = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Di 'Hola desde OpenAI' en una frase."}]
        )
        print("Respuesta de OpenAI:", r_openai.choices[0].message.content)
    except Exception as e:
        print(f"❌ Error con OpenAI: {e}")
