#!/usr/bin/env python3
"""
Script para resumir y analizar transcripciones usando Ollama en local.
Lee un archivo .txt generado por el transcriptor (o cualquier texto)
y genera un informe estructurado en formato Markdown (.md).
"""

import sys

# Asegurar codificación UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import time
import subprocess
import argparse
import requests
from pathlib import Path

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"

def ensure_ollama_running():
    """Verifica si Ollama está corriendo y si no, lo inicia automáticamente en segundo plano"""
    try:
        if requests.get(OLLAMA_TAGS_URL, timeout=1).status_code == 200:
            return True
    except Exception:
        pass
        
    ollama_path = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if not ollama_path.exists():
        import shutil
        which_path = shutil.which("ollama")
        if which_path:
            ollama_path = Path(which_path)
            
    if ollama_path.exists():
        print("[*] Iniciando servicio Ollama en segundo plano...")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen([str(ollama_path), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        for _ in range(12):
            time.sleep(1)
            try:
                if requests.get(OLLAMA_TAGS_URL, timeout=1).status_code == 200:
                    print("[✔] Servicio Ollama iniciado y listo.")
                    return True
            except Exception:
                pass
    return False

def get_installed_models():
    """Obtiene la lista de modelos instalados en Ollama"""
    ensure_ollama_running()
    try:
        res = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if res.status_code == 200:
            return [m["name"] for m in res.json().get("models", [])]
    except Exception:
        pass
    return []

def summarize_text(text: str, model_name: str, language: str = "es") -> str:
    """Envía el texto a Ollama con streaming en consola"""
    system_prompt = (
        "Eres un asistente analítico experto en sintetizar reuniones, conferencias y videos. "
        "A partir de la transcripción proporcionada, genera un resumen profesional, claro y bien estructurado en Markdown. "
        "Estructura esperada:\n"
        "1. 📌 **Resumen Ejecutivo** (breve síntesis de 2-3 párrafos)\n"
        "2. 🔑 **Puntos Clave / Temas Tratados** (viñetas con los aspectos más importantes)\n"
        "3. 💡 **Conclusiones o Próximos Pasos / Acciones** (si aplica)\n"
        "Mantén un tono neutral, profesional y redacta en español."
    )

    user_prompt = f"Aquí está la transcripción a analizar:\n\n---\n{text}\n---\n\nPor favor, genera el resumen estructurado:"

    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "num_ctx": 8192
        }
    }

    print(f"[*] Conectando con Ollama usando modelo '{model_name}'...\n")
    response_parts = []
    
    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    import json
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    response_parts.append(token)
                    if chunk.get("done", False):
                        break
        print("\n")
        return "".join(response_parts)
    except requests.exceptions.ConnectionError:
        print("\n[!] Error: No se puede conectar a Ollama en http://127.0.0.1:11434.")
        print("    Asegúrate de que Ollama está en ejecución.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Resume transcripciones de texto usando modelos locales de Ollama")
    parser.add_argument(
        "input",
        help="Ruta al archivo de texto (.txt) de la transcripción o carpeta de transcripciones"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo de Ollama a usar (por defecto usa el mejor disponible, ej. qwen2.5:7b, llama3.1:8b o qwen2.5:1.5b)"
    )
    parser.add_argument(
        "--output",
        default="data/output",
        help="Directorio donde guardar el resumen generado en .md (por defecto: 'data/output/')"
    )

    args = parser.parse_args()
    
    available_models = get_installed_models()
    if not available_models:
        print("[!] No se detectaron modelos instalados en Ollama o el servicio no está corriendo.")
        print("    Ejecuta 'ollama pull qwen2.5:7b' o inicia Ollama.")
        sys.exit(1)

    # Seleccionar modelo
    selected_model = args.model
    if not selected_model:
        # Priorizar modelos de mayor a menor capacidad recomendados para 8GB
        preferred = ["qwen2.5:7b", "llama3.1:8b", "mistral:latest", "qwen2.5:1.5b", "qwen2.5:0.5b"]
        for pref in preferred:
            for avail in available_models:
                if avail.startswith(pref):
                    selected_model = avail
                    break
            if selected_model:
                break
        if not selected_model:
            selected_model = available_models[0]

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Error: El archivo o carpeta '{input_path}' no existe.")
        sys.exit(1)

    txt_files = []
    if input_path.is_file():
        txt_files.append(input_path)
    elif input_path.is_dir():
        txt_files = list(input_path.glob("*.txt"))
        if not txt_files:
            print(f"[!] No se encontraron archivos .txt en '{input_path}'.")
            return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in txt_files:
        print(f"==================================================")
        print(f"  📝 Resumiendo: {txt_file.name}")
        print(f"  Modelo:       {selected_model}")
        print(f"==================================================")
        
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            print(f"[!] El archivo '{txt_file.name}' está vacío. Omitiendo.")
            continue
            
        summary = summarize_text(content, selected_model)
        
        out_file = output_dir / f"{txt_file.stem}_resumen.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"# Resumen de {txt_file.name}\n\n")
            f.write(f"*Generado con modelo local: `{selected_model}`*\n\n")
            f.write(summary)
            
        print(f"[✔] Resumen guardado en: {out_file}\n")

if __name__ == "__main__":
    main()
