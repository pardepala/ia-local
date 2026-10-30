# 🧠 ia-local — Agente personal con IA local

Infraestructura de IA ejecutándose en local (Ollama + GPU) para construir, paso a paso,
un **agente personal** que entienda el ordenador, los proyectos y los datos de su dueño,
con acceso opcional a modelos en la nube (Gemini, OpenAI) y una interfaz web.

## 🖥️ Equipo

| Componente | Detalle |
| :--- | :--- |
| CPU | Intel Core Ultra 7 265K |
| RAM | 32 GB |
| GPU | NVIDIA RTX 4060 · 8 GB VRAM (CUDA) |
| SO | Windows 11 |

## 📁 Estructura

```text
ia-local/
├── agent/      # El agente: bucle, herramientas, memoria
├── tools/      # Herramientas reutilizables (transcripción, resúmenes...)
├── labs/       # Ejercicios de aprendizaje por fase (01_ollama, 02_litellm...)
├── docs/       # Apuntes y roadmap
├── config/     # Configuración (litellm.yaml, Modelfiles...)
├── data/       # Datos locales, índices RAG, salidas (ignorado por git)
├── .env.example
└── requirements.txt
```

## 🚀 Puesta en marcha

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # y rellena las claves que necesites

ollama serve                # si el servicio no está arrancado
ollama pull qwen2.5:7b
```

## 🧰 Herramientas disponibles

| Herramienta | Uso |
| :--- | :--- |
| `tools/transcribe.py` | Audio/vídeo → `.txt`, `.srt`, `.json` con faster-whisper en GPU |
| `tools/summarize.py` | Texto → resumen estructurado en Markdown con Ollama |
| `agent/primer_agente.py` | Demo de *function calling* con herramientas del sistema |

```powershell
.\.venv\Scripts\python tools\transcribe.py data\input\audio.m4a --model large-v3-turbo --language es
.\.venv\Scripts\python tools\summarize.py data\output\audio.txt
.\.venv\Scripts\python agent\primer_agente.py
```

## 🗺️ Roadmap

Consulta [docs/ROADMAP.md](docs/ROADMAP.md).
