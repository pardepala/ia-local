# Lab 01 · Ejercicio 1 — La CLI de Ollama

**Objetivo:** dominar los comandos básicos y entender qué pasa en la GPU cuando cargas un modelo.

## 1. Comprobar que el servicio está activo

```powershell
ollama --version
ollama list
```

- Si `ollama list` da error de conexión, el servicio está parado. Arráncalo abriendo la app **Ollama** desde el menú Inicio (se queda en la bandeja del sistema) o con `ollama serve` en otra terminal.
- `ollama list` muestra los modelos descargados en disco (`~\.ollama\models`).

## 2. Inspeccionar un modelo

```powershell
ollama show qwen2.5:7b
ollama show qwen2.5:7b --modelfile
```

Fíjate en:
| Campo | Qué significa |
| :--- | :--- |
| `parameters` | Tamaño del modelo (7.6B = 7.600 millones de pesos) |
| `quantization` | Compresión de los pesos (Q4_K_M = ~4,5 bits por peso) → por eso 7B cabe en ~4,7 GB |
| `context length` | Máximo de tokens que el modelo *puede* manejar |
| `capabilities` | `completion`, `tools`, `vision`, `embedding`... |

## 3. Chatear y ver la GPU

```powershell
ollama run qwen2.5:7b
```

Dentro del chat prueba los comandos internos:
| Comando | Efecto |
| :--- | :--- |
| `/?` | Ayuda |
| `/show info` | Info del modelo cargado |
| `/set verbose` | Muestra velocidad (tokens/s) tras cada respuesta |
| `/set parameter temperature 0` | Respuestas deterministas |
| `/set system "Eres un pirata"` | Cambia el prompt de sistema |
| `/clear` | Borra el historial de la conversación |
| `/bye` | Salir |

Mientras el modelo está cargado, en **otra terminal**:

```powershell
ollama ps        # modelo en memoria, % GPU/CPU y cuándo se descarga
nvidia-smi       # VRAM usada en la RTX 4060
```

> `ollama ps` debe mostrar **100% GPU**. Si aparece parte en CPU, el modelo no cabe entero en VRAM y será mucho más lento.

## 4. Descargar y liberar

```powershell
ollama stop qwen2.5:7b     # descarga el modelo de la VRAM (sigue en disco)
ollama pull llama3.2:3b    # descarga un modelo nuevo
ollama rm llama3.2:3b      # lo borra del disco
```

## ✅ Preguntas para comprobar que lo has entendido
1. ¿Cuántos tokens/s da `qwen2.5:7b` en tu equipo? ¿Y `qwen2.5:1.5b`?
2. ¿Cuánta VRAM ocupa cada uno según `nvidia-smi`?
3. ¿Qué cambia en las respuestas con `temperature 0` frente a `temperature 1.5`?
4. ¿Cuánto tarda en descargarse el modelo de la VRAM si no lo usas? (pista: `ollama ps`, columna `UNTIL`)
