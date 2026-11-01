# Lab 01 · Resultados en este equipo (RTX 4060 8 GB)

| Modelo | Generación (`eval rate`) | Lectura prompt | VRAM aprox. | Calidad (es) | Uso recomendado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen2.5:7b` | **~52 tok/s** | ~117 tok/s* | ~5 GB | Buena, redacción correcta | Chat, resúmenes, agente |
| `qwen2.5:1.5b` | **~174 tok/s** | ~142 tok/s* | ~1,5 GB | Floja (errores gramaticales, vaga) | Clasificar, enrutar, tareas simples |

\* Con prompts cortos; con prompts largos la lectura es mucho más rápida.

## Lecciones aprendidas
- Los comandos `/` del chat van **solos en su línea**.
- `ollama run` necesita el *tag* exacto del registro (`qwen2.5:1b` no existe → error de manifest).
- Windows + apps ocupan ~2,4 GB de VRAM → quedan ~800 MB libres con el 7b cargado.
- **El modelo no tiene memoria**: el chat reenvía todo el historial en cada mensaje (`prompt eval count` crece; `prompt eval cached` indica lo que se reaprovecha).
- `temperature 0` es determinista **solo si la entrada es idéntica**. Con historial distinto, la respuesta cambia. Usar `/clear` para comprobarlo.
