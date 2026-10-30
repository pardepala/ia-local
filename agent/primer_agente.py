# -*- coding: utf-8 -*-
"""
Primer Agente Local Autónomo con Ollama (Qwen 2.5 7B)
Demuestra el ciclo de Function Calling:
1. El usuario hace una pregunta que el modelo NO sabe de memoria (ej. datos de su propio ordenador).
2. El modelo detecta que necesita una herramienta y emite una llamada a la función en Python.
3. Python ejecuta la herramienta en tu máquina y le devuelve el resultado real.
4. El modelo sintetiza la respuesta final con los datos reales obtenidos.
"""

import sys
from pathlib import Path

import psutil
from ollama import Client

# Asegurar codificación UTF-8 en consola de Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

client = Client(host='http://127.0.0.1:11434')
MODELO = 'qwen2.5:7b'

# --- 1. DEFINIMOS NUESTRAS HERRAMIENTAS (TOOLS EN PYTHON) ---

def obtener_estado_sistema() -> str:
    """Devuelve el uso actual de CPU, RAM y espacio en disco del ordenador local."""
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disco = psutil.disk_usage('C:')
    return (
        f"CPU al {cpu}% | "
        f"RAM usada: {ram.used / (1024**3):.1f} GB de {ram.total / (1024**3):.1f} GB ({ram.percent}%) | "
        f"Disco C: Libre {disco.free / (1024**3):.1f} GB de {disco.total / (1024**3):.1f} GB"
    )

def listar_directorio(ruta: str) -> str:
    """Lista las carpetas y archivos que contiene un directorio del ordenador local.

    Args:
        ruta: Ruta absoluta del directorio a listar, por ejemplo C:\\desarrollo
    """
    p = Path(ruta)
    if not p.is_dir():
        return f"La ruta '{ruta}' no existe o no es un directorio."
    entradas = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    lineas = [f"[DIR]  {e.name}" if e.is_dir() else f"[FILE] {e.name}" for e in entradas[:100]]
    return "\n".join(lineas) or "(directorio vacío)"

# Mapeo para que el Agente sepa ejecutar la función correspondiente
HERRAMIENTAS_MAP = {
    'obtener_estado_sistema': obtener_estado_sistema,
    'listar_directorio': listar_directorio,
}

# --- 2. BUCLE DEL AGENTE ---

def ejecutar_agente(pregunta_usuario: str):
    print("\n" + "="*70)
    print(f"👤 USUARIO: {pregunta_usuario}")
    print("="*70)

    mensajes = [
        {
            'role': 'system',
            'content': (
                'Eres un Asistente Agéntico de Inteligencia Artificial ejecutándose en local en el equipo de Pablo. '
                'Tienes herramientas para consultar el estado del hardware y explorar carpetas del ordenador. '
                'Si el usuario te pregunta por su ordenador o sus archivos, USA SIEMPRE tus herramientas. '
                'Sé conciso, riguroso y responde en español.'
            )
        },
        {'role': 'user', 'content': pregunta_usuario}
    ]

    # Primera llamada al modelo pasándole las herramientas disponibles
    print("🤖 Agente pensando...")
    respuesta = client.chat(
        model=MODELO,
        messages=mensajes,
        tools=list(HERRAMIENTAS_MAP.values())
    )

    # Comprobamos si el modelo decidió hablar directamente o si solicitó usar una herramienta
    if respuesta.message.tool_calls:
        mensajes.append(respuesta.message)
        for tool_call in respuesta.message.tool_calls:
            nombre_funcion = tool_call.function.name
            argumentos = tool_call.function.arguments or {}

            print(f"\n⚙️  [DECISIÓN DEL AGENTE]: Ejecutar herramienta -> `{nombre_funcion}({argumentos})`")

            # Ejecutamos la función de Python en tu PC
            funcion_a_ejecutar = HERRAMIENTAS_MAP.get(nombre_funcion)
            if funcion_a_ejecutar:
                resultado = funcion_a_ejecutar(**argumentos)
                print(f"📥 [RESULTADO DE PYTHON]: {resultado}")
                mensajes.append({'role': 'tool', 'content': str(resultado), 'tool_name': nombre_funcion})
            else:
                print(f"❌ Función no encontrada: {nombre_funcion}")

        # Segunda llamada: El modelo lee el resultado de la herramienta y redacta la respuesta final
        print("\n🤖 Agente sintetizando respuesta final...")
        respuesta_final = client.chat(model=MODELO, messages=mensajes)
        print("\n💬 RESPUESTA DEL AGENTE:")
        print(respuesta_final.message.content)
    else:
        print("\n💬 RESPUESTA DIRECTA (sin herramientas):")
        print(respuesta.message.content)

if __name__ == '__main__':
    # Test 1: Pregunta que requiere consultar el hardware de tu ordenador
    ejecutar_agente("¿Cómo va de carga de RAM y CPU mi equipo ahora mismo?")

    # Test 2: Pregunta que requiere explorar el sistema de archivos
    ejecutar_agente("¿Qué proyectos tengo en la carpeta C:\\desarrollo?")

    # Test 3: Pregunta que NO requiere herramientas (responderá con su conocimiento general)
    ejecutar_agente("En 2 líneas, ¿cuál es la diferencia entre memoria RAM y disco duro?")
