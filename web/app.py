# -*- coding: utf-8 -*-
"""
Panel de Control y Chat del Agente Personal Local
Interficie web interactiva con Streamlit:
- 💬 Chat con streaming y selección de modelo (Ollama / Gemini).
- 📚 Backoffice de Fuentes y RAG (estado de ChromaDB, reindexación, prueba de búsqueda).
- ⚙️ Backoffice de Modelos y APIs (estado de Ollama, claves en .env).
- 📊 Monitor de Hardware (CPU, RAM, GPU RTX 4060 VRAM en tiempo real).
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import psutil
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from ollama import Client as OllamaClient
import litellm

# Cargar variables de entorno
load_dotenv()

# Configuración de LiteLLM
litellm.telemetry = False
litellm.suppress_debug_info = True

CARPETA_RAIZ = Path(r"C:\desarrollo")
CARPETA_CHROMA = Path("data/chroma_db")
client_ollama = OllamaClient(host="http://127.0.0.1:11434")


# --- CLASE EMBEDDINGS LOCAL ---
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            res = client_ollama.embeddings(model="nomic-embed-text", prompt=doc)
            embeddings.append(res.embedding)
        return embeddings


def get_chroma_collection():
    try:
        chroma_client = chromadb.PersistentClient(path=str(CARPETA_CHROMA))
        return chroma_client.get_or_create_collection(
            name="proyectos_reales",
            embedding_function=OllamaEmbeddingFunction()
        )
    except Exception as e:
        st.error(f"Error conectando a ChromaDB: {e}")
        return None


def get_gpu_info():
    """Obtiene información de la GPU vía nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        name, total, used, free, temp, util = [x.strip() for x in out.split(",")]
        return {
            "name": name,
            "total_mb": float(total),
            "used_mb": float(used),
            "free_mb": float(free),
            "temp_c": float(temp),
            "util_percent": float(util)
        }
    except Exception:
        return None


# --- CONFIGURACIÓN DE PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="IA Local — Centro de Control",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BARRA LATERAL (ESTADO GENERAL) ---
with st.sidebar:
    st.title("🧠 Agente Local")
    st.caption("Estación de IA Personal · Windows 11")

    st.divider()

    # Estado de Ollama
    try:
        modelos_ollama = [m.model for m in client_ollama.list().models]
        st.success(f"🟢 **Ollama Activo** ({len(modelos_ollama)} modelos)")
    except Exception:
        st.error("🔴 **Ollama Desconectado** (ejecuta `ollama serve`)")
        modelos_ollama = []

    # Estado de Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        st.success("🟢 **Google Gemini Activo** (Free Tier)")
    else:
        st.warning("🟡 **Gemini sin clave en .env**")

    st.divider()

    # Monitor Rápido de GPU
    gpu = get_gpu_info()
    if gpu:
        vram_pct = (gpu["used_mb"] / gpu["total_mb"]) * 100
        st.metric(
            label=f"🎮 {gpu['name']}",
            value=f"{gpu['used_mb']:.0f} / {gpu['total_mb']:.0f} MB",
            delta=f"{gpu['temp_c']} °C"
        )
        st.progress(vram_pct / 100.0)
    else:
        st.info("GPU no detectada vía nvidia-smi")

    st.divider()
    st.caption("Proyecto `ia-local` · Versionado en GitHub")


# --- PESTAÑAS PRINCIPALES DEL BACKOFFICE ---
tab_chat, tab_rag, tab_modelos, tab_hardware = st.tabs([
    "💬 Chat con el Agente",
    "📚 Backoffice RAG (Fuentes)",
    "⚙️ Configuración & Modelos",
    "📊 Hardware & Recursos"
])


# ==============================================================================
# PESTAÑA 1: CHAT CON EL AGENTE
# ==============================================================================
with tab_chat:
    col_izq, col_der = st.columns([3, 1])

    with col_der:
        st.subheader("Configuración")
        # Selector de modelo
        opciones_modelo = ["ollama/agente-local", "gemini/gemini-3.5-flash", "ollama/qwen2.5:1.5b"]
        modelo_seleccionado = st.selectbox("Modelo a utilizar:", opciones_modelo, index=0)

        # Toggle de RAG
        usar_rag = st.toggle("🔍 Activar RAG de Proyectos", value=True)
        num_docs_rag = st.slider("Fragmentos a recuperar:", min_value=1, max_value=6, value=4) if usar_rag else 0

        if st.button("🗑️ Limpiar Conversación", use_container_width=True):
            st.session_state["mensajes"] = []
            st.rerun()

    with col_izq:
        # Inicializar historial de chat
        if "mensajes" not in st.session_state:
            st.session_state["mensajes"] = [
                {
                    "role": "assistant",
                    "content": "¡Hola, Pablo! Soy tu Asistente Personal de IA. Tengo acceso a tus proyectos en local y a modelos en la nube. ¿En qué te ayudo hoy?"
                }
            ]

        # Mostrar mensajes previos
        for msg in st.session_state["mensajes"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "fuentes" in msg and msg["fuentes"]:
                    with st.expander("📎 Fuentes consultadas"):
                        for f in msg["fuentes"]:
                            st.write(f"- `{f}`")

        # Entrada de nuevo mensaje
        if prompt_usuario := st.chat_input("Escribe tu pregunta para el agente..."):
            st.session_state["mensajes"].append({"role": "user", "content": prompt_usuario})
            with st.chat_message("user"):
                st.markdown(prompt_usuario)

            # Preparar contexto si RAG está activo
            fuentes_usadas = []
            contexto_extra = ""

            if usar_rag:
                coleccion = get_chroma_collection()
                if coleccion and coleccion.count() > 0:
                    try:
                        res_rag = coleccion.query(query_texts=[prompt_usuario], n_results=num_docs_rag)
                        docs = res_rag["documents"][0]
                        metas = res_rag["metadatos" if "metadatos" in res_rag else "metadatas"][0]
                        fuentes_usadas = list(dict.fromkeys([f"[{m['proyecto']}] {m['ruta_relativa']}" for m in metas]))
                        contexto_extra = "\n\n---\n\n".join(docs)
                    except Exception as e:
                        st.warning(f"Aviso de RAG: {e}")

            # Construir prompt
            system_prompt = (
                "Eres el Asistente Técnico Personal de Pablo, ejecutándote en su ordenador local.\n"
                "Sé claro, técnico, directo y responde en español.\n"
            )
            if contexto_extra:
                system_prompt += (
                    "\nUtiliza la siguiente información recuperada de sus proyectos locales para responder con precisión:\n"
                    f"--- CONTEXTO ---\n{contexto_extra}\n----------------"
                )

            mensajes_api = [{"role": "system", "content": system_prompt}]
            # Pasar últimos 6 mensajes de historial para mantener contexto conversacional
            for m in st.session_state["mensajes"][-6:]:
                mensajes_api.append({"role": m["role"], "content": m["content"]})

            # Generar respuesta con streaming
            with st.chat_message("assistant"):
                placeholder = st.empty()
                texto_acumulado = ""

                try:
                    respuesta_stream = litellm.completion(
                        model=modelo_seleccionado,
                        messages=mensajes_api,
                        stream=True,
                        timeout=30
                    )

                    for chunk in respuesta_stream:
                        token = chunk.choices[0].delta.content or ""
                        texto_acumulado += token
                        placeholder.markdown(texto_acumulado + "▌")

                    placeholder.markdown(texto_acumulado)

                    if fuentes_usadas:
                        with st.expander("📎 Fuentes consultadas"):
                            for f in fuentes_usadas:
                                st.write(f"- `{f}`")

                    # Guardar en sesión
                    msg_guardar = {"role": "assistant", "content": texto_acumulado}
                    if fuentes_usadas:
                        msg_guardar["fuentes"] = fuentes_usadas
                    st.session_state["mensajes"].append(msg_guardar)

                except Exception as e:
                    placeholder.error(f"Error generando respuesta con {modelo_seleccionado}: {e}")


# ==============================================================================
# PESTAÑA 2: BACKOFFICE RAG (GESTIÓN DE FUENTES)
# ==============================================================================
with tab_rag:
    st.header("📚 Backoffice de Fuentes y Conocimiento")
    st.write("Gestiona la base de datos vectorial local (ChromaDB) que alimenta a tu agente.")

    coleccion = get_chroma_collection()
    total_chunks = coleccion.count() if coleccion else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Colección Activa", "proyectos_reales")
    col2.metric("Fragmentos Indexados", total_chunks)
    col3.metric("Modelo de Embeddings", "nomic-embed-text (Local)")

    st.divider()

    st.subheader("Proyectos vinculados en `C:\\desarrollo`")
    st.write("Actualmente se sincronizan la documentación y archivos `.md` de:")
    st.code("hoth\nopotests\nmonitoriza-ws\nia-local")

    if st.button("🔄 Lanzar Reindexación Completa de Proyectos", type="primary"):
        with st.spinner("Escaneando proyectos y generando embeddings locales en la GPU..."):
            try:
                # Importar y ejecutar función de reindexado
                import importlib.util
                spec = importlib.util.spec_from_file_location("idx", "labs/03_rag/02_indexador_proyectos.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.indexar_proyectos(coleccion)
                st.success(f"¡Reindexación completada! Total fragmentos: {coleccion.count()}")
                st.rerun()
            except Exception as e:
                st.error(f"Error durante el reindexado: {e}")

    st.divider()

    st.subheader("🔍 Probador de Búsqueda Vectorial (Sandbox)")
    st.write("Prueba qué fragmentos exactos recupera ChromaDB ante una consulta:")
    query_prueba = st.text_input("Texto de búsqueda:", value="política de backups en Hoth")
    if query_prueba and coleccion:
        res = coleccion.query(query_texts=[query_prueba], n_results=3)
        docs = res["documents"][0]
        metas = res["metadatos" if "metadatos" in res else "metadatas"][0]

        for i, (d, m) in enumerate(zip(docs, metas)):
            with st.expander(f"Resultado #{i+1} — [{m.get('proyecto', '')}] {m.get('ruta_relativa', '')}"):
                st.text(d)


# ==============================================================================
# PESTAÑA 3: CONFIGURACIÓN Y MODELOS
# ==============================================================================
with tab_modelos:
    st.header("⚙️ Configuración del Sistema de Modelos")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Ollama (Modelos Locales)")
        if modelos_ollama:
            st.dataframe(
                [{"Modelo": m} for m in modelos_ollama],
                use_container_width=True
            )
        else:
            st.info("Sin conexión con Ollama en el puerto 11434.")

    with col_b:
        st.subheader("Proveedores en la Nube")
        st.text_input("Google AI Studio (GEMINI_API_KEY):", value="Configurada en .env ✅" if gemini_key else "No configurada ❌", disabled=True)
        st.text_input("OpenAI (OPENAI_API_KEY):", value="Configurada en .env ✅" if os.getenv("OPENAI_API_KEY") else "No configurada ❌", disabled=True)

    st.divider()

    st.subheader("Configuración Central de LiteLLM (`config/litellm.yaml`)")
    yaml_path = Path("config/litellm.yaml")
    if yaml_path.exists():
        st.code(yaml_path.read_text(encoding="utf-8"), language="yaml")


# ==============================================================================
# PESTAÑA 4: HARDWARE Y RECURSOS
# ==============================================================================
with tab_hardware:
    st.header("📊 Monitor de Hardware y Rendimiento")

    cpu_pct = psutil.cpu_percent(interval=0.2)
    ram = psutil.virtual_memory()
    disco = psutil.disk_usage("C:")

    col_hw1, col_hw2, col_hw3 = st.columns(3)
    col_hw1.metric("Uso de CPU", f"{cpu_pct}%")
    col_hw2.metric("Memoria RAM", f"{ram.used / (1024**3):.1f} / {ram.total / (1024**3):.1f} GB", f"{ram.percent}%")
    col_hw3.metric("Disco C:", f"{disco.free / (1024**3):.1f} GB libres", f"{(disco.used/disco.total)*100:.1f}% usado")

    st.divider()

    st.subheader("🎮 NVIDIA GeForce RTX 4060 (8 GB)")
    if gpu:
        gcol1, gcol2, gcol3 = st.columns(3)
        gcol1.metric("VRAM Usada", f"{gpu['used_mb']:.0f} MB", f"de {gpu['total_mb']:.0f} MB")
        gcol2.metric("VRAM Libre", f"{gpu['free_mb']:.0f} MB")
        gcol3.metric("Temperatura", f"{gpu['temp_c']} °C")
        st.progress((gpu["used_mb"] / gpu["total_mb"]))
    else:
        st.warning("No se pudo obtener información en tiempo real de la GPU con `nvidia-smi`.")
