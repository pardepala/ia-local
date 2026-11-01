# -*- coding: utf-8 -*-
"""
Lab 03 · Ejercicio 1 — Tu primer RAG Local con ChromaDB + Ollama

1. Indexa los archivos markdown de 'data/conocimiento/' en una base vectorial local.
2. Usa 'nomic-embed-text' (Ollama) para vectorizar sin enviar nada a internet.
3. Permite hacer una pregunta: busca los fragmentos relevantes y se los pasa a
   'agente-local' para que responda con la verdad absoluta de tus archivos.
"""

import os
import sys
from pathlib import Path
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from ollama import Client
import litellm

# Asegurar UTF-8 en consola
sys.stdout.reconfigure(encoding="utf-8")

client_ollama = Client(host="http://127.0.0.1:11434")
CARPETA_NOTAS = Path("data/conocimiento")
CARPETA_CHROMA = Path("data/chroma_db")


# --- 1. FUNCIÓN DE EMBEDDINGS PERSONALIZADA USANDO OLLAMA LOCAL --------------
class OllamaEmbeddingFunction(EmbeddingFunction):
    """Genera embeddings localmente usando nomic-embed-text en Ollama."""
    def __init__(self):
        super().__init__()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            res = client_ollama.embeddings(model="nomic-embed-text", prompt=doc)
            embeddings.append(res.embedding)
        return embeddings


# --- 2. INICIALIZAR BASE DE DATOS VECTORIAL LOCAL ---------------------------
print(f"[*] Conectando con ChromaDB en '{CARPETA_CHROMA}'...")
chroma_client = chromadb.PersistentClient(path=str(CARPETA_CHROMA))

# Colección para tus proyectos y notas
coleccion = chroma_client.get_or_create_collection(
    name="mis_proyectos",
    embedding_function=OllamaEmbeddingFunction()
)


# --- 3. INGESTA DE DOCUMENTOS LOCALES ---------------------------------------
def indexar_documentos():
    print(f"\n[*] Escaneando documentos en '{CARPETA_NOTAS}'...")
    archivos = list(CARPETA_NOTAS.glob("*.md")) + list(CARPETA_NOTAS.glob("*.txt"))
    if not archivos:
        print(f"[!] No se encontraron documentos en {CARPETA_NOTAS}")
        return

    ids = []
    documentos = []
    metadatos = []

    for archivo in archivos:
        contenido = archivo.read_text(encoding="utf-8").strip()
        if not contenido:
            continue
        ids.append(archivo.name)
        documentos.append(contenido)
        metadatos.append({"origen": str(archivo)})
        print(f"  📄 Indexando: {archivo.name} ({len(contenido)} caracteres)")

    # Añadimos (o actualizamos si ya existen)
    coleccion.upsert(ids=ids, documents=documentos, metadatas=metadatos)
    print(f"[✔] Total documentos en la base de datos: {coleccion.count()}")


# --- 4. CONSULTA AL RAG (Búsqueda + Respuesta del Agente) -------------------
def preguntar_al_rag(pregunta: str):
    print(f"\n{'=' * 70}")
    print(f"❓ PREGUNTA: {pregunta}")
    print(f"{'=' * 70}")

    # 1. Recuperar los fragmentos más relevantes de ChromaDB
    resultados = coleccion.query(query_texts=[pregunta], n_results=1)

    fragmentos_recuperados = resultados["documents"][0]
    fuentes = resultados["metadatas"][0]

    contexto = "\n\n---\n\n".join(fragmentos_recuperados)
    origen = fuentes[0]["origen"]
    print(f"🔍 [CHROMA]: Encontrado fragmento relevante en: {origen}\n")

    # 2. Construir el prompt aumentado (RAG)
    system_prompt = (
        "Eres el Asistente Personal de Pablo. "
        "Responde a la pregunta del usuario utilizando EXCLUSIVAMENTE la información "
        "del siguiente contexto recuperado de sus archivos locales. "
        "Si la respuesta no está en el contexto, di claramente que no dispones de esa información.\n\n"
        f"--- CONTEXTO DE SUS ARCHIVOS ---\n{contexto}\n-------------------------------"
    )

    mensajes = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": pregunta}
    ]

    # 3. Generar la respuesta con el modelo local a través de LiteLLM
    print("🤖 Agente local respondiendo con datos de tus archivos:")
    respuesta = litellm.completion(
        model="ollama/agente-local",
        messages=mensajes,
        stream=True
    )

    for chunk in respuesta:
        contenido = chunk.choices[0].delta.content or ""
        print(contenido, end="", flush=True)
    print("\n")


# --- 5. EJECUCIÓN -----------------------------------------------------------
if __name__ == "__main__":
    # Indexamos las notas
    indexar_documentos()

    # Probamos una pregunta sobre el servidor
    preguntar_al_rag("¿Qué puerto usa el servicio SSH de mi servidor y qué contenedores Docker hay?")

    # Probamos una pregunta sobre el proyecto Hoth
    preguntar_al_rag("¿Cómo se llama la variable de entorno para las alertas de Telegram en el proyecto Hoth?")
