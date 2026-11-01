# -*- coding: utf-8 -*-
"""
Lab 03 · Ejercicio 2 (Mejorado) — Indexador RAG de tus Proyectos Reales

Mejoras clave aplicadas:
1. Sin datos falsos de prueba (solo lee proyectos reales en C:\\desarrollo).
2. Inyección contextual en cada chunk (Proyecto + Archivo + Ruta) para búsquedas precisas.
3. Recupera los 4 mejores fragmentos (Top 4) para dar respuestas completas y contextualizadas.
"""

import os
import sys
from pathlib import Path
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from ollama import Client
import litellm

# Asegurar codificación UTF-8 en Windows
sys.stdout.reconfigure(encoding="utf-8")

client_ollama = Client(host="http://127.0.0.1:11434")
CARPETA_RAIZ = Path(r"C:\desarrollo")
CARPETA_CHROMA = Path("data/chroma_db")

# Proyectos reales a indexar
PROYECTOS_OBJETIVO = ["hoth", "opotests", "monitoriza-ws"]

# Carpetas a ignorar
IGNORAR_DIR = {".git", ".venv", "venv", "node_modules", "output", "scratch", "__pycache__", "dist", "build"}


# --- 1. EMBEDDINGS LOCALES OLLAMA -------------------------------------------
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            res = client_ollama.embeddings(model="nomic-embed-text", prompt=doc)
            embeddings.append(res.embedding)
        return embeddings


# --- 2. CHUNKING CONSOLAPADO Y ENCABEZADOS ----------------------------------
def fragmentar_documento(texto: str, max_tamano: int = 1400, solape: int = 200):
    lineas = texto.split("\n")
    chunks = []
    chunk_actual = []
    tamano_actual = 0

    for linea in lineas:
        longitud = len(linea) + 1
        if tamano_actual + longitud > max_tamano and chunk_actual:
            chunks.append("\n".join(chunk_actual).strip())
            chunk_actual = chunk_actual[-4:] if len(chunk_actual) >= 4 else chunk_actual
            tamano_actual = sum(len(l) + 1 for l in chunk_actual)

        chunk_actual.append(linea)
        tamano_actual += longitud

    if chunk_actual:
        resto = "\n".join(chunk_actual).strip()
        if resto:
            chunks.append(resto)

    return chunks or [texto]


# --- 3. ESCANEO E INDEXACIÓN CON INYECCIÓN DE CONTEXTO ---------------------
def indexar_proyectos(coleccion):
    print(f"\n📂 Escaneando proyectos reales en '{CARPETA_RAIZ}'...")
    total_archivos = 0
    total_chunks = 0

    for nombre_proj in PROYECTOS_OBJETIVO:
        dir_proj = CARPETA_RAIZ / nombre_proj
        if not dir_proj.is_dir():
            continue

        print(f"\n🔍 Proyecto: [{nombre_proj}]")
        archivos_md = []

        for p in dir_proj.rglob("*.md"):
            if any(ign in p.parts for ign in IGNORAR_DIR):
                continue
            archivos_md.append(p)

        # Priorizar README.md primero
        archivos_md.sort(key=lambda x: (x.name != "README.md", str(x)))

        for archivo in archivos_md:
            try:
                contenido = archivo.read_text(encoding="utf-8", errors="ignore").strip()
                if not contenido or len(contenido) < 40:
                    continue

                chunks = fragmentar_documento(contenido)
                ids = []
                docs = []
                metas = []

                ruta_relativa = str(archivo.relative_to(CARPETA_RAIZ))

                for idx, chunk in enumerate(chunks):
                    # INYECCIÓN CLAVE: Inyectamos el nombre del proyecto y archivo en el texto
                    # para que nomic-embed-text entienda exactamente a qué pertenece cada trozo.
                    doc_con_contexto = (
                        f"PROYECTO: {nombre_proj}\n"
                        f"ARCHIVO: {archivo.name} ({ruta_relativa})\n"
                        f"---\n"
                        f"{chunk}"
                    )

                    chunk_id = f"{nombre_proj}_{archivo.stem}_{idx}"
                    ids.append(chunk_id)
                    docs.append(doc_con_contexto)
                    metas.append({
                        "proyecto": nombre_proj,
                        "archivo": archivo.name,
                        "ruta_relativa": ruta_relativa,
                        "chunk": idx
                    })

                coleccion.upsert(ids=ids, documents=docs, metadatas=metas)
                total_archivos += 1
                total_chunks += len(chunks)
                print(f"  📄 {archivo.relative_to(dir_proj)} -> {len(chunks)} fragmentos")
            except Exception as e:
                print(f"  ⚠️ Error leyendo {archivo.name}: {e}")

    print(f"\n[✔] Indexación completada: {total_archivos} archivos, {total_chunks} fragmentos indexados.")
    print(f"[✔] Total fragmentos en ChromaDB: {coleccion.count()}")


# --- 4. BÚSQUEDA Y RESPUESTA DEL AGENTE -------------------------------------
def consultar_agente(coleccion, pregunta: str):
    n_recuperar = min(4, coleccion.count())
    if n_recuperar == 0:
        print("[!] No hay documentos indexados en la base de datos.")
        return

    resultados = coleccion.query(query_texts=[pregunta], n_results=n_recuperar)

    documentos = resultados["documents"][0]
    metadatos = resultados["metadatas"][0]

    bloques_contexto = []
    fuentes_usadas = []
    for doc, meta in zip(documentos, metadatos):
        fuente = f"[{meta['proyecto']}] {meta['ruta_relativa']}"
        fuentes_usadas.append(fuente)
        bloques_contexto.append(doc)

    contexto = "\n\n" + ("=" * 40) + "\n\n".join(bloques_contexto)

    system_prompt = (
        "Eres el Asistente Técnico Personal de Pablo, con acceso a la documentación "
        "de sus proyectos reales (Hoth, OpoNexo, etc.).\n"
        "INSTRUCCIONES:\n"
        "1. Responde a la pregunta del usuario utilizando la información de los fragmentos recuperados.\n"
        "2. Sintetiza la respuesta de forma clara y directa.\n"
        "3. Cita al final de tu respuesta los archivos de los que has extraído la información.\n"
        "4. Si la respuesta no está en el contexto, indícalo con naturalidad sin inventar.\n\n"
        f"--- CONTEXTO RECUPERADO DE TUS PROYECTOS ---\n{contexto}\n--------------------------------------------"
    )

    print("\n🔍 Fragmentos recuperados:")
    for f in list(dict.fromkeys(fuentes_usadas)):
        print(f"   📎 {f}")

    print(f"\n🤖 Agente ({'agente-local'}):\n", end="", flush=True)
    respuesta = litellm.completion(
        model="ollama/agente-local",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pregunta}
        ],
        stream=True
    )

    for chunk in respuesta:
        contenido = chunk.choices[0].delta.content or ""
        print(contenido, end="", flush=True)
    print("\n")


# --- 5. MODO INTERACTIVO ----------------------------------------------------
def main():
    print("=" * 70)
    print("🧠 RAG DE PROYECTOS REALES (Mejorado con inyección de contexto)")
    print("=" * 70)

    chroma_client = chromadb.PersistentClient(path=str(CARPETA_CHROMA))
    coleccion = chroma_client.get_or_create_collection(
        name="proyectos_reales",
        embedding_function=OllamaEmbeddingFunction()
    )

    # Si la colección está vacía, indexamos automáticamente
    if coleccion.count() == 0:
        indexar_proyectos(coleccion)
    else:
        print(f"[*] Base de datos cargada ({coleccion.count()} fragmentos).")
        opcion = input("¿Deseas reindexar los proyectos para actualizar cambios? (s/N): ").strip().lower()
        if opcion in ("s", "si", "sí", "y"):
            indexar_proyectos(coleccion)

    print("\n" + "=" * 70)
    print("💬 MODO CONSULTA INTERACTIVA (Escribe 'salir' para terminar o 'reindexar' para actualizar)")
    print("=" * 70)

    while True:
        try:
            pregunta = input("\n👤 Tu pregunta: ").strip()
            if not pregunta:
                continue
            if pregunta.lower() in ("salir", "exit", "quit", "q"):
                print("👋 ¡Hasta luego!")
                break
            if pregunta.lower() == "reindexar":
                indexar_proyectos(coleccion)
                continue

            consultar_agente(coleccion, pregunta)
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break


if __name__ == "__main__":
    main()
