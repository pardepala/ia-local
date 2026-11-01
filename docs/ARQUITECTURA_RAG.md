# 🧠 Arquitectura RAG Local (ChromaDB + Ollama)

```mermaid
flowchart TD
    subgraph Ingesta["1. Ingesta (Tus Archivos)"]
        A["data/conocimiento/ (*.md, *.txt, código)"] --> B["Chunking & Lectura"]
        B --> C["Ollama: nomic-embed-text"]
        C -->|Vectores de 768 dims| D[("ChromaDB Local (data/chroma_db/)")]
    end

    subgraph Consulta["2. Consulta Inteligente"]
        Q["Usuario: '¿Qué puerto usa SSH en mi servidor?'"] --> VQ["nomic-embed-text"]
        VQ -->|Vector de consulta| D
        D -->|Top 1 fragmento más similar| CTX["Fragmento exacto recuperado"]
        CTX --> Prompt["System Prompt Aumentado\n(Instrucción + Contexto Real)"]
        Prompt --> Agent["agente-local (Ollama RTX 4060)"]
        Agent --> R["Respuesta precisa con cita del archivo"]
    end
```

## Ventajas de este diseño
1. **100% Privado**: Tus archivos nunca salen de tu ordenador. Los embeddings y el modelo corren en local.
2. **Cero alucinaciones**: El modelo responde basándose en el fragmento recuperado de tu disco.
3. **Persistente**: ChromaDB guarda los índices en `data/chroma_db/`. No hay que reindexar todo cada vez.
