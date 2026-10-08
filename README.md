# Taller 2 – Sistema RAG para EcoMarket 🌱

Extensión del [Taller 1](https://github.com/Joules17/prompt-engineering-taller-1): el asistente de atención
al cliente de **EcoMarket** consulta una base de conocimiento interna (RAG) antes de responder.

- **Fase 1:** [Selección de componentes del sistema RAG](FASE-1.md)
- **Fase 2:** [Creación de la base de conocimiento](FASE-2.md)
- **Fase 3:** integración y ejecución del código (este repositorio)

El código parte del repositorio de referencia
[rag-chromadb-streamlit](https://github.com/avila196/rag-chromadb-streamlit) (idea original:
[Doc-Sage](https://dev.to/ngonidzashe/doc-sage-create-a-smart-rag-app-with-langchain-and-streamlit-4lin)),
adaptado para usar **Google Gemini** en lugar de OpenAI.

## Cambios respecto al repositorio de referencia

| Pieza | Original (OpenAI) | Este repo (Gemini) |
|---|---|---|
| Paquete | `langchain-openai` | `langchain-google-genai` |
| LLM | `ChatOpenAI(model="gpt-4o-mini")` | `ChatGoogleGenerativeAI(model=GEMINI_MODEL)` |
| Embeddings | `OpenAIEmbeddings()` | `GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")` |
| Variable de entorno | `OPENAI_API_KEY` | `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL` |
| Lectura de `.txt` / `.csv` | Codificación por defecto del SO (cp1252 en Windows: falla con tildes) | `encoding="utf-8"` |

> Si cambias el modelo de embeddings, borra la carpeta `persist/` y vuelve a cargar los documentos:
> los vectores de modelos distintos no son comparables.

## Instalación con uv

```bash
uv venv
uv pip install -r requirements.txt
```

Crea el `.env` a partir de la plantilla y agrega tu API key
(se obtiene gratis en [Google AI Studio](https://aistudio.google.com/apikey)):

```bash
copy .env.example .env        # Windows  (Linux/Mac: cp .env.example .env)
```

> `.env` está en `.gitignore`: nunca subas tu API key al repositorio.

### Si el proyecto está dentro de OneDrive

OneDrive no admite los *hardlinks* que uv usa por defecto e intenta sincronizar los miles de archivos
del `.venv`. Si ves errores como `os error 396`, crea el entorno fuera de OneDrive:

```powershell
uv venv $HOME\.venvs\rag-taller-2
$env:VIRTUAL_ENV = "$HOME\.venvs\rag-taller-2"
$env:UV_LINK_MODE = "copy"
uv pip install -r requirements.txt
& "$HOME\.venvs\rag-taller-2\Scripts\Activate.ps1"
```

## Uso

```bash
# 1. Crear las tablas de SQLite (solo la primera vez)
uv run python create_relational_db.py

# 2. Lanzar la app
uv run streamlit run chats.py
```

Si usas el entorno fuera de OneDrive, actívalo y ejecuta los comandos sin `uv run`
(`python create_relational_db.py`, `streamlit run chats.py`).

En la app: crea un chat, sube documentos (`.txt`, `.pdf`, `.docx`, `.csv`, `.html`, `.md`) o agrega
links desde la barra lateral, y haz preguntas.
