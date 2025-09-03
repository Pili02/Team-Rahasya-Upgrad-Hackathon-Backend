# 🧠 AI MindMap Mentor

An intelligent backend system that converts vague goal descriptions into dynamic, hierarchical mindmaps. Unlike traditional linear roadmaps, this system creates interconnected concept trees with explanations, time estimates, and RAG-powered resources.

## 🚀 Features

- **Mindmap Generation**: Builds tree structures instead of flat lists
- **RAG-powered Resources**: Real, verified educational resources via vector database
- **Time Estimation**: Granular time estimates for each learning node
- **Explainability**: Query any node to understand why it exists
- **Interoperability**: Export as JSON, linear roadmap, or graph visualization

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **LLM**: LLaMA 3 via Ollama (local inference)
- **RAG**: LangChain + ChromaDB
- **Schema**: Pydantic models
- **Deployment**: Docker ready

## 📋 Prerequisites

- Python 3.8+
- Ollama installed and running locally
- `ollama run llama3` command available

## 🚀 Quick Start

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama with LLaMA 3**:

   ```bash
   ollama run llama3
   ```

3. **Run the application**:

   ```bash
   uvicorn app.main:app --reload
   ```

4. **Access the API**:
   - API docs: http://localhost:8000/docs
   - Generate mindmap: POST http://localhost:8000/generate_mindmap

## 📚 API Endpoints

### POST /generate_mindmap

Generate a mindmap from a goal description.

**Input**:

```json
{
  "description": "I want to learn AI in 6 months"
}
```

**Output**: Structured mindmap JSON with nodes, descriptions, time estimates, and resources.

### GET /explain_node/{id}

Get explanation for why a specific node exists in the mindmap.

### GET /linearize

Convert the mindmap to a flattened, linear roadmap format.

## 🏗️ Project Structure

```
ai-mindmap-mentor/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── models.py            # Pydantic schemas
│   ├── routes.py            # API endpoints
│   ├── services/            # Core business logic
│   └── utils/               # Helper functions
├── data/                    # Knowledge base for RAG
├── requirements.txt
└── README.md
```

## 🎯 Example Output

The system generates hierarchical mindmaps like:

```json
{
  "root": "Learn Artificial Intelligence",
  "nodes": [
    {
      "id": 1,
      "title": "Programming Foundations",
      "description": "Learn Python and programming basics",
      "time_left": "3 weeks",
      "resources": ["https://docs.python.org/3/tutorial/"],
      "children": [...]
    }
  ]
}
```

## 🔧 Development

- **Run tests**: `python -m pytest`
- **Format code**: `black .`
- **Lint code**: `flake8`

## 📝 License

MIT License - feel free to use this for your hackathon projects!
