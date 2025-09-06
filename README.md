# 🧠 AI MindMap Mentor

An intelligent backend system that converts vague goal descriptions into dynamic, hierarchical mindmaps. Unlike traditional linear roadmaps, this system creates interconnected concept trees with explanations, time estimates, and real educational resources fetched live from the web.

## 🚀 Features
- **Fast Mindmap Generation**: Builds tree structures with parallel processing (10-20s response time)
- **Resource Enrichment**: Real, verified educational resources via Tavily API
- **Background Enrichment**: Resources are added in the background for immediate user response
- **Parallel LLM Processing**: Multiple nodes expanded simultaneously for 60-80% speed improvement
- **Time Estimation**: Granular time estimates for each learning node
- **Explainability**: Query any node to understand why it exists
- **Interoperability**: Export as JSON, linear roadmap, or graph visualization

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python)
- **LLM**: Google Gemini (via API)
- **Resource Search**: Tavily API
- **Schema**: Pydantic models
- **Deployment**: Docker ready

## 📋 Prerequisites
- Python 3.8+
- Tavily API key (set TAVILY_API_KEY in your environment)

## 🚀 Quick Start
1. **Install dependencies**:

  ```bash
  pip install -r requirements.txt
  ```

2. **Run the application**:

  ```bash
  uvicorn app.main:app --reload
  ```

3. **Access the API**:
  - API docs: http://localhost:8000/docs
  - Generate mindmap: POST http://localhost:8000/generate_mindmap

## 📚 API Endpoints

### POST /generate_mindmap

Generate a mindmap from a goal description with fast initial response and background enrichment.

**Input**:

```json
{
  "description": "I want to learn AI in 6 months",
  "max_depth": 3,
  "time_constraint": "6 months"
}
```

**Output**: Initial mindmap structure (10-20 seconds) with resources populated in background.

### GET /mindmap/enriched/{timestamp}

Retrieve the enriched mindmap with resources after background processing is complete.

**Parameters**:

- `timestamp`: The `generated_at` timestamp from the initial mindmap response

**Output**: Complete mindmap with all resources populated.

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
├── data/                    # (Optional) Local cache or temp data
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

## ⚡ Performance Optimizations

The system has been optimized for fast response times:

### 🚀 Speed Improvements

- **Parallel LLM Processing**: Nodes at the same level are expanded simultaneously
- **Background Enrichment**: Resources are added in the background after initial response
- **Fast Initial Response**: Users get mindmap structure in 10-20 seconds instead of 80-120 seconds

### 📊 Expected Performance Gains

- **60-80% reduction** in LLM processing time through parallelization
- **Immediate user response** with mindmap structure
- **Background resource enrichment** for complete experience

### 🧪 Testing Performance

Run the performance test to verify improvements:

```bash
python test_performance.py
```

This will test:

1. Initial mindmap generation speed
2. Background enrichment process
3. Enriched mindmap retrieval

## 🔧 Development

- **Run tests**: `python -m pytest`
- **Test performance**: `python test_performance.py`
- **Format code**: `black .`
- **Lint code**: `flake8`

## 📝 License

MIT License - feel free to use this for your hackathon projects!
