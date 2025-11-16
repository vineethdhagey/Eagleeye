# Eagleeye: Maritime Analytics Platform

Eagleeye is a comprehensive maritime analytics platform that combines real-time port insights with advanced AI-powered conversational analytics. The platform consists of two main components: a web-based dashboard for visualizing port traffic and congestion data, and a Retrieval-Augmented Generation (RAG) pipeline for intelligent question-answering about maritime operations.

## 🏗️ Architecture Overview

### PortInsights Dashboard
A modern web application built with Flask that provides:
- **Interactive Map**: Real-time vessel tracking and port visualization using Mapbox
- **Analytics Dashboard**: Charts and KPIs for arrivals, congestion, and emissions
- **AI Chat Interface**: Conversational analytics powered by Grok-style interactions
- **Mock API Endpoints**: Simulated MarineTraffic API responses for development

### Maritime RAG Pipeline
A machine learning pipeline that includes:
- **Data Processing**: Text chunking and vector embeddings using FAISS
- **Synthetic Data Generation**: Instruction-response pairs for model training
- **QLoRA Fine-tuning**: Parameter-efficient fine-tuning of language models
- **Retrieval Evaluation**: Performance metrics for the RAG system

## 🚀 Features

### PortInsights Dashboard
- 🌍 **Interactive Map**: Visualize Baltic Sea ports with real-time vessel positions
- 📊 **Insights Dashboard**: View arrivals, waiting times, and CO₂ emissions
- 💬 **AI Assistant**: Ask questions about port traffic and get instant answers
- 🎨 **Modern UI**: Dark theme with animated backgrounds and responsive design
- 🔄 **Real-time Updates**: Simulated live vessel movement and port congestion

### Maritime RAG Pipeline
- 📈 **Vector Search**: Semantic search over maritime event data
- 🤖 **Model Fine-tuning**: QLoRA adaptation for domain-specific knowledge
- 📝 **Synthetic Training**: Generate instruction datasets from maritime data
- 📊 **Evaluation Suite**: Comprehensive metrics for retrieval and generation quality

## 🛠️ Tech Stack

### Frontend & Backend
- **Flask**: Python web framework for API endpoints
- **HTML/CSS/JavaScript**: Modern web interface
- **Chart.js**: Interactive data visualizations
- **Mapbox GL JS**: Interactive mapping and vessel tracking
- **Vanta.js**: Animated 3D backgrounds

### Machine Learning Pipeline
- **LangChain**: Framework for RAG applications
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **PEFT/QLoRA**: Parameter-efficient fine-tuning
- **Transformers**: Hugging Face model ecosystem

## 📁 Project Structure

```
Eagleeye/
├── README.md                           # This file
├── PortInsights/                       # Web Dashboard
│   └── PortInsights/
│       ├── app.py                      # Flask API server
│       ├── index.html                  # Landing page with chat
│       ├── insights.html               # Analytics dashboard
│       └── map.html                    # Interactive vessel map
└── maritime_rag_pipeline (2)/          # ML Pipeline
    └── maritime_rag_pipeline/
        ├── README.md                   # Pipeline documentation
        ├── requirements.txt            # Python dependencies
        ├── scripts/                    # Pipeline scripts
        │   ├── chunk_data.py          # Data chunking & embedding
        │   ├── generate_instructions.py # Synthetic data generation
        │   ├── fine_tune_qlora.py     # Model fine-tuning
        │   ├── evaluate_retrieval.py  # Retrieval evaluation
        │   └── evaluate_llm.py        # LLM evaluation
        ├── data/                      # Input data directory
        ├── vector_db/                 # FAISS vector database
        └── training_data/             # Generated training data
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js (optional, for frontend development)
- GPU recommended for model fine-tuning

### PortInsights Dashboard Setup

1. **Navigate to the dashboard directory:**
   ```bash
   cd PortInsights/PortInsights
   ```

2. **Install dependencies:**
   ```bash
   pip install flask
   ```

3. **Run the Flask server:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   - Main dashboard: `http://localhost:5050`
   - Insights page: `http://localhost:5050/insights.html`
   - Map view: `http://localhost:5050/map.html`

### Maritime RAG Pipeline Setup

1. **Navigate to the pipeline directory:**
   ```bash
   cd maritime_rag_pipeline\ \(2\)/maritime_rag_pipeline
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data:**
   - Place your cleaned maritime CSV as `data/final_data.csv`
   - Ensure columns include: `portName`, `portArrival`, `portDeparture`, `ais_VesselType`

5. **Build the vector database:**
   ```bash
   python scripts/chunk_data.py --input data/final_data.csv --output vector_db --sample_rows 0 --chunk_size 800 --chunk_overlap 150 --embedding_model sentence-transformers/all-MiniLM-L6-v2
   ```

6. **Generate training data:**
   ```bash
   python scripts/generate_instructions.py --input data/final_data.csv --output training_data/instructions.jsonl --num_samples 20000
   ```

7. **Fine-tune the model:**
   ```bash
   python scripts/fine_tune_qlora.py --dataset training_data/instructions.jsonl --model meta-llama/Llama-3-8b-instruct --output_dir training_data/qlora-adapter --epochs 2 --batch_size 2 --learning_rate 2e-4
   ```

## 📊 API Endpoints

The PortInsights Flask app provides the following mock endpoints:

- `GET /api/congestion/<portid>` - Port congestion data
- `GET /api/vessels/<portid>` - Vessel data for specific port
- `GET /api/vessels/` - All vessel data

Example response:
```json
{
  "PORTID": "KLA",
  "PORTNAME": "Klaipeda",
  "TIME_ANCH": 1.5,
  "TIME_PORT": 2.3,
  "VESSELS": 12
}
```

## 🎯 Use Cases

### Maritime Operations
- **Port Authorities**: Monitor congestion and plan berth allocations
- **Shipping Companies**: Track vessel positions and optimize routes
- **Environmental Agencies**: Monitor CO₂ emissions from port operations

### AI-Powered Analytics
- **Conversational Queries**: "What's the average waiting time in Klaipeda?"
- **Predictive Insights**: Historical pattern analysis and forecasting
- **Automated Reporting**: Generate insights from complex maritime data

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development` for debug mode
- `MAPBOX_ACCESS_TOKEN`: Required for map functionality (currently hardcoded)

### Model Configuration
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Base LLM**: `meta-llama/Llama-3-8b-instruct`
- **Fine-tuning**: QLoRA with 2 epochs, batch size 2

## 📈 Performance & Evaluation

### Retrieval Metrics
- Precision@K: Measure of relevant chunks retrieved
- Recall@K: Coverage of relevant information
- Mean Reciprocal Rank (MRR)

### Generation Metrics
- ROUGE scores for answer quality
- Factual accuracy on maritime queries
- Response coherence and relevance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Porathon Event**: Special thanks to the Porathon event for providing the platform and inspiration for this project
- **Professor Lawrence Henesey**: Special thanks to our professor for guidance and mentorship throughout the development process
- **MarineTraffic API**: Inspiration for data structure and endpoints
- **Hugging Face**: Model hosting and transformers library
- **Mapbox**: Interactive mapping capabilities
- **Vanta.js**: Beautiful animated backgrounds

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the individual component READMEs for detailed documentation
- Review the code comments for implementation details

---















<img width="1901" height="856" alt="Screenshot 2025-10-03 212829" src="https://github.com/user-attachments/assets/5f1fb2ae-4c4a-4da2-967f-00fb71b4da81" />


<img width="1917" height="862" alt="Screenshot 2025-10-03 212913" src="https://github.com/user-attachments/assets/d5eeebd2-9496-4c51-90de-6f3775daecda" />
