# Deep Chatbot

An intelligent chatbot built using Azure Bot Services, featuring containerized ML models and CI/CD pipelines for scalable deployment. Integrates OpenAI API and Azure AI Search for high-accuracy query handling and structured resolution.

## Features

- **Azure Bot Services Integration**: Leverages Azure Bot Framework for conversational AI capabilities.
- **Containerized ML Models**: OpenAI models deployed in containers for efficient, scalable inference.
- **CI/CD Pipelines**: Automated deployment and updates using Azure DevOps or GitHub Actions.
- **Persistent Memory**: Azure AI Search stores and retrieves conversation history using vector similarity.
- **High-Accuracy Query Handling**: Combines vector search with text fallback for precise context retrieval.
- **Structured Resolution**: User profiles and memory enable personalized, context-aware responses.

## Architecture

- **Bot Framework**: Azure Bot Services handles user interactions and routing.
- **ML Models**: Containerized OpenAI GPT-3.5-turbo and text-embedding-ada-002 for generation and embeddings.
- **Vector Search**: Azure AI Search index with HNSW for fast similarity search on conversation vectors.
- **Memory Management**: Documents stored with userId, content, vectors, timestamps, and profiles.
- **CI/CD**: Pipelines for building containers, testing, and deploying to Azure.

## Setup

1. **Prerequisites**:
   - Azure subscription with Bot Services, OpenAI, AI Search, Container Registry, DevOps.
   - Docker for containerization.

2. **Environment Variables**:
   ```
   AZURE_OPENAI_DEPLOYMENT=gpt35-turbo
   AZURE_OPENAI_API_VERSION=2023-12-01-preview
   AZURE_OPENAI_API_KEY=your_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
   AZURE_SEARCH_API_KEY=your_key
   AZURE_SEARCH_INDEX_NAME=deepchatbot
   AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
   ```

3. **Containerization**:
   - Dockerfile: Build image with Flask app and dependencies.
   - Deploy to Azure Container Instances or Kubernetes.

4. **CI/CD**:
   - GitHub Actions/Azure DevOps pipeline: Build on push, test, deploy containers.

5. **Run Locally**:
   ```
   pip install -r requirements.txt
   python main.py
   ```

## Usage

- **Bot Interface**: Interact via Azure Bot Channels (Web Chat, Teams, etc.).
- **API Endpoints**:
  - `/chat`: Handle chat messages with memory.
  - `/chat/completions`: OpenAI-compatible completions.
  - `/memories/<userId>`: Debug memory retrieval.
  - `/user_profile/<userId>`: Update profiles.

## Memory Example: Norway Investment Recall

First conversation: User asks about Norway → Bot stores general info.

Later query on investments → Bot retrieves Norway context and mentions Norway's US market investments, showcasing high-accuracy query handling.

### Screenshots

#### Bot Interface Setup
![Bot interface setup](Screenshot%202025-10-10%20225204.png)

#### Initial Norway Query
![Initial Norway query](Screenshot%202025-10-10%20235724.png)

#### Investment Query with Recall
![Investment query with recall](Screenshot%202025-10-11%20000820.png)

#### Structured Resolution in Action
![Structured resolution in action](Screenshot%202025-10-11%20001111.png)

## How Azure AI Search Enables High-Accuracy

- Queries embedded and searched via vectors for semantic similarity.
- Filters by userId ensure personalized results.
- Fallback to text search maintains reliability.
- Structured data (profiles, timestamps) aids resolution.

## Contributing

Follow CI/CD for changes. Test memory with multiple userIds.
