import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import uuid
import time
import json
import logging

load_dotenv()


logging.basicConfig(filename='flask_app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Load environment variables
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Separate embedding deployment name
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log environment variables for debugging
logger.info(f"AZURE_SEARCH_INDEX_NAME: {AZURE_SEARCH_INDEX_NAME}")
logger.info(f"AZURE_SEARCH_ENDPOINT: {AZURE_SEARCH_ENDPOINT}")
logger.info(f"AZURE_SEARCH_API_KEY: {'set' if AZURE_SEARCH_API_KEY else 'not set'}")
logger.info(f"AZURE_OPENAI_API_KEY: {'set' if AZURE_OPENAI_API_KEY else 'not set'}")
logger.info(f"AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT}")

# Initialize Azure Chat LLM
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    openai_api_key=AZURE_OPENAI_API_KEY,
)

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    openai_api_key=AZURE_OPENAI_API_KEY,
)

# Initialize Azure Search Client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    api_version="2023-11-01"
)

def encode_text(text: str):
    """Encode text to embeddings with proper error handling"""
    try:
        if not text or not isinstance(text, str):
            text = "empty"
        
        # Clean the text
        cleaned_text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Remove control characters
        import re
        cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_text)
        
        if not cleaned_text.strip():
            cleaned_text = "empty text"
        
        # Get embedding using embed_query for single text
        return embeddings.embed_query(cleaned_text)
        
    except Exception as e:
        print(f"Error encoding text: {e}")
        return embeddings.embed_query("default text")

def search_memories_vector(user_id: str, query: str, top_k: int = 5):
    """Search Azure AI Search using vector similarity"""
    try:
        # Get query embedding
        query_vector = encode_text(query)
        
        # Create vectorized query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="contentVector"
        )
        
        # Search with vector similarity and user filter
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=f"userId eq '{user_id}'",
            top=top_k
        )
        
        memories = []
        for result in results:
            if 'content' in result:
                memories.append({
                    'content': result['content'],
                    'score': result.get('@search.score', 0),
                    'timestamp': result.get('timestamp', 0)
                })

        logger.info(f"Vector search returned {len(memories)} memories for user {user_id}")
        return memories
        
    except Exception as e:
        print(f"Error in vector search: {e}")
        return search_memories_text(user_id, query, top_k)

def search_memories_text(user_id: str, query: str, top_k: int = 5):
    """Fallback: Search Azure AI Search using text search"""
    try:
        results = search_client.search(
            search_text=query,
            filter=f"userId eq '{user_id}'",
            top=top_k,
            search_mode='any'
        )
        
        memories = []
        for result in results:
            if 'content' in result:
                memories.append({
                    'content': result['content'],
                    'score': result.get('@search.score', 0),
                    'timestamp': result.get('timestamp', 0)
                })

        logger.info(f"Text search returned {len(memories)} memories for user {user_id}")
        return memories
        
    except Exception as e:
        print(f"Error in text search: {e}")
        return []

def add_memory_to_search(user_id: str, user_message: str, bot_response: str, user_profile: dict = None):
    """Add memory directly to Azure AI Search with optional user profile data"""
    try:
        content = f"User: {user_message}\nAssistant: {bot_response}"
        content_vector = encode_text(content)
        
        memory_document = {
            "id": f"{user_id}-{int(time.time() * 1000)}-{str(uuid.uuid4())[:8]}",
            "content": content,
            "contentVector": content_vector,
            "userId": user_id,
            "userMessage": user_message,
            "botResponse": bot_response,
            "timestamp": time.time()
        }
        
        if user_profile:
            # Flatten user profile dict into string for storage
            profile_str = json.dumps(user_profile)
            memory_document["userProfile"] = profile_str
        
        # Upload to Azure Search
        search_client.upload_documents(documents=[memory_document])
        print(f"Memory added for user {user_id}")
        
    except Exception as e:
        print(f"Error adding memory to Azure Search: {e}")

def get_relevant_memories(user_id: str, query: str, top_k: int = 3):
    """Get relevant memories using vector search with text fallback"""
    # Try vector search first
    memories = search_memories_vector(user_id, query, top_k)
    
    # If vector search fails or returns no results, try text search
    if not memories:
        memories = search_memories_text(user_id, query, top_k)

    relevant_memories = [mem['content'] for mem in memories]
    logger.info(f"Retrieved {len(relevant_memories)} relevant memories for user {user_id}")
    return relevant_memories

def generate_prompt_with_memory(user_input: str, user_id: str):
    """Generate prompt with memory context and user profile from Azure AI Search"""
    try:
        # Get relevant memories
        relevant_memories = get_relevant_memories(user_id, user_input, top_k=3)
        
        # Retrieve user profile from memories if available
        user_profile = None
        try:
            results = search_client.search(
                search_text="*",
                filter=f"userId eq '{user_id}' and userProfile ne null",
                top=1,
                order_by=["timestamp desc"]
            )
            for result in results:
                if "userProfile" in result:
                    user_profile = json.loads(result["userProfile"])
                    break
        except Exception as e:
            print(f"Error retrieving user profile: {e}")
        
        if relevant_memories:
            memory_context = "\n---\n".join(relevant_memories)
            profile_context = f"User Profile: {json.dumps(user_profile)}" if user_profile else "User Profile: Not available"
            system_message = f"""You are a helpful personal assistant with perfect memory. You remember all previous conversations with this user.

{profile_context}

Here are the most relevant past interactions:
{memory_context}

Use this context to provide personalized and contextually aware responses. Reference past conversations when relevant, but don't be repetitive."""
        else:
            profile_context = f"User Profile: {json.dumps(user_profile)}" if user_profile else "User Profile: Not available"
            system_message = f"You are a helpful personal assistant. This appears to be your first conversation with this user.\n\n{profile_context}"
        
        messages = [
            ("system", system_message),
            ("human", user_input)
        ]
        
        prompt = ChatPromptTemplate(messages)
        return prompt.format()
        
    except Exception as e:
        print(f"Error generating prompt: {e}")
        # Fallback to simple prompt
        simple_prompt = ChatPromptTemplate([
            ("system", "You are a helpful personal assistant."),
            ("human", user_input)
        ])
        return simple_prompt.format()

def chat_with_memory(user_input: str, user_id: str, user_profile: dict = None):
    """Main chat function with Azure AI Search memory and user profile"""
    try:
        # Generate prompt with memory context
        prompt = generate_prompt_with_memory(user_input, user_id)
        
        # Log the prompt for debugging
        logger.info(f"Generated prompt: {prompt}")
        
        # Get response from LLM
        # Fix: pass prompt as input parameter 'input' as expected by AzureChatOpenAI
        response = llm.invoke(input=prompt)
        response_content = response.content
        
        # Store the conversation in Azure AI Search with user profile
        add_memory_to_search(user_id, user_input, response_content, user_profile)
        
        return response_content
        
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        return "I apologize, but I encountered an error processing your request. Please try again."

# Flask app setup
app = Flask(__name__, static_folder='templates', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        user_id = data.get('userId', 'default_user')
        user_profile = data.get('userProfile', None)
        
        if not user_input:
            return jsonify({'error': 'Empty message'}), 400
        
        response_text = chat_with_memory(user_input, user_id, user_profile)
        return jsonify({'response': response_text})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/memories/<user_id>', methods=['GET'])
def get_user_memories(user_id):
    """Get all memories for a user (for debugging)"""
    try:
        results = search_client.search(
            search_text="*",
            filter=f"userId eq '{user_id}'",
            top=50,
            order_by=["timestamp desc"]
        )
        
        memories = []
        for result in results:
            memories.append({
                'content': result.get('content', ''),
                'timestamp': result.get('timestamp', 0)
            })
        
        return jsonify({'memories': memories, 'count': len(memories)})
        
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return jsonify({'error': 'Error retrieving memories'}), 500

@app.route('/user_profile/<user_id>', methods=['POST'])
def update_user_profile(user_id):
    """Endpoint to update user profile information"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No profile data provided'}), 400

        # Store user profile as a special memory document with userProfile field
        profile_document = {
            "id": f"profile-{user_id}",
            "userId": user_id,
            "userProfile": json.dumps(data),
            "timestamp": time.time()
        }

        search_client.upload_documents(documents=[profile_document])
        return jsonify({'message': 'User profile updated successfully'})
    except Exception as e:
        print(f"Error updating user profile: {e}")
        return jsonify({'error': 'Failed to update user profile'}), 500

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = request.json
        if not data or 'messages' not in data:
            return jsonify({'error': 'Invalid request: messages required'}), 400

        messages = data['messages']
        if not messages or not isinstance(messages, list):
            return jsonify({'error': 'Invalid messages format'}), 400

        # Convert messages to LangChain format
        from langchain.schema import SystemMessage, HumanMessage, AIMessage
        langchain_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'system':
                langchain_messages.append(SystemMessage(content=content))
            elif role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
            else:
                continue  # Skip unknown roles

        if not langchain_messages:
            return jsonify({'error': 'No valid messages provided'}), 400

        # Get response from LLM
        response = llm.invoke(langchain_messages)

        # Format response like OpenAI
        completion = {
            'id': f'chatcmpl-{str(uuid.uuid4())}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': data.get('model', AZURE_OPENAI_DEPLOYMENT),
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': response.content
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 0,  # Placeholder
                'completion_tokens': 0,  # Placeholder
                'total_tokens': 0  # Placeholder
            }
        }

        return jsonify(completion)

    except Exception as e:
        logger.error(f"Error in chat/completions: {e}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

if __name__ == "__main__":
    print("Starting chat application with Azure AI Search memory...")
    print(f"Azure Search Endpoint: {AZURE_SEARCH_ENDPOINT}")
    print(f"Search Index: {AZURE_SEARCH_INDEX_NAME}")
    app.run(host='0.0.0.0', port=8000, debug=True)
