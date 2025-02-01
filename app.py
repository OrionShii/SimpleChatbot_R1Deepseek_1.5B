from flask import Flask, request, jsonify, send_from_directory
import requests
import logging
import json
from datetime import datetime
import time
import os
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
MODEL_NAME = os.getenv('MODEL_NAME', 'deepseek-r1:1.5b')
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', 1))
TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))

class AIServiceError(Exception):
    """Custom exception for AI service related errors"""
    pass

def query_deepseek(prompt):
    """
    Query local Deepseek model through Ollama with improved error handling and retries
    
    Args:
        prompt (str): The input prompt for the model
        
    Returns:
        str: The model's response
        
    Raises:
        AIServiceError: If all retry attempts fail
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempt {attempt + 1}: Sending request to Ollama")
            
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "max_tokens": 10000
                },
                timeout=TIMEOUT
            )
            
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            if 'response' in data:
                logger.info("Successfully received response from Ollama")
                return data['response']
            
            logger.warning(f"Unexpected response format: {data}")
            raise AIServiceError("Unexpected response format from Ollama")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise AIServiceError(f"Failed to connect to Ollama after {MAX_RETRIES} attempts")
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise AIServiceError(f"Unexpected error occurred: {str(e)}")

def get_fallback_response(prompt):
    """
    Provide a meaningful fallback response based on the input
    
    Args:
        prompt (str): The original user prompt
        
    Returns:
        str: A contextual fallback response
    """
    fallback_responses = {
        "what": "I understand you're asking for information. Could you please rephrase your question or break it down into smaller parts?",
        "how": "You're asking about a process or method. Could you be more specific about which part you'd like to understand?",
        "why": "I see you're looking for an explanation. Could you provide more context or specify what aspect you're most interested in?",
        "when": "You're asking about timing. Could you clarify the specific timeframe or event you're interested in?",
        "where": "I understand you're asking about location. Could you provide more context about what you're trying to find?",
        "who": "You're asking about someone. Could you provide more details about who you're interested in learning about?",
        "default": "I'm currently having difficulty processing your request. Could you rephrase your question or break it down into smaller parts?"
    }
    
    prompt_lower = prompt.lower().strip()
    for question_word, response in fallback_responses.items():
        if prompt_lower.startswith(question_word):
            return response
    
    return fallback_responses["default"]

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the static directory"""
    return send_from_directory(app.static_folder, path)

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat API endpoint
    
    Expected JSON payload:
    {
        "message": "user message",
        "history": [{"role": "user", "content": "previous message"}, ...]  # optional
    }
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Invalid request format',
                'timestamp': datetime.now().isoformat()
            }), 400

        user_message = data['message'].strip()
        chat_history = data.get('history', [])
        
        # Construct context-aware prompt
        prompt = f"""Context: Previous messages in conversation (if any):
{json.dumps(chat_history, indent=2)}

Current user message: {user_message}

Please provide a helpful, informative, and contextually appropriate response."""

        try:
            response = query_deepseek(prompt)
            return jsonify({
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            })
            
        except AIServiceError as e:
            logger.error(f"AI service error: {str(e)}")
            fallback = get_fallback_response(user_message)
            return jsonify({
                'response': fallback,
                'timestamp': datetime.now().isoformat(),
                'status': 'fallback',
                'error': str(e)
            }), 503
            
    except Exception as e:
        logger.exception("Unexpected error in chat endpoint")
        return jsonify({
            'response': "I apologize, but I encountered an unexpected error. Please try again.",
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Resource not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Flask application on {host}:{port}")
    app.run(host=host, port=port, debug=debug)