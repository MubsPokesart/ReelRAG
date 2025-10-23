import os
import sys
from flask import Flask,jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.retriever import Retriever
from managers.report_manager import ReportManager
from api.search import search_bp
from api.topics import topics_bp
from api.report import report_bp

import logging

def create_app():
    """Application factory for the Flask app."""
    app = Flask(__name__)
    CORS(app) # Allow all origins for simplicity in this example
    load_dotenv()  # make GOOGLE_API_KEY available if .env is present

    # Configure logging
    if not app.debug:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, 'app.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('ReelRAG API startup')


    # --- Initialize core components ---
    # These are initialized once and shared across requests
    # In a larger app, you might manage this with a proper DI container
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    DB_DIR = os.path.join(DATA_DIR, 'reels_db')
    
    app.retriever = Retriever(db_path=DB_DIR, logger=app.logger)
    app.report_manager = ReportManager(logger=app.logger)
    app.db_path = DB_DIR # For topics endpoint

    # --- Register Blueprints ---
    app.register_blueprint(search_bp)
    app.register_blueprint(topics_bp)
    app.register_blueprint(report_bp)

    @app.route("/")
    def index():
        app.logger.info('Index route accessed')
        return jsonify({"status": "ok", "message": "ReelRAG API is running."})

    return app

app = create_app()

if __name__ == '__main__':
    # This allows running the app directly for development
    # For production, use a WSGI server like Gunicorn or Waitress
    app.run(debug=True, port=5000)
