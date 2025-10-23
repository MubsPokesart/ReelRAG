import chromadb
from flask import Blueprint, jsonify, current_app

topics_bp = Blueprint('topics_bp', __name__)

@topics_bp.route('/topics', methods=['GET'])
def get_topics():
    """Endpoint to retrieve all unique topic tags from the database."""
    try:
        current_app.logger.info("Retrieving all unique topic tags.")
        # In a real app, you might cache this result.
        client = chromadb.PersistentClient(path=current_app.db_path)
        collection = client.get_collection("reels")
        
        # Fetch all metadata. This could be slow on very large datasets.
        metadata = collection.get(include=["metadatas"])['metadatas']
        
        all_tags = set()
        for meta_item in metadata:
            if 'topic_tags' in meta_item and meta_item['topic_tags']:
                tags = [tag.strip() for tag in meta_item['topic_tags'].split(',')]
                all_tags.update(tags)
        
        return jsonify(sorted(list(all_tags)))
    except Exception as e:
        current_app.logger.error(f"Failed to retrieve topics: {e}")
        return jsonify({"error": "Could not retrieve topics."}), 500
