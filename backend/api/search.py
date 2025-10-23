from flask import Blueprint, request, jsonify, current_app

search_bp = Blueprint('search_bp', __name__)

@search_bp.route('/search', methods=['POST'])
def search():
    """Endpoint for performing a hybrid search."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400

    query = data.get('query')
    k = data.get('k', 10)
    use_rocchio = data.get('use_rocchio', False)
    # topic_filter and date_range are not implemented in the retriever yet
    # but are received here for future use.
    topic_filter = data.get('topic_filter')
    date_range = data.get('date_range')

    try:
        current_app.logger.info(f"Performing search with query: '{query}', k={k}, use_rocchio={use_rocchio}")
        retriever = current_app.retriever
        results = retriever.search(query, k=k, use_rocchio=use_rocchio)
        return jsonify(results)
    except Exception as e:
        current_app.logger.error(f"Search failed: {e}")
        return jsonify({"error": "An error occurred during search."}), 500
