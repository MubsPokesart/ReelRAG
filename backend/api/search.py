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
    semantic_weight = data.get('semantic_weight', 0.7) # New
    num_paraphrases = data.get('num_paraphrases', 3) # New

    try:
        current_app.logger.info(
            f"Search: query='{query}', k={k}, use_rocchio={use_rocchio}, "
            f"semantic_weight={semantic_weight}, num_paraphrases={num_paraphrases}"
        )
        retriever = current_app.retriever
        results = retriever.search(
            query,
            k=k,
            use_rocchio=use_rocchio,
            semantic_weight=semantic_weight,
            num_paraphrases=num_paraphrases
        )
        return jsonify(results)
    except Exception as e:
        current_app.logger.error(f"Search failed: {e}")
        return jsonify({"error": "An error occurred during search."}), 500
