from flask import Blueprint, request, jsonify, current_app

report_bp = Blueprint('report_bp', __name__)

@report_bp.route('/report', methods=['POST'])
def generate_report():
    """Endpoint to generate a narrative report from a query."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400

    query = data.get('query')
    n = data.get('n', 5) # Number of documents to base the report on

    try:
        current_app.logger.info(f"Generating report for query: '{query}' with n={n}")
        # 1. Retrieve the most relevant documents
        retriever = current_app.retriever
        retrieved_docs = retriever.search(query, k=n)

        if not retrieved_docs:
            return jsonify({"summary": "No relevant documents found for the query.", "citations": []})

        # 2. Generate the report based on these documents
        report_manager = current_app.report_manager
        report = report_manager.generate_report(query, retrieved_docs)

        if report:
            return jsonify(report)
        else:
            return jsonify({"error": "Failed to generate report."}), 500

    except Exception as e:
        current_app.logger.error(f"Report generation failed: {e}")
        return jsonify({"error": "An unexpected error occurred during report generation."}), 500
