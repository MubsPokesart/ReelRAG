import React from 'react';

const ResultCard = ({ result }) => {
    if (!result) return null;

    const { metadata, snippet, score, id } = result;

    return (
        <div className="result-card">
            <div className="card-header">
                <a href={metadata.url} target="_blank" rel="noopener noreferrer" className="username-link">
                    @{metadata.username}
                </a>
                <span className="relevance-score">Relevance: {score.toFixed(3)}</span>
            </div>
            <div className="card-body">
                <p className="snippet">{snippet}</p>
            </div>
            <div className="card-footer">
                <span>Post Date: {metadata.post_date}</span>
                <span className="reel-id">ID: {id}</span>
            </div>
        </div>
    );
};

export default ResultCard;
