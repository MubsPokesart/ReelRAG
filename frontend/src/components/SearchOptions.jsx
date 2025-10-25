import React from 'react';

const SearchOptions = ({
    useRocchio,
    setUseRocchio,
    semanticWeight,
    setSemanticWeight,
    numParaphrases,
    setNumParaphrases,
    numResults, // New
    setNumResults, // New
    loading
}) => {
    return (
        <div className="search-options">
            <div className="option-item">
                <label htmlFor="rocchio-checkbox">
                    <input
                        type="checkbox"
                        id="rocchio-checkbox"
                        checked={useRocchio}
                        onChange={(e) => setUseRocchio(e.target.checked)}
                        disabled={loading}
                    />
                    Use Rocchio Refinement
                </label>
            </div>
            <div className="option-item">
                <label htmlFor="semantic-weight-slider">
                    Semantic Weight: {semanticWeight.toFixed(2)}
                </label>
                <input
                    type="range"
                    id="semantic-weight-slider"
                    min="0"
                    max="1"
                    step="0.05"
                    value={semanticWeight}
                    onChange={(e) => setSemanticWeight(parseFloat(e.target.value))}
                    disabled={loading}
                    className="slider"
                />
            </div>
            <div className="option-item">
                <label htmlFor="num-paraphrases-slider">
                    Query Paraphrases: {numParaphrases}
                </label>
                <input
                    type="range"
                    id="num-paraphrases-slider"
                    min="0"
                    max="5"
                    step="1"
                    value={numParaphrases}
                    onChange={(e) => setNumParaphrases(parseInt(e.target.value, 10))}
                    disabled={loading}
                    className="slider"
                />
            </div>
            <div className="option-item">
                <label htmlFor="num-results-input">
                    Number of Results:
                </label>
                <input
                    type="number"
                    id="num-results-input"
                    value={numResults}
                    onChange={(e) => setNumResults(parseInt(e.target.value, 10) || 0)}
                    disabled={loading}
                    className="number-input"
                    min="1"
                    max="100"
                />
            </div>
        </div>
    );
};

export default SearchOptions;
