import React from 'react';

const SearchBar = ({ query, setQuery, onSearch, onGenerateReport, loading }) => {
    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            onSearch();
        }
    };

    return (
        <div className="search-bar-container">
            <input
                type="text"
                className="search-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g., best practices for summer internships"
                disabled={loading}
            />
            <div className="button-group">
                <button onClick={onSearch} disabled={loading || !query} className="search-button">
                    {loading ? '...' : 'Search'}
                </button>
                <button onClick={onGenerateReport} disabled={loading || !query} className="report-button">
                    {loading ? '...' : 'Generate Report'}
                </button>
            </div>
        </div>
    );
};

export default SearchBar;
