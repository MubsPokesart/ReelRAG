import React, { useRef, useEffect } from 'react';

const SearchBar = ({ query, setQuery, onSearch, onGenerateReport, loading }) => {
    const textareaRef = useRef(null);

    // Auto-resize textarea height based on content
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            // 24px is line-height (1.5) * font-size (16px)
            const scrollHeight = textareaRef.current.scrollHeight;
            textareaRef.current.style.height = `${scrollHeight}px`;
        }
    }, [query]);

    const handleKeyDown = (e) => {
        // Submit on Enter, allow newline with Shift+Enter
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent new line in textarea
            onSearch();
        }
    };

    return (
        <div className="search-bar-wrapper">
            <div className="search-box-container">
                <textarea
                    ref={textareaRef}
                    className="search-input"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="e.g., best practices for summer internships..."
                    disabled={loading}
                    rows={1} // Start with a single line
                />
            </div>
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
