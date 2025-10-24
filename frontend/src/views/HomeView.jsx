import React from 'react';
import SearchBar from '../components/SearchBar';
import ResultCard from '../components/ResultCard';
import ReportViewer from '../components/ReportViewer';
import useSearchViewModel from '../viewmodels/useSearchViewModel';

const HomeView = () => {
    const {
        query,
        setQuery,
        results,
        report,
        loading,
        error,
        handleSearch,
        handleGenerateReport
    } = useSearchViewModel();

    return (
        <div className="home-view">
            <p className="intro-text">Search through the indexed reel transcriptions to find insights.</p>
            <SearchBar 
                query={query} 
                setQuery={setQuery} 
                onSearch={handleSearch} 
                onGenerateReport={handleGenerateReport}
                loading={loading}
            />

            {error && <div className="error-message">Error: {error}</div>}

            {loading && query && <div className="loading-indicator">Searching...</div>}

            <div className="results-container">
                {results.length > 0 && (
                    <div className="search-results">
                        <h2>Search Results</h2>
                        {results.map(result => (
                            <ResultCard key={result.id} result={result} />
                        ))}
                    </div>
                )}

                {report && (
                    <div className="report-section">
                        <h2>Generated Report</h2>
                        <ReportViewer report={report} />
                    </div>
                )}
            </div>
        </div>
    );
};

export default HomeView;
