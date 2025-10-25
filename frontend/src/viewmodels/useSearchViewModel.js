import { useState } from 'react';
import axios from 'axios';

const useSearchViewModel = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // New state for search options
    const [useRocchio, setUseRocchio] = useState(false);
    const [semanticWeight, setSemanticWeight] = useState(0.7);
    const [numParaphrases, setNumParaphrases] = useState(3);
    const [numResults, setNumResults] = useState(5); // New

    const API_BASE_URL = '/api'; // Using proxy

    const handleSearch = async () => {
        if (!query) return;
        
        setLoading(true);
        setError('');
        setResults([]);
        setReport(null);

        try {
            const payload = {
                query,
                use_rocchio: useRocchio,
                semantic_weight: semanticWeight,
                num_paraphrases: numParaphrases,
                k: numResults, // New
            };
            const response = await axios.post(`${API_BASE_URL}/search`, payload);
            setResults(response.data);
        } catch (err) {
            setError(err.response?.data?.error || 'An error occurred during search.');
        }
        setLoading(false);
    };

    const handleGenerateReport = async () => {
        if (!query) return;

        setLoading(true);
        setError('');
        setResults([]);
        setReport(null);

        try {
            // Pass numResults as 'n' for report generation
            const payload = {
                query,
                n: numResults, // New
            };
            const response = await axios.post(`${API_BASE_URL}/report`, payload);
            setReport(response.data);
        } catch (err) {
            setError(err.response?.data?.error || 'An error occurred while generating the report.');
        }
        setLoading(false);
    };

    return {
        query,
        setQuery,
        results,
        report,
        loading,
        error,
        handleSearch,
        handleGenerateReport,
        // Expose new state and setters
        useRocchio,
        setUseRocchio,
        semanticWeight,
        setSemanticWeight,
        numParaphrases,
        setNumParaphrases,
        numResults, // New
        setNumResults, // New
    };
};

export default useSearchViewModel;
