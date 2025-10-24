import { useState } from 'react';
import axios from 'axios';

const useSearchViewModel = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const API_BASE_URL = '/api'; // Using proxy

    const handleSearch = async () => {
        if (!query) return;
        
        setLoading(true);
        setError('');
        setResults([]);
        setReport(null);

        try {
            const response = await axios.post(`${API_BASE_URL}/search`, { query });
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
            const response = await axios.post(`${API_BASE_URL}/report`, { query });
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
        handleGenerateReport
    };
};

export default useSearchViewModel;
