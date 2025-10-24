import React from 'react';

const ReportViewer = ({ report }) => {
    if (!report) return null;

    const { summary, citations } = report;

    // A simple function to format the summary text, replacing markdown-like bolding
    const formatSummary = (text) => {
        return text.split('\n').map((paragraph, index) => {
            if (paragraph.trim() === '') return null;
            // Simple replacement for **text** to <strong>text</strong>
            const formattedParagraph = paragraph.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');
            return <p key={index} dangerouslySetInnerHTML={{ __html: formattedParagraph }} />;
        });
    };

    return (
        <div className="report-viewer">
            <div className="summary-content">
                {formatSummary(summary)}
            </div>
            {citations && citations.length > 0 && (
                <div className="citations-section">
                    <h3>Cited Sources</h3>
                    <ul className="citations-list">
                        {citations.map((citation, index) => (
                            <li key={index}>
                                [Doc {index + 1}] - 
                                <a href={citation.url} target="_blank" rel="noopener noreferrer">
                                    @{citation.username} (ID: {citation.document_id})
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default ReportViewer;
