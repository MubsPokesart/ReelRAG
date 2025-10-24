import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomeView from './views/HomeView';
import './theme/global.css';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="app-header">
          <h1>ReelRAG Analyzer</h1>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<HomeView />} />
            {/* Add other routes for reports, etc. here if needed */}
          </Routes>
        </main>
        <footer className="app-footer">
          <p>Powered by Gemini</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
