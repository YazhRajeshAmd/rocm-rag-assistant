import { useState, useEffect } from 'react';
import { Client } from '@gradio/client';
import Header from './components/Header/Header';
import Footer from './components/Footer/Footer';
import QueryForm from './components/QueryForm/QueryForm';
import ResultsPanel from './components/ResultsPanel/ResultsPanel';
import MetricsBar from './components/MetricsBar/MetricsBar';
import FeatureCards from './components/FeatureCards/FeatureCards';
import OnboardingTour from './components/OnboardingTour/OnboardingTour';

const MODEL_KEYS = [
  { key: 'mixtral', label: 'Mixtral 8x7B' },
  { key: 'llama', label: 'Llama 3.1 8B' },
  { key: 'gemma', label: 'Gemma 2 27B' },
  { key: 'phi', label: 'Phi 3 14B' },
];

const DEFAULT_METRICS = [
  { label: 'LLMs Running', value: 4, suffix: '' },
  { label: 'GPU Memory', value: 192, suffix: 'GB' },
  { label: 'ROCm Doc Pages', value: 36, suffix: '' },
  { label: 'HBM3 Bandwidth', value: 5300, suffix: 'GB/s' },
];

export default function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('rocm_theme') || 'dark'
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [sessionMetrics, setSessionMetrics] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('rocm_theme', theme);
  }, [theme]);

  const handleSubmit = async (question) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const client = await Client.connect('http://localhost:7866');
      const result = await client.predict('/process_question', { question });

      const data = result.data;

      const modelResults = MODEL_KEYS.map((m, i) => ({
        model: m.label,
        response: data[i * 2] ?? 'No response received.',
        metrics: data[i * 2 + 1] ?? {},
      }));

      setResults(modelResults);
      setSessionMetrics({ session: data[8], system: data[9] });
    } catch (err) {
      setError(err?.message || 'Could not reach backend on port 7866.');
    } finally {
      setLoading(false);
    }
  };

  const avgTokensPerSec = results
    ? results.reduce((sum, r) => {
        const v = parseFloat(r.metrics?.['Tokens/Second'] ?? 0);
        return sum + (isNaN(v) ? 0 : v);
      }, 0) / results.length
    : 0;

  const activeMetrics = results
    ? [
        { label: 'LLMs Queried', value: 4, suffix: '' },
        { label: 'Avg Tokens/s', value: Math.round(avgTokensPerSec), suffix: '' },
        {
          label: 'Sources Used',
          value: Number(results[0]?.metrics?.['Sources Used'] ?? 0),
          suffix: '',
        },
        { label: 'GPU Memory', value: 192, suffix: 'GB' },
      ]
    : DEFAULT_METRICS;

  return (
    <>
      <Header />
      <main style={{ flex: 1, position: 'relative', zIndex: 1 }}>
        <QueryForm onSubmit={handleSubmit} loading={loading} />
        <MetricsBar metrics={activeMetrics} />
        {(loading || results || error) && (
          <ResultsPanel results={results} loading={loading} error={error} />
        )}
        <FeatureCards />
      </main>
      <Footer />
      <OnboardingTour
        theme={theme}
        onToggle={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />
    </>
  );
}
