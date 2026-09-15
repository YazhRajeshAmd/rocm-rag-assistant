import { useState } from 'react';
import { motion } from 'framer-motion';
import styles from './QueryForm.module.css';

const EXAMPLES = [
  'How do I install ROCm on Ubuntu 22.04?',
  'What are the system requirements for AMD MI300X?',
  'How do I optimize memory usage in ROCm?',
  "What's the difference between HIP and CUDA?",
  'How do I debug ROCm applications?',
  'How do I install PyTorch with ROCm support?',
];

export default function QueryForm({ onSubmit, loading }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !loading) onSubmit(query.trim());
  };

  const handleExample = (ex) => {
    setQuery(ex);
  };

  return (
    <motion.section
      className={styles.section}
      data-tour="form"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className={styles.card}>
        <h2 className={styles.heading}>Ask the ROCm RAG Assistant</h2>
        <p className={styles.sub}>
          Query across 4 LLMs simultaneously — grounded in official AMD ROCm documentation
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <textarea
            className={styles.textarea}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. How do I install ROCm for machine learning?"
            rows={3}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit(e);
            }}
          />
          <div className={styles.actions}>
            <button
              type="submit"
              className={styles.submitBtn}
              disabled={!query.trim() || loading}
              data-tour="submit"
            >
              {loading ? 'Querying Models...' : 'Submit to All 4 Models'}
            </button>
            <button
              type="button"
              className={styles.clearBtn}
              onClick={() => setQuery('')}
              disabled={!query}
            >
              Clear
            </button>
          </div>
        </form>

        <div className={styles.examples}>
          <p className={styles.examplesLabel}>Example questions</p>
          <div className={styles.chips}>
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                className={styles.chip}
                onClick={() => handleExample(ex)}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      </div>
    </motion.section>
  );
}
