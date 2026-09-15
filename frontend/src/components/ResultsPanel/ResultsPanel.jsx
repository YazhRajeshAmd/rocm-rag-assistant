import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import AMDLoader from '../AMDLoader/AMDLoader';
import styles from './ResultsPanel.module.css';

const TABS = [
  { key: 'mixtral', label: 'Mixtral 8x7B' },
  { key: 'llama', label: 'Llama 3.1 8B' },
  { key: 'gemma', label: 'Gemma 2 27B' },
  { key: 'phi', label: 'Phi 3 14B' },
];

const METRIC_LABELS = {
  'Input Tokens': 'Input Tokens',
  'Output Tokens': 'Output Tokens',
  'Retrieval Time': 'Retrieval',
  'Inference Time': 'Inference',
  'Tokens/Second': 'Tokens/s',
  'Total Time': 'Total',
  'Sources Used': 'Sources',
};

export default function ResultsPanel({ results, loading, error }) {
  const [activeTab, setActiveTab] = useState(0);

  if (loading) return (
    <section className={styles.section} data-tour="results">
      <AMDLoader label="Querying 4 models with AMD MI300X..." />
    </section>
  );

  if (error) return (
    <section className={styles.section} data-tour="results">
      <div className={styles.error}>
        <p className={styles.errorTitle}>Connection Error</p>
        <p className={styles.errorMsg}>{error}</p>
        <p className={styles.errorHint}>Ensure the backend is running: <code>python gpu1/gpu1.py</code></p>
      </div>
    </section>
  );

  if (!results) return null;

  const active = results[activeTab];

  return (
    <motion.section
      className={styles.section}
      data-tour="results"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className={styles.card}>
        <div className={styles.tabBar}>
          {TABS.map((tab, i) => (
            <button
              key={tab.key}
              className={`${styles.tab} ${activeTab === i ? styles.tabActive : ''}`}
              onClick={() => setActiveTab(i)}
            >
              {tab.label}
            </button>
          ))}
          <div
            className={styles.tabIndicator}
            style={{ left: `calc(${activeTab * 25}%)`, width: '25%' }}
          />
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className={styles.content}
          >
            <div className={styles.responseBlock}>
              <p className={styles.responseLabel}>Response</p>
              <pre className={styles.responseText}>{active?.response || 'No response received.'}</pre>
            </div>

            {active?.metrics && Object.keys(active.metrics).length > 0 && (
              <div className={styles.metricsRow}>
                {Object.entries(METRIC_LABELS).map(([key, label]) => (
                  active.metrics[key] !== undefined && (
                    <div key={key} className={styles.metricChip}>
                      <span className={styles.metricValue}>{active.metrics[key]}</span>
                      <span className={styles.metricLabel}>{label}</span>
                    </div>
                  )
                ))}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.section>
  );
}
