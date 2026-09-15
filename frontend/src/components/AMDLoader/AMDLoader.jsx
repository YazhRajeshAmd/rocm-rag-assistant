import styles from './AMDLoader.module.css';

export default function AMDLoader({ label = 'Querying with AMD MI300X...' }) {
  return (
    <div className={styles.wrapper}>
      <svg width="56" height="56" viewBox="0 0 56 56" className={styles.spinner}>
        <defs>
          <linearGradient id="shimmerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#00C2DE">
              <animate attributeName="stop-color" values="#00C2DE;#C1A968;#00C2DE" dur="1.6s" repeatCount="indefinite" />
            </stop>
            <stop offset="100%" stopColor="#C1A968">
              <animate attributeName="stop-color" values="#C1A968;#00C2DE;#C1A968" dur="1.6s" repeatCount="indefinite" />
            </stop>
          </linearGradient>
        </defs>
        <circle cx="28" cy="28" r="22" fill="none" stroke="var(--amd-border)" strokeWidth="3" />
        <circle
          cx="28" cy="28" r="22"
          fill="none"
          stroke="url(#shimmerGrad)"
          strokeWidth="3"
          strokeDasharray="80 60"
          strokeLinecap="round"
        >
          <animateTransform attributeName="transform" type="rotate" from="0 28 28" to="360 28 28" dur="1.2s" repeatCount="indefinite" />
        </circle>
      </svg>
      <p className={styles.label}>{label}</p>
      <div className={styles.dots}>
        <span className={styles.dot} style={{ animationDelay: '0ms' }} />
        <span className={styles.dot} style={{ animationDelay: '160ms' }} />
        <span className={styles.dot} style={{ animationDelay: '320ms' }} />
      </div>
    </div>
  );
}
