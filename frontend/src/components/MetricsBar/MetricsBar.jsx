import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import styles from './MetricsBar.module.css';

function useCountUp(target, duration = 1400) {
  const [value, setValue] = useState(0);
  const prevTarget = useRef(0);

  useEffect(() => {
    const start = prevTarget.current;
    const end = target;
    prevTarget.current = target;
    if (start === end) return;

    const startTime = performance.now();
    const tick = (now) => {
      const elapsed = now - startTime;
      const t = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(start + (end - start) * eased));
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target, duration]);

  return value;
}

function Metric({ label, value, suffix, decimals = 0 }) {
  const displayed = useCountUp(Math.round(value));
  return (
    <div className={styles.metric}>
      <span className={styles.value}>
        {decimals > 0 ? value.toFixed(decimals) : displayed}
        {suffix && <span className={styles.suffix}>{suffix}</span>}
      </span>
      <span className={styles.label}>{label}</span>
    </div>
  );
}

export default function MetricsBar({ metrics }) {
  return (
    <motion.div
      className={styles.bar}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.2, duration: 0.4 }}
    >
      {metrics.map((m) => (
        <Metric key={m.label} {...m} />
      ))}
    </motion.div>
  );
}
