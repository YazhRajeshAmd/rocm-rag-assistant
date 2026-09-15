import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <p className={styles.text}>
        Powered by{' '}
        <span className={styles.accent}>AMD Instinct MI300X</span>
        {' & '}
        <span className={styles.accent}>ROCm Open Software</span>
        {' '}·{' '}
        &copy; {new Date().getFullYear()} Advanced Micro Devices, Inc.
      </p>
    </footer>
  );
}
