import { motion } from 'framer-motion'
import styles from './Header.module.css'

const ROCM_LOGO = 'https://raw.githubusercontent.com/ROCm/rocm-docs-core/main/src/rocm_docs/rocm_docs_theme/static/images/rocm-logo.png'
const AMD_LOGO  = 'https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg'

// If logos fail to load (corporate network / CORS), the img element stays invisible
// but the pill header still renders correctly. Do NOT replace logos with text or SVG fallbacks.

export default function Header() {
  return (
    <motion.div
      className={styles.wrapper}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      <div className={styles.pillGlow} data-tour="header">
      <header className={styles.header}>
        <div className={styles.inner}>

          <div className={styles.logoLockup}>
            <img src={AMD_LOGO}  alt="AMD — Advanced Micro Devices"              className={styles.amdLogo}  onError={(e) => { e.currentTarget.style.opacity = '0' }} />
            <span className={styles.logoSeparator} aria-hidden="true" />
            <img src={ROCM_LOGO} alt="ROCm open-source GPU compute platform"     className={styles.rocmLogo} onError={(e) => { e.currentTarget.style.opacity = '0' }} />
          </div>

          <nav className={styles.nav}>
            <span className={styles.navItem}>ROCm Assistant</span>
            <span className={styles.navDot} />
            <span className={styles.navItem}>ROCm Platform</span>
            <span className={styles.navDot} />
            <span className={styles.navItem}>AMD MI300X</span>
          </nav>

          <div className={styles.badge}>
            <span className={styles.badgeDot} />
            Live Demo
          </div>

        </div>
      </header>
      </div>
    </motion.div>
  )
}
