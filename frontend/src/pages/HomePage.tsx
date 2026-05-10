import { useNavigate } from 'react-router-dom'
import styles from './HomePage.module.css'

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Beyin Tümörü Segmentasyonu</h1>
      <p className={styles.subtitle}>
        3D U-Net ve Attention U-Net modelleri ile BraTS 2021 üzerinde eğitilmiş
        otomatik beyin tümörü segmentasyon sistemi.
      </p>
      <div className={styles.stats}>
        <div className={styles.stat}>
          <span className={styles.statValue}>0.922</span>
          <span className={styles.statLabel}>WT Dice (Attention U-Net)</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statValue}>0.904</span>
          <span className={styles.statLabel}>TC Dice</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statValue}>0.842</span>
          <span className={styles.statLabel}>ET Dice</span>
        </div>
      </div>
      <button className={styles.cta} onClick={() => navigate('/segment')}>
        Segmentasyona Başla →
      </button>
    </div>
  )
}
