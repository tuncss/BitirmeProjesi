import { type ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import styles from './Layout.module.css'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <Link to="/" className={styles.brand}>
          🧠 Beyin Tümörü Segmentasyonu
        </Link>
        <nav className={styles.nav}>
          <Link
            to="/"
            className={location.pathname === '/' ? styles.activeLink : styles.link}
          >
            Ana Sayfa
          </Link>
          <Link
            to="/segment"
            className={location.pathname === '/segment' ? styles.activeLink : styles.link}
          >
            Segmentasyon
          </Link>
        </nav>
      </header>
      <main className={styles.main}>{children}</main>
    </div>
  )
}
