import { useParams, Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { api } from '../services/api'
import type { TaskResult } from '../types'
import SegmentationViewer from '../components/SegmentationViewer/SegmentationViewer'
import VolumeReport from '../components/VolumeReport/VolumeReport'
import styles from './ResultsPage.module.css'

export default function ResultsPage() {
  const { taskId } = useParams<{ taskId: string }>()
  const [result, setResult] = useState<TaskResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!taskId) return
    api.getResults(taskId).then(setResult).catch((e: Error) => {
      setError(e.message ?? 'Sonuçlar yüklenemedi.')
    })
  }, [taskId])

  if (error) {
    return (
      <div className={styles.centered}>
        <p className={styles.error}>{error}</p>
        <Link to="/segment" className={styles.retryLink}>← Yeni Segmentasyon</Link>
      </div>
    )
  }

  if (!result || !result.results) {
    return (
      <div className={styles.centered}>
        <span className={styles.spinner} />
        <p className={styles.loadingText}>Sonuçlar yükleniyor…</p>
      </div>
    )
  }

  const { results } = result

  return (
    <div className={styles.container}>
      {/* Page header */}
      <div className={styles.header}>
        <h1 className={styles.title}>Segmentasyon Sonuçları</h1>
        <Link to="/segment" className={styles.newBtn}>+ Yeni Segmentasyon</Link>
      </div>

      {/* NIfTI viewer */}
      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>3D Görüntü Görüntüleyici</h2>
        <SegmentationViewer
          backgroundUrl={results.has_background ? api.downloadUrl(taskId!, 'background') : undefined}
          overlayUrl={api.downloadUrl(taskId!, 'segmentation')}
        />
      </section>

      {/* Volume report */}
      <section className={styles.section}>
        <VolumeReport
          volumes={results.tumor_volumes}
          modelName={results.model_name}
          processingTime={results.processing_time_seconds}
        />
      </section>

      {/* Download */}
      <div className={styles.downloadRow}>
        <a
          href={api.downloadUrl(taskId!, 'segmentation', 'attachment')}
          download={`segmentation_${taskId}.nii.gz`}
          className={styles.downloadBtn}
        >
          Segmentasyon Maskesini İndir (.nii.gz)
        </a>
      </div>
    </div>
  )
}
