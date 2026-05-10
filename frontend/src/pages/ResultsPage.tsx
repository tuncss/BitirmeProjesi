import { useParams, Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { api } from '../services/api'
import type { TaskResult } from '../types'
import SegmentationViewer from '../components/SegmentationViewer/SegmentationViewer'
import VolumeReport from '../components/VolumeReport/VolumeReport'
import ValidationMetricsCard from '../components/ValidationMetricsCard/ValidationMetricsCard'
import styles from './ResultsPage.module.css'

export default function ResultsPage() {
  const { taskId } = useParams<{ taskId: string }>()
  const [result, setResult] = useState<TaskResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!taskId) return
    api.getResults(taskId).then(setResult).catch((e: Error) => {
      setError(e.message ?? 'Sonu\u00e7lar y\u00fcklenemedi.')
    })
  }, [taskId])

  if (error) {
    return (
      <div className={styles.centered}>
        <p className={styles.error}>{error}</p>
        <Link to="/segment" className={styles.retryLink}>Yeni Segmentasyon</Link>
      </div>
    )
  }

  if (!result || !result.results) {
    return (
      <div className={styles.centered}>
        <span className={styles.spinner} />
        <p className={styles.loadingText}>Sonu&ccedil;lar y&uuml;kleniyor...</p>
      </div>
    )
  }

  const { results } = result
  const backgroundUrl = results.has_background ? api.downloadUrl(taskId!, 'background') : undefined
  const segmentationUrl = api.downloadUrl(taskId!, 'segmentation')
  const groundTruthUrl = results.validation ? api.getGroundTruthUrl(taskId!) : undefined

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>Segmentasyon Sonu&ccedil;lar&#305;</h1>
        <Link to="/segment" className={styles.newBtn}>+ Yeni Segmentasyon</Link>
      </div>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>3D G&ouml;r&uuml;nt&uuml; G&ouml;r&uuml;nt&uuml;leyici</h2>
        <div className={results.validation ? styles.viewerGrid : styles.viewerSingle}>
          <div className={styles.viewerPanel}>
            <h3 className={styles.viewerLabel}>Model &Ccedil;&#305;kt&#305;s&#305;</h3>
            <SegmentationViewer
              backgroundUrl={backgroundUrl}
              overlayUrl={segmentationUrl}
            />
          </div>

          {groundTruthUrl && (
            <div className={styles.viewerPanel}>
              <h3 className={styles.viewerLabel}>Uzman Segmentasyonu (Ground Truth)</h3>
              <SegmentationViewer
                backgroundUrl={backgroundUrl}
                overlayUrl={groundTruthUrl}
              />
            </div>
          )}
        </div>

        {!results.validation && (
          <p className={styles.infoBand}>
            Bu dosya BraTS dataset pattern'&#305;na uymad&#305;&#287;&#305; i&ccedil;in referans kar&#351;&#305;la&#351;t&#305;rma yap&#305;lamad&#305;.
          </p>
        )}
      </section>

      {results.validation && (
        <ValidationMetricsCard validation={results.validation} />
      )}

      <section className={styles.section}>
        <VolumeReport
          volumes={results.tumor_volumes}
          modelName={results.model_name}
          processingTime={results.processing_time_seconds}
        />
      </section>

      <div className={styles.downloadRow}>
        <a
          href={api.downloadUrl(taskId!, 'segmentation', 'attachment')}
          download={`segmentation_${taskId}.nii.gz`}
          className={styles.downloadBtn}
        >
          Segmentasyon Maskesini &#304;ndir (.nii.gz)
        </a>
      </div>
    </div>
  )
}
