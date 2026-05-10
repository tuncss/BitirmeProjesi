import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../services/api'
import type { TaskResult } from '../../types'
import styles from './ProgressTracker.module.css'

const STEPS = [
  { key: 'preprocessing', label: 'Ön İşleme', icon: '🔬' },
  { key: 'inference', label: 'Model Çıkarımı', icon: '🧠' },
  { key: 'postprocessing', label: 'Son İşleme', icon: '✨' },
] as const

const STEP_KEYS = STEPS.map((s) => s.key)

const POLL_INTERVAL_MS = 2000

interface ProgressTrackerProps {
  taskId: string
}

export default function ProgressTracker({ taskId }: ProgressTrackerProps) {
  const [result, setResult] = useState<TaskResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    const poll = async () => {
      try {
        const data = await api.getResults(taskId)
        setResult(data)

        if (data.status === 'completed') {
          clearInterval(timer)
          setTimeout(() => navigate(`/results/${taskId}`), 1200)
        }

        if (data.status === 'failed') {
          clearInterval(timer)
          setError(data.message ?? 'Segmentasyon başarısız oldu.')
        }
      } catch {
        // Geçici ağ hatası — bir sonraki poll'da tekrar dene
      }
    }

    const timer = setInterval(poll, POLL_INTERVAL_MS)
    poll() // İlk poll'u hemen yap, interval'ı bekleme
    return () => clearInterval(timer)
  }, [taskId, navigate])

  const progress = result?.progress ?? 0
  const currentStep = result?.step ?? 'preprocessing'
  const currentStepIndex = STEP_KEYS.indexOf(currentStep as typeof STEP_KEYS[number])
  const isDone = result?.status === 'completed'

  return (
    <div className={styles.container}>
      <h3 className={styles.heading}>
        {isDone ? '✓ Segmentasyon Tamamlandı' : 'Segmentasyon İşleniyor…'}
      </h3>

      {/* Step indicators */}
      <div className={styles.steps}>
        {STEPS.map((step, i) => {
          const isCompleted = isDone || i < currentStepIndex
          const isActive = !isDone && i === currentStepIndex

          return (
            <div
              key={step.key}
              className={[
                styles.step,
                isActive ? styles.stepActive : '',
                isCompleted ? styles.stepCompleted : '',
              ]
                .filter(Boolean)
                .join(' ')}
            >
              <div className={styles.stepIconWrap}>
                {isCompleted ? (
                  <span className={styles.checkmark}>✓</span>
                ) : (
                  <span className={styles.stepIcon}>{step.icon}</span>
                )}
              </div>
              <span className={styles.stepLabel}>{step.label}</span>
            </div>
          )
        })}
      </div>

      {/* Overall progress bar */}
      <div className={styles.progressRow}>
        <div className={styles.progressBar}>
          <div
            className={[styles.progressFill, isDone ? styles.progressDone : '']
              .filter(Boolean)
              .join(' ')}
            style={{ width: `${isDone ? 100 : progress}%` }}
          />
        </div>
        <span className={styles.progressPct}>{isDone ? 100 : progress}%</span>
      </div>

      {/* Status message */}
      <p className={styles.message}>
        {isDone
          ? 'Sonuç sayfasına yönlendiriliyorsunuz…'
          : (result?.message ?? 'Görev kuyruğa alındı, başlatılıyor…')}
      </p>

      {/* Error */}
      {error && <p className={styles.error}>{error}</p>}
    </div>
  )
}
