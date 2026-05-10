import { useState } from 'react'
import { api } from '../services/api'
import { type ModelOption } from '../types'
import ImageUploader from '../components/ImageUploader/ImageUploader'
import ModelSelector from '../components/ModelSelector/ModelSelector'
import ProgressTracker from '../components/ProgressTracker/ProgressTracker'
import styles from './SegmentationPage.module.css'

type Step = 'upload' | 'configure' | 'processing'

interface UploadState {
  sessionId: string
  modalities: Record<string, string>
}

const STEP_LABELS: Record<Step, string> = {
  upload: '1. Dosya Yükle',
  configure: '2. Model Seç',
  processing: '3. İşleniyor',
}

export default function SegmentationPage() {
  const [step, setStep] = useState<Step>('upload')
  const [upload, setUpload] = useState<UploadState | null>(null)
  const [selectedModel, setSelectedModel] = useState<ModelOption['value']>('attention_unet3d')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)

  const handleUploadComplete = (sessionId: string, modalities: Record<string, string>) => {
    setUpload({ sessionId, modalities })
    setStep('configure')
  }

  const handleStartSegmentation = async () => {
    if (!upload || starting) return
    setStarting(true)
    setStartError(null)

    try {
      const response = await api.startSegmentation({
        session_id: upload.sessionId,
        model_name: selectedModel,
      })
      setTaskId(response.task_id)
      setStep('processing')
    } catch (err) {
      setStartError(err instanceof Error ? err.message : 'Segmentasyon başlatılamadı.')
    } finally {
      setStarting(false)
    }
  }

  const steps: Step[] = ['upload', 'configure', 'processing']

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>Segmentasyon</h1>
        <nav className={styles.stepper} aria-label="Adımlar">
          {steps.map((s, i) => (
            <span key={s} className={styles.stepperItem}>
              {i > 0 && <span className={styles.stepperArrow}>›</span>}
              <span
                className={[
                  styles.stepperLabel,
                  s === step ? styles.stepActive : '',
                  steps.indexOf(step) > i ? styles.stepDone : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
              >
                {STEP_LABELS[s]}
              </span>
            </span>
          ))}
        </nav>
      </div>

      {/* Step 1: Upload */}
      {step === 'upload' && (
        <div className={styles.stepContent}>
          <p className={styles.hint}>
            4 modaliteyi (T1, T1ce, T2, FLAIR) ayrı .nii.gz dosyaları olarak
            veya hepsini içeren tek bir .zip olarak yükleyin.
          </p>
          <ImageUploader onUploadComplete={handleUploadComplete} />
        </div>
      )}

      {/* Step 2: Configure */}
      {step === 'configure' && upload && (
        <div className={styles.stepContent}>
          {/* Detected modalities */}
          <div className={styles.modalityGrid}>
            {Object.entries(upload.modalities).map(([mod, path]) => (
              <div key={mod} className={styles.modalityChip}>
                <span className={styles.modalityTag}>{mod.toUpperCase()}</span>
                <span className={styles.modalityFile}>{path.split(/[\\/]/).pop()}</span>
              </div>
            ))}
          </div>

          <ModelSelector
            onSelect={setSelectedModel}
            disabled={starting}
          />

          {startError && <p className={styles.error}>{startError}</p>}

          <div className={styles.actions}>
            <button
              className={styles.backBtn}
              onClick={() => { setUpload(null); setStep('upload') }}
              disabled={starting}
            >
              ← Geri
            </button>
            <button
              className={styles.startBtn}
              onClick={handleStartSegmentation}
              disabled={starting}
            >
              {starting ? 'Başlatılıyor…' : 'Segmentasyonu Başlat'}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Processing */}
      {step === 'processing' && taskId && (
        <div className={styles.stepContent}>
          <ProgressTracker taskId={taskId} />
        </div>
      )}
    </div>
  )
}
