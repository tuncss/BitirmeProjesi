import { useState, useCallback, useRef, type DragEvent, type ChangeEvent } from 'react'
import { api } from '../../services/api'
import styles from './ImageUploader.module.css'

const VALID_EXTENSIONS = ['.nii', '.nii.gz', '.zip']

function isValidFile(file: File): boolean {
  return VALID_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))
}

function formatMB(bytes: number): string {
  return (bytes / 1_000_000).toFixed(1)
}

interface ImageUploaderProps {
  onUploadComplete: (sessionId: string, modalities: Record<string, string>) => void
}

export default function ImageUploader({ onUploadComplete }: ImageUploaderProps) {
  const [files, setFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const addFiles = useCallback((incoming: File[]) => {
    const valid = incoming.filter(isValidFile)
    const invalid = incoming.filter((f) => !isValidFile(f))

    if (invalid.length > 0) {
      setError(
        `Geçersiz format: ${invalid.map((f) => f.name).join(', ')}. Yalnızca .nii, .nii.gz veya .zip kabul edilir.`,
      )
    } else {
      setError(null)
    }

    if (valid.length > 0) {
      setFiles((prev) => {
        // Aynı isimli dosyaları ekleme
        const existingNames = new Set(prev.map((f) => f.name))
        return [...prev, ...valid.filter((f) => !existingNames.has(f.name))]
      })
    }
  }, [])

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)
      addFiles(Array.from(e.dataTransfer.files))
    },
    [addFiles],
  )

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => setIsDragging(false), [])

  const handleBrowse = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(Array.from(e.target.files))
    // Reset input so the same file can be re-selected if removed
    e.target.value = ''
  }, [addFiles])

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }, [])

  const handleUpload = async () => {
    if (files.length === 0 || uploading) return
    setUploading(true)
    setError(null)
    setProgress(0)

    try {
      const response = await api.upload(files, setProgress)
      onUploadComplete(response.session_id, response.modalities)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Yükleme başarısız oldu.')
      setProgress(0)
    } finally {
      setUploading(false)
    }
  }

  const totalMB = files.reduce((sum, f) => sum + f.size, 0)

  return (
    <div className={styles.container}>
      {/* Drop zone */}
      <div
        className={[
          styles.dropZone,
          isDragging ? styles.dragging : '',
          files.length > 0 ? styles.hasFiles : '',
        ]
          .filter(Boolean)
          .join(' ')}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        aria-label="Dosya yükleme alanı"
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".nii,.nii.gz,.zip"
          onChange={handleBrowse}
          className={styles.hiddenInput}
          aria-hidden="true"
        />
        <div className={styles.dropContent}>
          <span className={styles.dropIcon}>🧠</span>
          <p className={styles.dropTitle}>
            {isDragging ? 'Bırakın...' : 'MR Görüntülerini Sürükleyip Bırakın'}
          </p>
          <p className={styles.dropSub}>T1, T1ce, T2 ve FLAIR modaliteleri</p>
          <span className={styles.browseHint}>.nii · .nii.gz · .zip</span>
        </div>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className={styles.fileList}>
          <div className={styles.fileListHeader}>
            <span>
              {files.length} dosya seçildi
            </span>
            <span className={styles.totalSize}>Toplam: {formatMB(totalMB)} MB</span>
          </div>
          {files.map((file, i) => (
            <div key={file.name} className={styles.fileItem}>
              <span className={styles.fileName}>{file.name}</span>
              <span className={styles.fileSize}>{formatMB(file.size)} MB</span>
              <button
                className={styles.removeBtn}
                onClick={() => removeFile(i)}
                disabled={uploading}
                aria-label={`${file.name} dosyasını kaldır`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Upload progress */}
      {uploading && (
        <div className={styles.progressWrapper}>
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: `${progress}%` }} />
          </div>
          <span className={styles.progressLabel}>{progress}%</span>
        </div>
      )}

      {/* Error */}
      {error && <p className={styles.error}>{error}</p>}

      {/* Upload button */}
      <button
        className={styles.uploadBtn}
        onClick={handleUpload}
        disabled={files.length === 0 || uploading}
      >
        {uploading ? `Yükleniyor… ${progress}%` : 'Dosyaları Yükle'}
      </button>
    </div>
  )
}
