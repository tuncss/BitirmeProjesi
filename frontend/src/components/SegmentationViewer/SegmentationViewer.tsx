import { useEffect, useRef, useState } from 'react'
import { Niivue, NVImage, SLICE_TYPE, cmapper } from '@niivue/niivue'
import styles from './SegmentationViewer.module.css'

interface SegmentationViewerProps {
  overlayUrl: string
  backgroundUrl?: string
}

type ViewMode = 'multi' | 'axial' | 'coronal' | 'sagittal' | 'render'

const VIEW_MODES: { key: ViewMode; label: string; sliceType: SLICE_TYPE }[] = [
  { key: 'multi',    label: 'Multi',    sliceType: SLICE_TYPE.MULTIPLANAR },
  { key: 'axial',    label: 'Aksiyel',  sliceType: SLICE_TYPE.AXIAL },
  { key: 'coronal',  label: 'Koronal',  sliceType: SLICE_TYPE.CORONAL },
  { key: 'sagittal', label: 'Sagital',  sliceType: SLICE_TYPE.SAGITTAL },
  { key: 'render',   label: '3D',       sliceType: SLICE_TYPE.RENDER },
]

// BraTS labels are {0=bg, 1=NCR, 2=ED, 4=ET}. Pre-build the discrete label
// LUT so we can pass it via volume options at load time (avoids touching the
// volume after the WebGL upload).
const BRATS_LABEL_LUT = cmapper.makeLabelLut({
  R: [0, 239, 34,  0,  59],
  G: [0, 68,  197, 0,  130],
  B: [0, 68,  94,  0,  246],
  A: [0, 255, 255, 0,  255],
  I: [0, 1,   2,   3,  4],
  labels: ['Background', 'NCR/Necrosis', 'Edema (ED)', '', 'Enhancing Tumor (ET)'],
})

const LEGEND = [
  { label: 'NCR/Nekroz', color: '#ef4444' },
  { label: 'Ödem (ED)',  color: '#22c55e' },
  { label: 'ET (Tümör)', color: '#3b82f6' },
]

async function fetchVolumeBuffer(url: string, label: string): Promise<ArrayBuffer> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`${label} dosyası alınamadı (HTTP ${response.status}).`)
  }
  const buf = await response.arrayBuffer()
  console.log(`[SegmentationViewer] fetched ${label}: ${buf.byteLength} bytes from ${url}`)
  if (buf.byteLength < 4) {
    throw new Error(`${label} boş geldi (${buf.byteLength} byte).`)
  }
  const head = new Uint8Array(buf, 0, 4)
  console.log(`[SegmentationViewer] ${label} magic bytes: ${[...head].map(b => b.toString(16).padStart(2, '0')).join(' ')}`)
  return buf
}

export default function SegmentationViewer({ backgroundUrl, overlayUrl }: SegmentationViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('multi')
  const [opacity, setOpacity] = useState(0.7)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let cancelled = false
    setLoading(true)
    setError(null)

    const nv = new Niivue({
      backColor: [0.06, 0.09, 0.13, 1],
      crosshairColor: [1, 1, 0, 0.5],
      show3Dcrosshair: true,
    })
    nvRef.current = nv

    ;(async () => {
      try {
        // Pre-fetch volumes ourselves so we control errors and bypass
        // Niivue's URL parsing entirely (our /api/download/{id}?type=...
        // URL has no .nii.gz extension which trips its filename-based
        // format detection). loadVolumes() drops the `buffer` field on
        // its way through addVolumesFromUrl, so we can't use it; instead
        // we build NVImage instances directly and addVolume() each one.
        const overlayBuffer = await fetchVolumeBuffer(overlayUrl, 'Segmentasyon maskesi')
        const backgroundBuffer = backgroundUrl
          ? await fetchVolumeBuffer(backgroundUrl, 'Arka plan MR')
          : null
        if (cancelled) return

        await nv.attachToCanvas(canvas)
        if (cancelled) return

        if (backgroundBuffer) {
          const bg = await NVImage.loadFromUrl({
            url: backgroundUrl!,
            buffer: backgroundBuffer,
            name: 'background.nii.gz',
            colormap: 'gray',
            opacity: 1,
          })
          if (cancelled) return
          nv.addVolume(bg)
        }

        const overlay = await NVImage.loadFromUrl({
          url: overlayUrl,
          buffer: overlayBuffer,
          name: 'segmentation.nii.gz',
          opacity: backgroundBuffer ? opacity : 1,
          cal_min: 0,
          cal_max: 4,
        })
        if (cancelled) return
        // NVImage.loadFromUrl drops the colormapLabel option (it isn't in
        // its destructured parameters), so attach it manually before the
        // volume is uploaded to the GPU.
        overlay.colormapLabel = BRATS_LABEL_LUT
        nv.addVolume(overlay)

        nv.setSliceType(SLICE_TYPE.MULTIPLANAR)
      } catch (e) {
        if (cancelled) return
        const message = e instanceof Error ? e.message : String(e)
        console.error('[SegmentationViewer] load failed:', e)
        setError(`Görüntü yüklenemedi: ${message}`)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()

    return () => {
      cancelled = true
      nvRef.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backgroundUrl, overlayUrl])

  const handleViewMode = (mode: ViewMode, sliceType: SLICE_TYPE) => {
    setViewMode(mode)
    nvRef.current?.setSliceType(sliceType)
  }

  const handleOpacity = (value: number) => {
    setOpacity(value)
    // volIdx 1 when background is present, 0 when overlay-only
    nvRef.current?.setOpacity(backgroundUrl ? 1 : 0, value)
  }

  return (
    <div className={styles.wrapper}>
      {/* Toolbar */}
      <div className={styles.toolbar}>
        <div className={styles.viewButtons}>
          {VIEW_MODES.map(({ key, label, sliceType }) => (
            <button
              key={key}
              className={[styles.viewBtn, viewMode === key ? styles.viewBtnActive : ''].filter(Boolean).join(' ')}
              onClick={() => handleViewMode(key, sliceType)}
              disabled={loading}
            >
              {label}
            </button>
          ))}
        </div>

        <div className={styles.opacityRow}>
          <span className={styles.opacityLabel}>Maske</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => handleOpacity(parseFloat(e.target.value))}
            className={styles.slider}
            disabled={loading}
            aria-label="Maske opaklığı"
          />
          <span className={styles.opacityValue}>{Math.round(opacity * 100)}%</span>
        </div>
      </div>

      {/* Canvas area */}
      <div className={styles.canvasWrap}>
        {loading && (
          <div className={styles.overlay}>
            <span className={styles.spinner} />
            <span>Görüntü yükleniyor…</span>
          </div>
        )}
        {error && <div className={styles.errorOverlay}>{error}</div>}
        <canvas ref={canvasRef} className={styles.canvas} />
      </div>

      {/* Legend */}
      <div className={styles.legend}>
        {LEGEND.map(({ label, color }) => (
          <div key={label} className={styles.legendItem}>
            <span className={styles.legendDot} style={{ backgroundColor: color }} />
            <span className={styles.legendLabel}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
