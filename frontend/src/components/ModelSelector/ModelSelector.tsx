import { useState } from 'react'
import { MODEL_OPTIONS, type ModelOption } from '../../types'
import styles from './ModelSelector.module.css'

interface ModelSelectorProps {
  onSelect: (modelName: ModelOption['value']) => void
  disabled?: boolean
}

export default function ModelSelector({ onSelect, disabled = false }: ModelSelectorProps) {
  const [selected, setSelected] = useState<ModelOption['value']>('attention_unet3d')

  const handleSelect = (value: ModelOption['value']) => {
    if (disabled) return
    setSelected(value)
    onSelect(value)
  }

  return (
    <div className={styles.container}>
      <h3 className={styles.heading}>Model Seçimi</h3>
      <div className={styles.cards}>
        {MODEL_OPTIONS.map((model) => (
          <button
            key={model.value}
            className={[
              styles.card,
              selected === model.value ? styles.selected : '',
              disabled ? styles.disabled : '',
            ]
              .filter(Boolean)
              .join(' ')}
            onClick={() => handleSelect(model.value)}
            disabled={disabled}
            aria-pressed={selected === model.value}
          >
            <div className={styles.cardHeader}>
              <span className={styles.cardLabel}>{model.label}</span>
              {model.value === 'attention_unet3d' && (
                <span className={styles.badgeRecommended}>Önerilen</span>
              )}
              {model.value === 'unet3d' && (
                <span className={styles.badgeBaseline}>Baseline</span>
              )}
            </div>
            <p className={styles.cardDesc}>{model.description}</p>
            <div className={styles.metrics}>
              <Metric label="WT" value={model.wt_dice} />
              <Metric label="TC" value={model.tc_dice} />
              <Metric label="ET" value={model.et_dice} />
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className={styles.metric}>
      <span className={styles.metricLabel}>{label}</span>
      <span className={styles.metricValue}>{value.toFixed(3)}</span>
    </div>
  )
}
