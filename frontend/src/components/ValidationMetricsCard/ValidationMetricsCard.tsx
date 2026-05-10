import type { ValidationResult } from '../../types'
import styles from './ValidationMetricsCard.module.css'

interface ValidationMetricsCardProps {
  validation: ValidationResult
}

const REGIONS = [
  {
    key: 'WT',
    label: 'WT',
    description: 'Tum tumor',
  },
  {
    key: 'TC',
    label: 'TC',
    description: 'Tumor core',
  },
  {
    key: 'ET',
    label: 'ET',
    description: 'Enhancing tumor',
  },
] as const

function diceStatus(dice: number): 'good' | 'warn' | 'bad' {
  if (dice >= 0.85) return 'good'
  if (dice >= 0.7) return 'warn'
  return 'bad'
}

function formatDice(value: number): string {
  return value.toFixed(3)
}

function formatHd95(value: number): string {
  if (value < 0) return '\u2014'
  return `${value.toFixed(1)} mm`
}

function statusIcon(status: 'good' | 'warn' | 'bad'): string {
  if (status === 'good') return '+'
  if (status === 'warn') return '~'
  return '!'
}

export default function ValidationMetricsCard({ validation }: ValidationMetricsCardProps) {
  return (
    <section className={styles.container} aria-labelledby="validation-metrics-title">
      <div className={styles.header}>
        <div>
          <h2 id="validation-metrics-title" className={styles.heading}>
            Referans Uyum Metrikleri
          </h2>
          <p className={styles.subject}>{validation.subject_id}</p>
        </div>
        <button
          type="button"
          className={styles.helpBtn}
          aria-label="Bolge aciklamalari"
          title="WT: tum tumor, TC: tumor core, ET: enhancing tumor"
        >
          ?
        </button>
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Bolge</th>
              <th scope="col">Dice</th>
              <th scope="col">HD95</th>
            </tr>
          </thead>
          <tbody>
            {REGIONS.map((region) => {
              const metrics = validation.metrics[region.key]
              const status = diceStatus(metrics.dice)

              return (
                <tr key={region.key}>
                  <th scope="row">
                    <span className={styles.regionCode}>{region.label}</span>
                    <span className={styles.regionDesc}>{region.description}</span>
                  </th>
                  <td className={styles.numCell}>
                    <span className={[styles.score, styles[status]].join(' ')}>
                      <span className={styles.scoreIcon} aria-hidden="true">
                        {statusIcon(status)}
                      </span>
                      {formatDice(metrics.dice)}
                    </span>
                  </td>
                  <td className={styles.numCell}>{formatHd95(metrics.hd95)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
