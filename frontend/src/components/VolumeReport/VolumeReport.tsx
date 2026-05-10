import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell,
  PieChart, Pie, ResponsiveContainer,
} from 'recharts'
import type { TumorVolumes } from '../../types'
import styles from './VolumeReport.module.css'

interface VolumeReportProps {
  volumes: TumorVolumes
  modelName: string
  processingTime: number
}

const COLORS = {
  WT: '#4A90D9',
  TC: '#E8A838',
  ET: '#D94A4A',
}

const MODEL_LABEL: Record<string, string> = {
  attention_unet3d: '3D Attention U-Net',
  unet3d: '3D U-Net',
}

export default function VolumeReport({ volumes, modelName, processingTime }: VolumeReportProps) {
  const chartData = [
    { name: 'Whole Tumor (WT)', value: volumes.WT_cm3, color: COLORS.WT, description: 'Tüm tümör bileşenleri (NCR + ED + ET)' },
    { name: 'Tumor Core (TC)',  value: volumes.TC_cm3, color: COLORS.TC, description: 'Ödem hariç tümör çekirdeği (NCR + ET)' },
    { name: 'Enhancing (ET)',   value: volumes.ET_cm3, color: COLORS.ET, description: 'Kontrast tutan aktif tümör bölgesi' },
  ]

  const totalVolume = volumes.WT_cm3

  return (
    <div className={styles.container}>
      <h3 className={styles.heading}>Hacimsel Analiz Raporu</h3>

      {/* Summary cards */}
      <div className={styles.summaryCards}>
        <div className={styles.summaryCard}>
          <span className={styles.cardLabel}>Model</span>
          <span className={styles.cardValue}>{MODEL_LABEL[modelName] ?? modelName}</span>
        </div>
        <div className={styles.summaryCard}>
          <span className={styles.cardLabel}>İşlem Süresi</span>
          <span className={styles.cardValue}>{processingTime.toFixed(1)} s</span>
        </div>
        <div className={[styles.summaryCard, styles.summaryCardHighlight].join(' ')}>
          <span className={styles.cardLabel}>Toplam Tümör Hacmi</span>
          <span className={styles.cardValue}>{totalVolume.toFixed(1)} cm³</span>
        </div>
      </div>

      {/* Detail table */}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Bölge</th>
              <th>Hacim (cm³)</th>
              <th>Oran (%)</th>
              <th>Açıklama</th>
            </tr>
          </thead>
          <tbody>
            {chartData.map((item) => (
              <tr key={item.name}>
                <td>
                  <span className={styles.colorDot} style={{ backgroundColor: item.color }} />
                  {item.name}
                </td>
                <td className={styles.numCell}>{item.value.toFixed(2)}</td>
                <td className={styles.numCell}>
                  {totalVolume > 0 ? ((item.value / totalVolume) * 100).toFixed(1) : '0.0'}%
                </td>
                <td className={styles.descCell}>{item.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Charts */}
      <div className={styles.chartsRow}>
        <div className={styles.chartBox}>
          <h4 className={styles.chartTitle}>Hacim Karşılaştırması</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#64748b', fontSize: 11 }}
                angle={-25}
                textAnchor="end"
                interval={0}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 11 }}
                label={{ value: 'cm³', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                labelStyle={{ color: '#cbd5e1' }}
                itemStyle={{ color: '#94a3b8' }}
                formatter={(val: number) => [`${val.toFixed(2)} cm³`, 'Hacim']}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className={styles.chartBox}>
          <h4 className={styles.chartTitle}>Bölge Oranları</h4>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={chartData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={75}
                label={({ name, percent }: { name: string; percent: number }) => {
                  const short = name.match(/\((\w+)\)/)?.[1] ?? name
                  return `${short} ${(percent * 100).toFixed(0)}%`
                }}
                labelLine={{ stroke: '#475569' }}
              >
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                labelStyle={{ color: '#cbd5e1' }}
                itemStyle={{ color: '#94a3b8' }}
                formatter={(val: number) => [`${val.toFixed(2)} cm³`, 'Hacim']}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Disclaimer */}
      <p className={styles.disclaimer}>
        Bu sonuçlar klinik karar destek amaçlıdır. Kesin tanı için radyoloji uzmanına başvurun.
      </p>
    </div>
  )
}
