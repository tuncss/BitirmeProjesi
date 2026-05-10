// ── Upload ──────────────────────────────────────────────────────────────────

export interface UploadResponse {
  session_id: string
  modalities: Record<string, string>
  message: string
}

// ── Segmentation ─────────────────────────────────────────────────────────────

export interface SegmentRequest {
  session_id: string
  model_name?: 'unet3d' | 'attention_unet3d'
}

export interface SegmentResponse {
  task_id: string
  celery_task_id: string
  status: 'submitted'
  model_name: string
  message: string
}

// ── Results ──────────────────────────────────────────────────────────────────

export interface TumorVolumes {
  WT_cm3: number
  TC_cm3: number
  ET_cm3: number
}

export interface ValidationMetrics {
  dice: number
  hd95: number
}

export interface ValidationResult {
  subject_id: string
  metrics: {
    WT: ValidationMetrics
    TC: ValidationMetrics
    ET: ValidationMetrics
  }
  gt_filename: string
}

export interface ResultFiles {
  segmentation: string
  metadata: string
  result_dir: string
  background?: string
  ground_truth?: string
}

export interface SegmentationMetadata {
  task_id?: string
  model_name: string
  processing_time_seconds: number
  tumor_volumes: TumorVolumes
  original_shape: number[]
  segmentation_classes: Record<string, string>
  has_background?: boolean
  files?: ResultFiles
  validation?: ValidationResult
}

export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface TaskResult {
  task_id: string
  status: TaskStatus
  progress: number
  step?: string
  message?: string
  // only when status === 'completed'
  results?: SegmentationMetadata
  // only when status === 'failed'
  error?: string
  // optional fields present when status === 'pending'
  model_name?: string
  celery_task_id?: string
}

// ── Health ───────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string
  environment: string
  gpu_available: boolean
  gpu_name: string | null
  device: 'cuda' | 'cpu'
  models: Record<string, string>
}

// ── UI helpers ───────────────────────────────────────────────────────────────

export interface ModelOption {
  value: 'unet3d' | 'attention_unet3d'
  label: string
  description: string
  wt_dice: number
  tc_dice: number
  et_dice: number
}

// Test seti üzerinde 126 vakada ölçülen ortalama Dice skorları
// (model_artifacts_final/final_model_comparison_report.json)
export const MODEL_OPTIONS: ModelOption[] = [
  {
    value: 'attention_unet3d',
    label: '3D Attention U-Net',
    description: 'Dikkat mekanizmalı geliştirilmiş model (önerilen)',
    wt_dice: 0.922,
    tc_dice: 0.904,
    et_dice: 0.842,
  },
  {
    value: 'unet3d',
    label: '3D U-Net',
    description: 'Temel encoder-decoder segmentasyon modeli',
    wt_dice: 0.923,
    tc_dice: 0.905,
    et_dice: 0.846,
  },
]
