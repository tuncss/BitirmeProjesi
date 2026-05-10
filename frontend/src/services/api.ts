import axios, { type AxiosProgressEvent } from 'axios'
import type {
  UploadResponse,
  SegmentRequest,
  SegmentResponse,
  TaskResult,
  HealthResponse,
} from '../types'

const client = axios.create({
  baseURL: '/api',
  timeout: 300_000, // 5 min — large NIfTI uploads
})

type ArtifactType = 'segmentation' | 'background' | 'ground_truth'
type ArtifactMode = 'inline' | 'attachment'

const apiBaseUrl = (): string => (
  import.meta.env.DEV
    ? (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000')
    : ''
)

// Global error normalizer — surface backend detail messages when available.
// Backend envelope: { error: { code, message, details }, meta: {...} }
// FastAPI default validation error envelope: { detail: [...] | "..." }
client.interceptors.response.use(
  (res) => res,
  (err) => {
    const data = err.response?.data
    const detail =
      data?.error?.message ??
      data?.detail ??
      data?.message
    if (detail) {
      err.message = typeof detail === 'string' ? detail : JSON.stringify(detail)
    }
    return Promise.reject(err)
  },
)

export const api = {
  /** System health check — also confirms GPU availability */
  health: async (): Promise<HealthResponse> => {
    const { data } = await client.get<HealthResponse>('/health')
    return data
  },

  /**
   * Upload NIfTI files (T1, T1ce, T2, FLAIR or a ZIP containing them).
   * onProgress receives 0-100 as upload bytes are sent.
   */
  upload: async (
    files: File[],
    onProgress?: (percent: number) => void,
  ): Promise<UploadResponse> => {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))

    const { data } = await client.post<UploadResponse>('/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (event: AxiosProgressEvent) => {
        if (event.total && onProgress) {
          onProgress(Math.round((event.loaded / event.total) * 100))
        }
      },
    })
    return data
  },

  /** Enqueue a segmentation job; returns task_id for polling */
  startSegmentation: async (req: SegmentRequest): Promise<SegmentResponse> => {
    const { data } = await client.post<SegmentResponse>('/segment', req)
    return data
  },

  /**
   * Poll task status.
   * status: 'pending' | 'processing' | 'completed' | 'failed'
   * When completed, data.results contains SegmentationMetadata.
   */
  getResults: async (taskId: string): Promise<TaskResult> => {
    const { data } = await client.get<TaskResult>(`/results/${taskId}`)
    return data
  },

  /**
   * Returns a URL for a task artifact.
   *  - mode='inline'     → octet-stream, no Content-Disposition (Niivue fetch)
   *  - mode='attachment' → application/gzip with attachment filename (download)
   *
   * In dev we point at the backend directly. Vite's proxy stalls on large
   * binary streams; CORS is already configured for localhost:5173. In a
   * built deployment Nginx proxies /api/* so the relative URL is correct.
   */
  downloadUrl: (
    taskId: string,
    type: ArtifactType = 'segmentation',
    mode: ArtifactMode = 'inline',
  ): string => {
    return `${apiBaseUrl()}/api/download/${taskId}?type=${type}&disposition=${mode}`
  },

  /**
   * Returns the inline URL for the optional BraTS ground-truth segmentation.
   * Callers should only use it when results.validation is present.
   */
  getGroundTruthUrl: (taskId: string): string => {
    return `${apiBaseUrl()}/api/results/${taskId}/files/ground_truth.nii.gz`
  },
}
