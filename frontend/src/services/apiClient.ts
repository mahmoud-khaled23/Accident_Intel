// API client — Axios instance + typed helpers for Accident-Intel
// Live API is the default. Demo fixtures require an explicit opt-in.

import axios, {
  type AxiosInstance,
  type AxiosResponse,
} from 'axios';

// ---------------------------------------------------------------------------
// Environment flags
// ---------------------------------------------------------------------------

export const isMockMode: boolean = import.meta.env.VITE_USE_MOCK === 'true';

const BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const client: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
});

/** Same origin in Studio; an explicit base URL also applies to video playback. */
export function apiUrl(path: string): string {
  return `${BASE_URL}${path}`;
}

// Response interceptor — normalize errors into a consistent shape
client.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error)) {
      const detail: unknown = error.response?.data?.detail;
      const message = typeof detail === 'string' ? detail
        : Array.isArray(detail) ? detail.map((item) => item.msg ?? 'Invalid input').join('; ')
        : !error.response
          ? 'Cannot reach the Accident Intel API. Start the FastAPI server on port 8000, then refresh this page.'
          : error.message || 'An unknown API error occurred';
      return Promise.reject(new Error(message));
    }
    return Promise.reject(error);
  },
);

// ---------------------------------------------------------------------------
// Typed helper wrappers
// ---------------------------------------------------------------------------

/** GET request — returns unwrapped response data */
export async function get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
  const response = await client.get<T>(url, { params });
  return response.data;
}

/** POST request with a JSON body — returns unwrapped response data */
export async function post<T>(url: string, body?: unknown): Promise<T> {
  const response = await client.post<T>(url, body);
  return response.data;
}

/** POST request with a FormData (multipart) body — returns unwrapped response data */
export async function postForm<T>(url: string, formData: FormData): Promise<T> {
  const response = await client.post<T>(url, formData, {
    timeout: 15 * 60_000, // Large videos may take several minutes to upload.
  });
  return response.data;
}

/** DELETE request — returns unwrapped response data */
export async function del<T>(url: string): Promise<T> {
  const response = await client.delete<T>(url);
  return response.data;
}

/** Artificial latency helper for realistic mock responses (ms) */
export function mockDelay(ms: number = 600): Promise<void> {
  const jitter = Math.floor(Math.random() * 200);
  return new Promise((resolve) => setTimeout(resolve, ms + jitter));
}

export default client;
