// API client — Axios instance + typed helpers for Accident-Intel
// Demo mode is enabled unless explicitly disabled. This makes a fresh clone
// usable without requiring a running FastAPI server.

import axios, {
  type AxiosInstance,
  type AxiosResponse,
  type InternalAxiosRequestConfig,
} from 'axios';

// ---------------------------------------------------------------------------
// Environment flags
// ---------------------------------------------------------------------------

export const isMockMode: boolean = import.meta.env.VITE_USE_MOCK !== 'false';

const BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const client: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
});

// Request interceptor — attach Content-Type for JSON payloads
client.interceptors.request.use(
  (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    if (!config.headers['Content-Type']) {
      config.headers['Content-Type'] = 'application/json';
    }
    return config;
  },
  (error: unknown) => Promise.reject(error),
);

// Response interceptor — normalize errors into a consistent shape
client.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error)) {
      const message: string =
        (error.response?.data as { detail?: string })?.detail ??
        error.message ??
        'An unknown API error occurred';
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
    headers: { 'Content-Type': 'multipart/form-data' },
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
