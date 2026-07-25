// Thin fetch wrapper around the accidental-computer Go API.
// Base URL is configurable via VITE_API_BASE (default http://localhost:8088).
// No other network hosts are ever contacted.

export const API_BASE = (
  import.meta.env.VITE_API_BASE || 'http://localhost:8088'
).replace(/\/$/, '')

class ApiError extends Error {
  constructor(message, { status, cause } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.cause = cause
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { Accept: 'application/json' },
      ...options,
    })
  } catch (err) {
    // Network-level failure: server down, CORS, DNS, etc.
    throw new ApiError(
      `Could not reach the API at ${API_BASE}. Is the Go server running?`,
      { cause: err },
    )
  }
  if (!res.ok) {
    let body = ''
    try {
      body = await res.text()
    } catch {
      /* ignore */
    }
    throw new ApiError(
      `API returned ${res.status} ${res.statusText}${body ? `: ${body}` : ''}`,
      { status: res.status },
    )
  }
  return res.json()
}

export const api = {
  health: () => request('/api/health'),
  // Live single-rollup per-block STF proofs (grouped, one namespace).
  blockProofs: () => request('/api/blockproofs'),
  // Per-block record for the Verify / Get Proof buttons.
  blockProof: (ns, height) =>
    request(`/api/blockproof?ns=${encodeURIComponent(ns)}&height=${height}`),
}

export { ApiError }
