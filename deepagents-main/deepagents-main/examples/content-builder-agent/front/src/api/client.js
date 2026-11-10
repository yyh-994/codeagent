/**
 * HTTP 客户端 — 基础 fetch 封装 + 鉴权
 */
const BASE_URL = 'http://127.0.0.1:2024'

function getToken() {
  return localStorage.getItem('access_token') || ''
}

/**
 * 通用 JSON 请求 — 自动带 token、自动解析 JSON、统一抛错。
 */
export async function request(method, path, body = null, isFormData = false) {
  const url = `${BASE_URL}${path}`
  const headers = {}
  if (!isFormData) headers['Content-Type'] = 'application/json'
  const token = getToken()
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(url, {
    method,
    headers,
    body: isFormData ? body : (body ? JSON.stringify(body) : null),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    const error = new Error(err.msg || err.detail || `HTTP ${res.status}`)
    error.status = res.status
    throw error
  }
  return res.json()
}

/**
 * 流式 POST — 返回原始 Response，由调用方自行读取 SSE 流。
 */
export async function streamPost(path, body, signal = null) {
  const token = getToken()
  return fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
    signal,
  })
}
