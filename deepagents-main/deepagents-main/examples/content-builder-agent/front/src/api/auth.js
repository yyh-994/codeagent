/**
 * 认证 API
 */
import { request } from './client.js'

export function login(username, password) {
  return request('POST', '/auth/login', { username, password })
}

export function logout() {
  return request('POST', '/auth/logout')
}
