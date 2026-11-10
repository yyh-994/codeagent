/**
 * 认证状态 — 全局单例响应式对象
 */
import { reactive } from 'vue'
import { login as httpLogin, logout as httpLogout } from '../api/auth.js'

export const auth = reactive({
  token: localStorage.getItem('access_token') || '',
  username: localStorage.getItem('username') || '',
  get isLoggedIn() { return !!this.token },

  async login(username, password) {
    const data = await httpLogin(username, password)
    this.token = data.access_token
    this.username = username
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('username', username)
    return data
  },

  async logout() {
    try { await httpLogout() } catch {}
    localStorage.removeItem('access_token')
    localStorage.removeItem('username')
    this.token = ''
    this.username = ''
  },
})
