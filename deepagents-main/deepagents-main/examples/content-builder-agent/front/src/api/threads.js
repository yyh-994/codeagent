/**
 * 会话线程 API
 */
import { request } from './client.js'

export function fetchThreads() {
  return request('GET', '/threads')
}

export function fetchThreadState(threadId) {
  return request('GET', `/threads/${threadId}/state`)
}
