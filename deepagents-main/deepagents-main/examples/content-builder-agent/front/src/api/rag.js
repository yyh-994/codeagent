/**
 * 知识库（RAG）API
 */
import { request } from './client.js'

export function uploadFile(formData) {
  return request('POST', '/rag/upload', formData, true)
}

export function deleteDoc(md5) {
  return request('DELETE', `/rag/docs/${md5}`)
}
