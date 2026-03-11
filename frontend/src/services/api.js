/**
 * API service – centralised HTTP calls to the FastAPI backend.
 */

import axios from 'axios'
import { logger } from './logger.js'

const api = axios.create({ baseURL: '/api' })

api.interceptors.request.use(config => {
  logger.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`)
  return config
})

api.interceptors.response.use(
  res => {
    logger.debug(`[API] ${res.status} ${res.config.url}`)
    return res
  },
  err => {
    logger.error(`[API] request failed: ${err.config?.url}`, err.message)
    return Promise.reject(err)
  }
)

export async function uploadDocument(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/documents/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function listDocuments() {
  const { data } = await api.get('/documents/')
  return data
}

export async function deleteDocument(id) {
  await api.delete(`/documents/${id}`)
}

export function getDocumentFileUrl(id) {
  return `/api/documents/${id}/file`
}

export async function sendChatMessage(query, documentIds = []) {
  const { data } = await api.post('/chat/', {
    query,
    document_ids: documentIds,
  })
  return data
}
