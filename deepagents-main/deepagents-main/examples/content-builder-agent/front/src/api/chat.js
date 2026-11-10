/**
 * 聊天 SSE 流 — streamChat / resumeAgent + 通用流解析
 */
import { streamPost } from './client.js'

// ── SSE 流解析（streamChat / resumeAgent 共用）──
async function* parseSSEStream(response) {
  const reader = response.body.getReader()
  const d = new TextDecoder()
  let buf = '', eventType = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += d.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() || ''
    for (const line of lines) {
      const t = line.trim()
      if (!t) { eventType = ''; continue }
      //保存event的值
      if (t.startsWith('event:')) { eventType = t.slice(6).trim(); continue }
      //跳过event行和空行
      if (!t.startsWith('data:')) continue
      //解析event对应的data字典
      try {
        const j = JSON.parse(t.slice(5).trim())
        if (eventType === 'error') throw new Error(j.error || '服务器错误')
        if (eventType === 'tool_call' && j.tool_call) {
          yield { type: 'tool_call', tool: j.tool_call }
        }
        if (eventType === 'thinking' && j.thinking) yield { type: 'thinking', content: j.thinking }
        if (eventType === 'content' && j.content) yield { type: 'content', content: j.content }
        if (eventType === 'tool_result' && j.tool_result) {
          yield { type: 'tool_result', tool: j.tool_result.tool, content: j.tool_result.content || '' }
        }
        if (eventType === 'interrupt') {
          yield { type: 'interrupt', data: j }
          return
        }
      } catch {}
    }
  }
}

async function _throwIfError(res) {
  if (res.ok) return
  const err = await res.json().catch(() => ({}))
  const error = new Error(err.msg || err.detail || `HTTP ${res.status}`)
  error.status = res.status
  throw error
}

//*代表生成器函数
export async function* streamChat(messages, threadId, intent = 'auto', signal = null) {
  const res = await streamPost('/runs/stream', {
    input: { messages },
    config: { configurable: { thread_id: threadId, intent } },
  }, signal)
  await _throwIfError(res)
  yield* parseSSEStream(res)
}

export async function* resumeAgent(threadId, intent, action, signal = null) {
  const res = await streamPost('/runs/resume', {
    thread_id: threadId, intent, action,
  }, signal)
  await _throwIfError(res)
  yield* parseSSEStream(res)
}
