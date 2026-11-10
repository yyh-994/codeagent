/**
 * 线程管理 — 创建、切换、删除、自动标题、localStorage 持久化
 */
import { marked } from 'marked'
import { auth } from './useAuth.js'
import { fetchThreads, fetchThreadState } from '../api/threads.js'

export function useThreads({ threadId, messages, streaming, streamThinking, threads, runningThreads, timer, elapsed, emit, scrollDown }) {
  const THREADS_KEY = 'cb_threads_v4_'

  // ══════════════════════════════════════════════
  // localStorage 存取
  // ══════════════════════════════════════════════
  function restoreMessages() {
    try { messages.value = JSON.parse(localStorage.getItem(threadId.value)) || [] }
    catch { messages.value = [] }
  }

  function persistMessages() {
    localStorage.setItem(threadId.value, JSON.stringify(messages.value))
  }

  function _persistThreads() {
    localStorage.setItem(THREADS_KEY + auth.username, JSON.stringify(threads.value))
  }

  function restoreThreads() {
    try {
      const r = localStorage.getItem(THREADS_KEY + auth.username)
      if (r) threads.value = JSON.parse(r)
    } catch {}
  }

  // ══════════════════════════════════════════════
  // 线程 CRUD
  // ══════════════════════════════════════════════
  function newThread() {
    threadId.value = 'th-' + Math.random().toString(36).slice(2, 10)
    messages.value = []
  }

  async function loadThreads() {
    try {
      const d = await fetchThreads()
      for (const t of (d || [])) {
        if (!threads.value.find(x => x.id === t.thread_id)) {
          threads.value.unshift({ id: t.thread_id, title: t.title || _extractTitle(t) })
        }
      }
      _persistThreads()
    } catch (e) {
      if (e.status === 401) { auth.logout(); emit('logout') }
    }
  }

  function _extractTitle(t) {
    const c = t.values?.messages?.[0]?.content
    if (!c) return null
    return (typeof c === 'string' ? c : String(c)).slice(0, 20) + (String(c).length > 20 ? '...' : '')
  }

  async function switchThread(id) {
    threadId.value = id
    if (runningThreads.has(id)) {
      const full = localStorage.getItem(id)
      messages.value = full ? (() => { try { return JSON.parse(full).filter(m => m.role === 'user') } catch { return [] } })() : []
      if (!full) restoreMessages()
      streaming.value = true; elapsed.value = 0
      if (timer.t) clearInterval(timer.t)
      timer.t = setInterval(() => elapsed.value++, 1000)

      try {
        const raw = localStorage.getItem(id + '_stream')
        if (raw) {
          const { steps = [], txt = '' } = JSON.parse(raw)
          streamThinking.value = [...steps, ...(txt ? [{ text: txt, tool: null }] : [])]
        }
      } catch { streamThinking.value = [] }
      scrollDown()
    } else {
      restoreMessages()
      streaming.value = false; streamThinking.value = []
      if (timer.t) { clearInterval(timer.t); timer.t = null }
      if (!messages.value.length) {
        try {
          const s = await fetchThreadState(id)
          if (s?.values?.messages) { messages.value = s.values.messages.map(_formatMsg); persistMessages() }
        } catch (e) {
          if (e.status === 401) { auth.logout(); emit('logout') }
        }
      }
      scrollDown()
    }
  }

  //发送消息后，确保侧边栏列表里能看到当前对话
  function ensureThreadInList(msg) {
    // ① 截前 20 个字（中文/英文都算）做标题，超长补 "..."
    const tt = msg.slice(0, 20) + (msg.length > 20 ? '...' : '')
    // ② 查找当前 threadId 是否已在列表中
    const ex = threads.value.find(t => t.id === threadId.value)
    if (!ex) {
      // ③ 不在？说明是新对话，插入列表最前面
      threads.value.unshift({ id: threadId.value, title: tt })
      _persistThreads()
    } else if (!ex.title || ex.title.startsWith('📄')) {
      // ④ 在，但标题是空的/占位的 → 补上标题
      ex.title = tt
      _persistThreads()
    }
  }

  function autoTitle() {
    const ex = threads.value.find(t => t.id === threadId.value)
    if (!ex || (ex.title && !ex.title.startsWith('th-'))) return
    const firstUser = messages.value.find(m => m.role === 'user')
    if (!firstUser) return
    ex.title = firstUser.html.replace(/<[^>]*>/g, '').slice(0, 20) + '...'
    _persistThreads()
  }

  // ══════════════════════════════════════════════
  // 内部工具
  // ══════════════════════════════════════════════
  function _formatMsg(m) {
    if (m.type === 'human' || m.type === 'user') return { role: 'user', html: _escapeHtml(m.content || ''), thinking: [] }
    if (m.type === 'ai') return { role: 'agent', html: marked.parse(m.content || ''), thinking: [] }
    return { role: 'agent', html: '', thinking: [] }
  }

  function _escapeHtml(t) {
    return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  }

  return {
    restoreMessages, persistMessages, restoreThreads,
    newThread, loadThreads, switchThread,
    ensureThreadInList, autoTitle,
  }
}
