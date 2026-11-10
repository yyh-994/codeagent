/**
 * 消息发送 + HITL — SSE 流式对话核心逻辑
 */
import { marked } from 'marked'
import { auth } from './useAuth.js'
import { streamChat, resumeAgent } from '../api/chat.js'

export function useChat({
  input, messages, streaming, streamThinking, elapsed, threadId, agentMode,
  runningThreads,
  hitlWaiting, hitlData,
  persistMessages, ensureThreadInList, loadThreads,
  scrollDown, fixImagePaths, escapeHtml, emit,
}) {
  const timer = { t: null }
  let abortController = null

  const _renderHtml = (md) => {
    let fh = marked.parse(md) || ''
    return fixImagePaths(fh
      .replace(/src="\.?\/?output\//g, 'src="http://127.0.0.1:2024/output/')
      .replace(/src="output\//g, 'src="http://127.0.0.1:2024/output/'))
  }

  // SSE 流处理（send / handleHitlAction 共用）
  //content 事件，进入messages的html；
  // thinking 事件、tool_call 事件、tool_result 事件进入streamThinking和messages的thinking
  async function _runSSE(gen, tid, localMsgs, intent) {
    let steps = [], txt = ''

    const _savePartial = () => {
      const tns = steps.filter(s => s.tool).map(s => s.tool)
      const all = steps.filter(s => s.text).map(s => s.text).join('\n\n') + (txt ? '\n\n' + txt : '')
      const arr = [...localMsgs, { role: 'agent', html: _renderHtml(all), thinking: tns }]
      localStorage.setItem(tid, JSON.stringify(arr))
      localStorage.setItem(tid + '_stream', JSON.stringify({ steps, txt }))
    }

    // 把 content（正式回复）+ 当前 tool 列表一起推到对话气泡实时渲染
    const _refreshLive = () => {
      if (threadId.value !== tid || !txt.trim() && !steps.some(s => s.tool)) return
      const tns = steps.filter(s => s.tool).map(s => s.tool)
      const liveMsg = { role: 'agent', html: _renderHtml(txt), thinking: tns, _streaming: true }
      const arr = [...messages.value]
      const i = arr.findIndex(m => m._streaming)
      if (i >= 0) arr[i] = liveMsg
      else arr.push(liveMsg)
      messages.value = arr
    }

    try {
      for await (const c of gen) {
        if (c.type === 'tool_call') {
          if (txt) { steps.push({ text: txt, tool: null, _type: 'content' }); txt = '' }
          steps.push({ text: null, tool: c.tool })
        } else if (c.type === 'content') {
          txt += c.content
          _refreshLive()
        } else if (c.type === 'thinking') {
          steps.push({ text: c.content, tool: null, _type: 'thinking' })
        } else if (c.type === 'tool_result') {
          steps.push({ text: `🔧 ${c.tool} 完成`, tool: c.tool, _type: 'tool_result' })
        } else if (c.type === 'interrupt') {
          if (txt) { steps.push({ text: txt, tool: null, _type: 'content' }); txt = '' }
          steps.push({ text: '⏸ 等待确认操作...', tool: null })
          hitlWaiting.value = true
          hitlData.value = c.data?.action_requests?.[0]?.tool_call?.args?.content || '等待确认操作'
          if (threadId.value === tid) streamThinking.value = [...steps]
          _savePartial()
          return { interrupted: true, steps, txt }
        }
        const disp = txt ? [...steps, { text: txt, tool: null, _type: 'content' }] : [...steps]
        if (threadId.value === tid) { streamThinking.value = disp; scrollDown() }
        _savePartial()
      }
      return { interrupted: false, steps, txt }
    } catch (e) {
      if (e.name === 'AbortError') return { aborted: true }
      if (e.status === 401) { auth.logout(); emit('logout'); return { aborted: true } }
      return { error: '⚠️ ' + escapeHtml(e.message || '连接中断') + ' — 请重试' }
    }
  }

  // 结束直播：清空流式思考区域。（send / handleHitlAction / abort 共用）
  function _finalizeLiveMsg(tid, { stopped = false } = {}) {
    if (threadId.value === tid) {
      const arr = [...messages.value]
      const liveIdx = arr.findIndex(m => m._streaming)
      if (liveIdx >= 0) {
        arr[liveIdx] = { ...arr[liveIdx], _streaming: undefined, _showThinking: false, _stopped: stopped }
        messages.value = arr
      }
      streamThinking.value = []
      streaming.value = false
    }
  }

  function _doCleanup(tid) {
    runningThreads.delete(tid)
    abortController = null
    if (timer.t) { clearInterval(timer.t); timer.t = null }
  }

  // 流结束 → 收尾（send / handleHitlAction 共用）
  function _saveResult(tid) {
    _finalizeLiveMsg(tid)
    _doCleanup(tid)
    persistMessages()
    loadThreads()
    scrollDown()
  }

  // 发送消息
  async function send({ skipUserBubble = false } = {}) {
    const msg = input.value.trim()
    if (!msg || streaming.value) return

    // ── ❶ 秒更 UI（用户气泡立即上屏）─────────
    input.value = ''
    if (!skipUserBubble) {
      messages.value.push({ role: 'user', html: escapeHtml(msg), thinking: [] })
    }
    streaming.value = true
    streamThinking.value = []
    elapsed.value = 0
    timer.t = setInterval(() => elapsed.value++, 1000)
    scrollDown()

    // ── ❷ 侧边栏即时更新 ──────────────────────
    const myTid = threadId.value
    ensureThreadInList(msg)           // 新对话立刻出现在侧边栏
    runningThreads.add(myTid)         // 侧边栏显示"运行中"状态

    // ── ❸ 持久化（localStorage，刷新恢复用）────
    persistMessages()

    // ── ❹ 发后端 → 流式消费 ──────────────────
    const intent = agentMode.value
    abortController = new AbortController()
    const localMsgs = [...messages.value]
    const gen = streamChat([{ role: 'user', content: msg }], myTid, intent, abortController.signal)
    const res = await _runSSE(gen, myTid, localMsgs, intent)

    // ── ❺ 异常处理 ────────────────────────────
    if (res.interrupted) return   // HITL 暂停，_streaming 预览保留供恢复
    if (res.aborted) {
      _finalizeLiveMsg(myTid, { stopped: true })
      _doCleanup(myTid)
      persistMessages()
      return
    }
    if (res.error) {
      if (threadId.value === myTid) {
        const arr = [...messages.value]
        const liveIdx = arr.findIndex(m => m._streaming)
        const errMsg = { role: 'agent', html: res.error, thinking: [], _stopped: false }
        if (liveIdx >= 0) arr[liveIdx] = errMsg
        else arr.push(errMsg)
        messages.value = arr
        streaming.value = false
        streamThinking.value = []
      }
      _doCleanup(myTid)
      persistMessages()
      return
    }

    // ── ❻ 正常收尾 ────────────────────────────
    _saveResult(myTid)
  }


  async function handleHitlAction(action) {
    if (!hitlWaiting.value) return
    const tid = threadId.value; const intent = agentMode.value
    hitlWaiting.value = false; hitlData.value = ''

    if (action === 'reject') {
      // 拒绝：通知后端后直接结束，不重新生成
      abortController = new AbortController()
      try { for await (const _ of resumeAgent(tid, intent, 'reject', abortController.signal)) {} } catch {}
      _finalizeLiveMsg(tid, { stopped: true })
      _doCleanup(tid)
      persistMessages()
      loadThreads()
      return
    }

    // 确认：恢复执行
    let steps = [...streamThinking.value], txt = ''
    const st = localStorage.getItem(tid + '_stream')
    if (st) { const p = JSON.parse(st); steps = p.steps || []; txt = p.txt || '' }
    if (!steps.filter(s => s.text).length && !txt) txt = ''

    abortController = new AbortController()
    const gen = resumeAgent(tid, intent, action, abortController.signal)
    const res = await _runSSE(gen, tid, [], intent)
    if (res.steps) steps = res.steps
    if (res.txt) txt = res.txt
    if (res.interrupted || res.aborted || res.error) return

    _saveResult(tid)
  }

  // ══════════════════════════════════════════════
  // 消息操作
  // ══════════════════════════════════════════════
  function stopStream() {
    if (abortController) { abortController.abort(); abortController = null }
  }

  async function continueStream(msgIdx) {
    const m = messages.value[msgIdx]
    if (!m || m.role !== 'agent') return
    messages.value.splice(msgIdx, 1); persistMessages(); input.value = '继续'
    await send()
  }

  async function branchFrom(msgIdx) {
    const msgs = messages.value.slice(0, msgIdx)
    const lastUser = msgs.filter(m => m.role === 'user').pop()
    if (!lastUser) return
    messages.value = msgs; persistMessages()
    input.value = lastUser.html.replace(/<[^>]*>/g, '')
    await send({ skipUserBubble: true })
  }

  return { send, handleHitlAction, stopStream, continueStream, branchFrom }
}
