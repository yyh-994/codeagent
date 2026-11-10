<template>
  <div class="layout">
    <aside class="sidebar">
      <div class="sidebar-top">
        <el-button type="primary" @click="newThread" class="new-chat-btn" size="large">
          <span style="font-size:16px">+</span> 新对话
        </el-button>
      </div>
      <div class="thread-list">
        <div
          v-for="t in threads"
          :key="t.id"
          :class="['thread-item', { active: t.id === threadId }]"
          @click="switchThread(t.id)"
        >
          <span class="thread-icon">💬</span>
          <span class="thread-title">{{ t.title || '新对话' }}</span>
        </div>
      </div>
      <div class="sidebar-bottom">
        <div class="user-info">
          <span class="user-avatar">{{ auth.username[0]?.toUpperCase() }}</span>
          <span class="user-name">{{ auth.username }}</span>
        </div>
        <el-button text @click="handleLogout" style="color:#a0aec0">退出</el-button>
      </div>
    </aside>

    <div class="chat-container">
      <div class="chat-messages" ref="msgBox">
        <div class="chat-messages-inner">
          <div v-if="messages.length === 0 && !streaming" class="empty-state">
            <div class="empty-icon">🤖</div>
            <h2>Content Builder Agent</h2>
            <p>AI 驱动的内容创作助手</p>
          </div>

          <div v-if="streaming" class="stream-zone">
            <div class="think-wrap open">
              <div class="think-bar">
                ▼ 思考中<span class="dots-anim"></span>
                <span class="elapsed">(已用 {{ elapsed }}s)</span>
                <el-button size="small" type="danger" text @click="stopStream" style="margin-left:auto">⏹ 停止</el-button>
              </div>
              <div class="think-list">
                <div v-for="(s, j) in streamThinking" :key="j" class="think-step-row">
                  <div v-if="s.tool" class="tool-card" @click="toggleToolDetail(j)">
                    <span class="tool-name">{{ s.tool }}</span>
                    <span v-if="s.detail" class="tool-detail">{{ s.detail }}</span>
                    <span class="tool-arrow">{{ expandedTools.has(j) ? '▼' : '▶' }}</span>
                  </div>
                  <span v-if="s.text" class="think-line">{{ s.text }}</span>
                </div>
              </div>
            </div>
          </div>

          <div v-for="(m, i) in messages" :key="i" :class="['msg-row', m.role]">
            <div v-if="m.role === 'agent'" class="msg-avatar">🤖</div>
            <div class="msg-body">
              <div v-if="m.thinking?.length" class="think-wrap">
                <div class="think-bar" @click="toggleThink(i)">
                  <span class="think-arrow">{{ expandedThinking.has(i) ? '▼' : '▶' }}</span>
                  <span>思考过程 ({{ m.thinking.length }} 步)</span>
                </div>
                <div v-if="expandedThinking.has(i)" class="think-list">
                  <div v-for="(t, j) in m.thinking" :key="j" class="think-step">• {{ t }}</div>
                </div>
              </div>
              <div class="msg-content markdown" v-html="m.html"></div>
              <div v-if="m.role === 'agent'" class="msg-actions">
                <el-button v-if="m._stopped" size="small" text type="warning" @click="continueStream(i)">▶ 继续生成</el-button>
                <el-button size="small" text @click="branchFrom(i)">🔄 重新生成</el-button>
                <el-button size="small" text @click="copyContent(m.html)">📋 复制</el-button>
              </div>
            </div>
            <div v-if="m.role === 'user'" class="msg-avatar user-avatar-msg">{{ auth.username[0]?.toUpperCase() }}</div>
          </div>

        </div>
      </div>

      <footer class="chat-input">
        <div class="chat-input-inner">
          <div class="mode-tags">
            <span
              v-for="m in modes"
              :key="m.value"
              :class="['mode-tag', { active: agentMode === m.value }]"
              @click="switchMode(m.value)"
            >{{ m.label }}</span>
          </div>

          <KnowledgeBase v-if="showKB" />

          <div class="input-row">
            <el-input
              v-model="input" placeholder="输入需求..." @keyup.enter="send"
              :disabled="streaming" size="large" class="msg-input"
              :autosize="{ minRows: 2, maxRows: 6 }" type="textarea"
            />
            <el-button
              type="primary" @click="send" :loading="streaming"
              :disabled="streaming || !input.trim()" class="send-btn" size="large"
            >发送</el-button>
          </div>

          <div v-if="hitlWaiting" style="background:#fdf6ec;border:1px solid #e6a23c;border-radius:8px;padding:12px;margin-top:8px">
            <div style="font-weight:bold;color:#e6a23c;margin-bottom:8px">⏸ 等待确认 — 确认发布以下内容？</div>
            <div v-if="hitlData" style="font-size:13px;color:#606266;max-height:150px;overflow-y:auto;white-space:pre-wrap;background:#fff;border-radius:4px;padding:8px;margin-bottom:8px">{{ hitlData }}</div>
            <div style="display:flex;gap:8px;align-items:center">
              <el-button type="success" size="small" @click="handleHitlAction('approve')">✓ 确认执行</el-button>
              <el-button type="warning" size="small" @click="handleHitlAction('reject')">✗ 拒绝</el-button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { auth } from '../composables/useAuth.js'
import KnowledgeBase from '../components/KnowledgeBase.vue'
import { useThreads } from '../composables/useThreads.js'
import { useChat } from '../composables/useChat.js'

// ── 常量 ──
const modes = [
  { label: '🧠 自动', value: 'auto' },
  { label: '📄 知识库问答', value: 'rag' },
  { label: '📝 写博客', value: 'blog' },
  { label: '📱 社交媒体', value: 'social' },
]

// ── 响应式状态 ──
const emit = defineEmits(['logout'])
const messages = ref([])
const input = ref('')
const streaming = ref(false)
const streamThinking = ref([])
const threadId = ref(localStorage.getItem('curThread') || 'th-' + Math.random().toString(36).slice(2, 10))
const msgBox = ref(null)
const threads = ref([])
const elapsed = ref(0)
const agentMode = ref('auto')
const showKB = ref(false)
const hitlWaiting = ref(false)
const hitlData = ref('')

// ── 非响应式 ──
const runningThreads = new Set()
const expandedThinking = new Set()
const expandedTools = new Set()

// ── 工具函数 ──
function scrollDown() {
  //nextTick就是等前端dom渲染完之后，再执行
  nextTick(() => {
    //msgBox.value = <div> DOM 对象
    const e = msgBox.value
    if (e) e.scrollTop = e.scrollHeight
  })
}

function escapeHtml(t) {
  return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function fixImagePaths(html) {
  html = html.replace(/src="\.?\/?output\//g, 'src="http://127.0.0.1:2024/output/')
  html = html.replace(/src="output\//g, 'src="http://127.0.0.1:2024/output/')
  html = html.replace(
    /(https?:\/\/127\.0\.0\.1:2024)?\/?(output\/[^\s<>]*\.(png|jpg|jpeg|gif))/gi,
    '<br><img src="http://127.0.0.1:2024/$2" style="max-width:100%;border-radius:8px;margin:8px 0"><br>'
  )
  html = html.replace(/&quot;\s+alt=/g, '" alt=')
  html = html.replace(/""\s+alt=/g, '" alt=')
  return html
}

function toggleThink(i) {
  if (expandedThinking.has(i)) expandedThinking.delete(i)
  else expandedThinking.add(i)
  messages.value = [...messages.value]
}

function toggleToolDetail(j) {
  if (expandedTools.has(j)) expandedTools.delete(j)
  else expandedTools.add(j)
  streamThinking.value = [...streamThinking.value]
}

function switchMode(mode) {
  agentMode.value = mode
  showKB.value = mode === 'rag' ? !showKB.value : false
}

async function copyContent(html) {
  const txt = html.replace(/<[^>]*>/g, '')
  try { await navigator.clipboard.writeText(txt); ElMessage.success('已复制到剪贴板') }
  catch { ElMessage.warning('复制失败') }
}

function handleLogout() { auth.logout(); emit('logout') }

// ── Composables ──
const threadsModule = useThreads({
  threadId, messages, streaming, streamThinking, threads, runningThreads, timer: { t: null }, elapsed, emit, scrollDown,
})

const chatModule = useChat({
  input, messages, streaming, streamThinking, elapsed, threadId, agentMode,
  runningThreads,
  hitlWaiting, hitlData,
  persistMessages: threadsModule.persistMessages,
  ensureThreadInList: threadsModule.ensureThreadInList,
  loadThreads: threadsModule.loadThreads,
  scrollDown, fixImagePaths, escapeHtml,
  emit,
})

// ── 解构出来给模板用 ──
const { restoreMessages, restoreThreads, newThread, loadThreads, switchThread: _switch } = threadsModule
const switchThread = (id) => { if (hitlWaiting.value) { ElMessage.warning('请先完成或取消当前确认操作'); return } _switch(id) }
const { send, handleHitlAction, stopStream, continueStream, branchFrom } = chatModule

// ── 生命周期 ──
onMounted(() => {
  restoreMessages()
  restoreThreads()
  loadThreads()
})

watch(threadId, (id) => {
  localStorage.setItem('curThread', id)
  restoreMessages()
})
</script>

<style scoped>
.layout{display:flex;height:100vh;background:#111827;color:#e5e7eb;overflow:hidden}
.sidebar{width:260px;background:#1f2937;display:flex;flex-direction:column;border-right:1px solid #374151;flex-shrink:0}
.sidebar-top{padding:14px}
.new-chat-btn{width:100%;border-radius:8px;background:#374151;border:none;color:#e5e7eb;font-weight:500}
.thread-list{flex:1;overflow-y:auto;padding:6px 10px;scrollbar-width:thin;scrollbar-color:#374151 transparent}
.thread-list::-webkit-scrollbar{width:5px}
.thread-list::-webkit-scrollbar-track{background:transparent}
.thread-list::-webkit-scrollbar-thumb{background:#374151;border-radius:10px}
.thread-list::-webkit-scrollbar-button{display:none}
.thread-item{display:flex;align-items:center;padding:8px 10px;border-radius:8px;cursor:pointer;font-size:13px;color:#9ca3af;margin-bottom:2px}
.thread-item:hover{background:#374151;color:#e5e7eb}
.thread-item.active{background:#374151;color:#fff}
.thread-icon{margin-right:8px;font-size:14px}
.thread-title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sidebar-bottom{padding:14px;border-top:1px solid #374151;display:flex;align-items:center;justify-content:space-between}
.user-info{display:flex;align-items:center;gap:8px}
.user-avatar{width:28px;height:28px;border-radius:50%;background:#6366f1;display:flex;align-items:center;justify-content:center;font-size:13px;color:#fff}
.user-avatar-msg{background:#6366f1;color:#fff;font-size:12px;font-weight:600}
.user-name{font-size:13px;color:#d1d5db}
.chat-container{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}
.chat-messages{flex:1;overflow-y:auto;padding:20px 0;scrollbar-width:thin;scrollbar-color:#374151 transparent}
.chat-messages::-webkit-scrollbar{width:5px}
.chat-messages::-webkit-scrollbar-track{background:transparent}
.chat-messages::-webkit-scrollbar-thumb{background:#374151;border-radius:10px}
.chat-messages::-webkit-scrollbar-button{display:none}
.chat-messages-inner{max-width:960px;margin:0 auto;padding:0 24px}
.empty-state{text-align:center;margin-top:100px}
.empty-icon{font-size:48px;margin-bottom:12px}
.empty-state h2{color:#e5e7eb;font-size:22px;margin:10px 0}
.empty-state p{color:#6b7280;font-size:15px}
.msg-row{display:flex;margin-bottom:22px;gap:12px}
.msg-row.user{flex-direction:row-reverse}
.msg-avatar{width:32px;height:32px;border-radius:50%;background:#374151;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
.msg-body{max-width:80%;min-width:0}
.msg-content{padding:10px 14px;border-radius:12px;line-height:1.7;font-size:14px;word-break:break-word}
.msg-row.user .msg-content{background:#6366f1;color:#fff;border-radius:12px 4px 12px 12px}
.msg-row.agent .msg-content{background:#1f2937;color:#e5e7eb;border:1px solid #374151;border-radius:4px 12px 12px 12px}
.msg-actions{margin-top:4px;opacity:0;transition:opacity .15s}
.msg-row:hover .msg-actions{opacity:1}
.think-wrap{margin-bottom:8px}
.think-bar{padding:6px 12px;background:#1a1f2e;border-radius:6px;font-size:12px;color:#9ca3af;display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.think-wrap.open .think-bar{border-radius:6px 6px 0 0}
.think-arrow{font-size:10px}
.think-list{background:#1a1f2e;border-radius:0 0 6px 6px;padding:6px 12px 8px 18px;font-size:12px;color:#6b7280;line-height:1.8;min-height:80px;max-height:360px;overflow-y:auto}
.think-step{padding:2px 0}
.elapsed{color:#6b7280;font-size:11px;margin-left:auto}
.think-line{color:#9ca3af;line-height:1.6;display:block}
.think-step-row{padding:3px 0}
.tool-card{display:flex;align-items:center;gap:8px;padding:4px 8px;background:#374151;border-radius:6px;cursor:pointer;font-size:11px;color:#d1d5db}
.tool-card:hover{background:#4b5563}
.tool-name{color:#a5b4fc;font-weight:500}
.tool-detail{color:#9ca3af;font-size:10px;overflow:hidden;text-overflow:ellipsis}
.tool-arrow{font-size:9px;margin-left:auto}
.chat-input{padding:14px 0;border-top:1px solid #374151;background:#111827}
.chat-input-inner{max-width:960px;margin:0 auto;padding:0 24px}
.input-row{display:flex;gap:10px;align-items:flex-end}
.msg-input{flex:1}
.msg-input :deep(.el-textarea__inner){background:#1f2937;border:1px solid #374151;border-radius:12px;color:#e5e7eb;padding:10px 14px;font-size:14px;resize:none}
.msg-input :deep(.el-textarea__inner):focus{border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,.2)}
.send-btn{border-radius:8px;background:#6366f1;border:none;padding:6px 22px}
.send-btn:hover{background:#5558e6!important}
.mode-tags{display:flex;gap:6px;flex-wrap:wrap}
.mode-tag{padding:4px 14px;border-radius:14px;font-size:13px;background:#1f2937;color:#9ca3af;cursor:pointer;border:1px solid #374151;white-space:nowrap}
.mode-tag:hover{background:#374151;color:#d1d5db}
.mode-tag.active{background:#6366f1;color:#fff;border-color:#6366f1}
.mode-tag.upload{background:transparent;border:1px dashed #374151}
.dots-anim::after{content:'';animation:dots 1.2s steps(3,end) infinite}
@keyframes dots{0%{content:''}33%{content:'.'}66%{content:'..'}100%{content:'...'}}
.stream-zone{margin-bottom:16px}
</style>

<style>
.markdown h2,.markdown h3{margin:10px 0 5px;color:#a5b4fc}
.markdown table{border-collapse:collapse;margin:8px 0;font-size:13px;width:100%}
.markdown td,.markdown th{border:1px solid #374151;padding:6px 10px;color:#e5e7eb}
.markdown th{background:#1f2937}
.markdown pre{background:#1f2937;padding:10px;border-radius:6px;overflow-x:auto;font-size:12px;border:1px solid #374151}
.markdown code{background:#374151;padding:2px 6px;border-radius:3px;font-size:12px;color:#e5e7eb}
.markdown blockquote{border-left:3px solid #6366f1;padding-left:12px;color:#9ca3af;margin:8px 0}
.markdown ul,.markdown ol{padding-left:20px}
.markdown li{margin:4px 0}
.markdown p{margin:6px 0}
.markdown hr{border:none;border-top:1px solid #374151;margin:12px 0}
.markdown img{max-width:100%;border-radius:8px;margin:8px 0}
</style>
