<template>
  <div class="welcome">
    <div class="hero">
      <div class="logo">🤖</div>
      <h1>Content Builder Agent</h1>
      <p>AI 驱动的内容创作助手 — 写博客、社交媒体、邮件通讯，一键生成封面图</p>
      <div class="features">
        <div class="card"><span class="card-icon">📝</span><strong>博客写作</strong><span>研究 → 写作 → 封面图</span></div>
        <div class="card"><span class="card-icon">📱</span><strong>社交媒体</strong><span>平台适配自动优化</span></div>
        <div class="card"><span class="card-icon">📧</span><strong>邮件通讯</strong><span>Newsletter 一步到位</span></div>
      </div>
      <el-button class="start-btn" size="large" @click="showLogin = true">开始对话</el-button>
    </div>
    <LoginDialog v-model="showLogin" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import LoginDialog from '../components/LoginDialog.vue'
const showLogin = ref(false)
</script>

<style scoped>
.welcome { display:flex; align-items:center; justify-content:center; min-height:100vh; background:#111827; }
.hero { text-align:center; max-width:600px; padding:40px; }
.logo { font-size:56px; margin-bottom:16px; }
.hero h1 { font-size:32px; color:#e5e7eb; margin-bottom:12px; font-weight:600; }
.hero p { font-size:16px; color:#6b7280; margin-bottom:40px; }
.features { display:flex; gap:16px; margin-bottom:40px; justify-content:center; flex-wrap:wrap; }
.card { background:#1f2937; border:1px solid #374151; border-radius:12px; padding:20px 16px; width:170px; display:flex; flex-direction:column; align-items:center; gap:8px; }
.card-icon { font-size:24px; }
.card strong { color:#e5e7eb; font-size:14px; }
.card span { color:#6b7280; font-size:12px; }
.start-btn { background:#6366f1; border:none; color:#fff; border-radius:8px; padding:12px 40px; font-size:15px; font-weight:500; }
.start-btn:hover { background:#5558e6 !important; }
</style>
