<template>
  <el-dialog v-model="visible" width="400px" :close-on-click-modal="false" center class="login-dialog">
    <template #header>
      <div style="text-align:center;font-size:20px;font-weight:600;color:#e5e7eb">👋 欢迎回来</div>
    </template>
    <el-form @submit.prevent="handleLogin" style="margin-top:20px">
      <el-form-item>
        <el-input v-model="form.username" placeholder="用户名" size="large" style="background:#374151;border-radius:8px" />
      </el-form-item>
      <el-form-item>
        <el-input v-model="form.password" type="password" placeholder="密码" show-password size="large" />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="handleLogin" :loading="loading" size="large" style="width:100%;border-radius:8px;background:#6366f1;border:none;height:44px">
          登录
        </el-button>
      </el-form-item>
    </el-form>
    <div class="test-hint">
      <p>测试账号</p>
      <code>admin / admin123</code> — VIP（可生图）
      <code>test / test123</code> — 普通用户
    </div>
    <p v-if="error" class="error">{{ error }}</p>
  </el-dialog>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { auth } from '../composables/useAuth.js'
const visible = defineModel()
const form = reactive({ username: '', password: '' })
const loading = ref(false)
const error = ref('')
async function handleLogin() {
  loading.value = true; error.value = ''
  try {
    await auth.login(form.username, form.password)
    visible.value = false
  } catch (e) { error.value = e.message || '登录失败' }
  finally { loading.value = false }
}
</script>

<style scoped>
.test-hint { margin-top:16px; text-align:center; font-size:12px; color:#6b7280; }
.test-hint p { margin-bottom:6px; color:#9ca3af; }
.test-hint code { display:block; padding:2px 8px; background:#374151; border-radius:4px; margin:2px 0; font-size:11px; color:#a5b4fc; }
.error { text-align:center; color:#f87171; font-size:13px; margin-top:10px; }
</style>
