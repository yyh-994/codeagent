<template>
  <div class="kb-panel">
    <div class="kb-header">
      <h3>📚 知识库管理</h3>
      <input ref="fileInput" type="file" accept=".pdf" multiple style="display:none" @change="handleUpload" />
      <el-button type="primary" size="small" @click="fileInput.click()" :loading="uploading">
        📎 上传 PDF
      </el-button>
    </div>

    <div v-if="uploading" class="upload-bar">⏳ 索引中 ({{ doneCount }}/{{ totalCount }}): {{ currentName }}...</div>

    <div class="kb-list">
      <div v-for="doc in docs" :key="doc.md5" class="kb-item">
        <span class="kb-icon">📄</span>
        <span class="kb-name">{{ doc.name }}</span>
        <el-button size="small" text type="danger" @click="handleDelete(doc.md5, doc.name)">删除</el-button>
      </div>
      <div v-if="!docs.length" class="kb-empty">暂无文档，上传 PDF 开始构建知识库</div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { uploadFile, deleteDoc } from '../api/rag.js'
import { auth } from '../composables/useAuth.js'

const fileInput = ref(null)
const docs = ref([])
const uploading = ref(false)
const currentName = ref('')
const doneCount = ref(0)
const totalCount = ref(0)

async function handleUpload(e) {
  const files = e.target.files; if (!files.length) return
  uploading.value = true; totalCount.value = files.length; doneCount.value = 0

  for (const file of files) {
    currentName.value = file.name
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await uploadFile(fd)
      docs.value.push({ md5: res.md5, name: res.name })
      doneCount.value++
      ElMessage.success(`${file.name} 索引完成 (${doneCount.value}/${totalCount.value})`)
    } catch (ex) {
      if (ex.status === 401) { auth.logout(); return }
      ElMessage.error(`${file.name} 上传失败: ${ex.message}`)
    }
  }
  uploading.value = false; currentName.value = ''; e.target.value = ''
}

async function handleDelete(md5, name) {
  try {
    await deleteDoc(md5)
    docs.value = docs.value.filter(d => d.md5 !== md5)
    ElMessage.success(`已删除 ${name}`)
  } catch (ex) {
      if (ex.status === 401) { auth.logout(); return }
      ElMessage.error('删除失败: ' + ex.message)
    }
}
</script>

<style scoped>
.kb-panel{background:#1f2937;border-radius:12px;padding:16px;margin:12px 0}
.kb-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.kb-header h3{margin:0;color:#e5e7eb;font-size:15px}
.kb-list{max-height:240px;overflow-y:auto}
.kb-item{display:flex;align-items:center;padding:8px 10px;border-radius:6px;margin-bottom:4px;background:#374151;color:#d1d5db;font-size:13px}
.kb-icon{margin-right:8px}
.kb-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kb-empty{text-align:center;color:#6b7280;padding:20px;font-size:13px}
.upload-bar{color:#fbbf24;font-size:12px;padding:8px 0}
</style>
