/** vitest 全局 setup — 注册 ElementPlus 等全局插件 */
import { config } from '@vue/test-utils'
import ElementPlus from 'element-plus'

// 全局注册 ElementPlus（和 main.js 保持一致）
config.global.plugins = [ElementPlus]

// Mock localStorage (happy-dom 自带，但确保清空状态)
beforeEach(() => {
  localStorage.clear()
})
