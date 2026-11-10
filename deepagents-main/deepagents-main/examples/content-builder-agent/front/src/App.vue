<template>
  <Welcome v-if="!auth.isLoggedIn" />
  <Chat v-else />
</template>

<script setup>
import { auth } from './composables/useAuth.js'
import Welcome from './views/Welcome.vue'
import Chat from './views/Chat.vue'
</script>

<style>
html,body{margin:0;padding:0;overflow:hidden;height:100%}
#app{height:100%}
</style>
