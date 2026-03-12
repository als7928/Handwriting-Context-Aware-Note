<script setup>
/**
 * ChatPanel – Gemini-style chat interface for querying PDFs.
 */
import { ref, nextTick } from 'vue'
import { marked } from 'marked'
import { sendChatMessage } from '../services/api.js'
import { logger } from '../services/logger.js'

const props = defineProps({
  documentIds: { type: Array, default: () => [] },
})

const emit = defineEmits(['highlights', 'cite-click'])

const messages = ref([])
const input = ref('')
const loading = ref(false)
const chatContainer = ref(null)

function renderMarkdown(md) {
  return marked.parse(md || '')
}

async function send() {
  const query = input.value.trim()
  if (!query || loading.value) return

  messages.value.push({ role: 'user', content: query })
  input.value = ''
  loading.value = true

  await nextTick()
  scrollToBottom()

  try {
    const response = await sendChatMessage(query, props.documentIds)
    // Sort highlights by similarity score descending
    const sortedHighlights = [...(response.highlights || [])].sort(
      (a, b) => (b.score ?? 0) - (a.score ?? 0)
    )
    logger.info('[Chat] received response, highlights:', sortedHighlights.map(h => `${Math.round((h.score ?? 0) * 100)}%`))
    messages.value.push({
      role: 'assistant',
      content: response.answer,
      highlights: sortedHighlights,
    })
    emit('highlights', sortedHighlights)
  } catch (err) {
    logger.error('[Chat] sendChatMessage failed', err)
    messages.value.push({
      role: 'assistant',
      content: '⚠️ Something went wrong. Please try again.',
      highlights: [],
    })
  } finally {
    loading.value = false
    await nextTick()
    scrollToBottom()
  }
}

function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

function onCiteClick(highlight) {
  emit('cite-click', highlight)
}
</script>

<template>
  <div class="d-flex h-100 flex-column">
    <div class="card-header">
      <h2 class="h5 mb-1">Chat</h2>
      <p class="small text-muted mb-0">
        Ask about your annotated PDFs
      </p>
    </div>

    <div ref="chatContainer" class="chat-messages flex-grow-1 overflow-auto p-3 d-flex flex-column gap-3">
      <div v-if="messages.length === 0" class="d-flex h-100 align-items-center justify-content-center">
        <div class="text-center text-muted">
          <svg class="mb-2" width="44" height="44" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
          <p class="mb-1">Ask a question about your documents</p>
          <p class="small mb-0">e.g. "Show me the part I underlined"</p>
        </div>
      </div>

      <div
        v-for="(msg, i) in messages"
        :key="i"
        class="px-3 py-2"
        :class="msg.role === 'user' ? 'bubble-user' : 'bubble-assistant'"
      >
        <div v-if="msg.role === 'assistant'">
          <div v-html="renderMarkdown(msg.content)" />
          <!-- Source citations -->
          <div v-if="msg.highlights && msg.highlights.length" class="mt-2 border-top pt-2">
            <p class="small text-muted mb-1 fw-semibold">Sources</p>
            <div class="d-flex flex-column gap-1">
              <button
                v-for="(hl, idx) in msg.highlights"
                :key="idx"
                class="btn btn-sm btn-outline-primary text-start"
                style="font-size: 0.78rem"
                @click="onCiteClick(hl)"
              >
                <span class="badge bg-primary me-1">{{ Math.round((hl.score ?? 0) * 100) }}%</span>
                [{{ idx + 1 }}]{{ hl.filename ? ' ' + hl.filename + ' ·' : '' }} Page {{ hl.page_no }}{{ hl.text ? ' — ' + hl.text.slice(0, 60) + (hl.text.length > 60 ? '…' : '') : '' }}
              </button>
            </div>
          </div>
        </div>
        <span v-else>{{ msg.content }}</span>
      </div>

      <div v-if="loading" class="bubble-assistant px-3 py-2">
        <div class="spinner-border spinner-border-sm text-secondary" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
      </div>
    </div>

    <div class="card-footer">
      <div class="input-group">
        <textarea
          v-model="input"
          @keydown="handleKeydown"
          rows="1"
          placeholder="Type your question..."
          class="form-control"
        />
        <button
          @click="send"
          :disabled="loading || !input.trim()"
          class="btn btn-primary"
        >
          Send
        </button>
      </div>
    </div>
  </div>
</template>
