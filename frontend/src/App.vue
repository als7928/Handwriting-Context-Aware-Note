<script setup>
/**
 * Root layout – sidebar | PDF viewer | chat panel (Gemini-style).
 */
import { ref } from 'vue'
import DocumentSidebar from './components/DocumentSidebar.vue'
import PdfViewer from './components/PdfViewer.vue'
import ChatPanel from './components/ChatPanel.vue'

const selectedDocumentId = ref(null)
const highlights = ref([])
const jumpHighlight = ref(null)

function onSelectDocument(id) {
  selectedDocumentId.value = id
  highlights.value = []
  jumpHighlight.value = null
}

function onHighlights(hl) {
  highlights.value = hl

  // Auto-select the first highlighted document if none is selected
  if (!selectedDocumentId.value && hl.length > 0) {
    selectedDocumentId.value = hl[0].document_id
  }
}

function onCiteClick(highlight) {
  // Switch to the cited document and jump to the cited location
  selectedDocumentId.value = highlight.document_id
  highlights.value = [highlight]
  jumpHighlight.value = highlight
}
</script>

<template>
  <div class="container-fluid py-3 app-shell">
    <div class="row h-100 g-3">
      <aside class="col-12 col-lg-3 h-100">
        <div class="card h-100 sidebar-panel">
          <DocumentSidebar @select="onSelectDocument" />
        </div>
      </aside>

      <main class="col-12 col-lg-6 h-100">
        <div class="card h-100 viewer-panel">
          <PdfViewer :document-id="selectedDocumentId" :highlights="highlights" :jump-highlight="jumpHighlight" />
        </div>
      </main>

      <aside class="col-12 col-lg-3 h-100">
        <div class="card h-100 chat-panel">
          <ChatPanel :document-ids="[]" @highlights="onHighlights" @cite-click="onCiteClick" />
        </div>
      </aside>
    </div>
  </div>
</template>
