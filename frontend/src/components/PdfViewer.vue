<script setup>
/**
 * PdfViewer – renders a PDF using PDF.js and overlays highlight rectangles.
 *
 * Props:
 *   documentId    – UUID of the document to display
 *   highlights    – array of { page_no, x0, y0, x1, y1 } to draw
 *   jumpHighlight – a single highlight object to jump to (cite-click)
 */
import { ref, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import { getDocumentFileUrl } from '../services/api.js'

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.mjs',
  import.meta.url,
).toString()

const props = defineProps({
  documentId: { type: String, default: null },
  highlights: { type: Array, default: () => [] },
  jumpHighlight: { type: Object, default: null },
})

const containerRef = ref(null)
const totalPages = ref(0)
const currentPage = ref(1)
const pageInput = ref(1)
const scale = ref(1.5)
const minScale = 0.5
const maxScale = 3.0
const scalePresets = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3]

let pdfDoc = null
const canvasRefs = new Map()

/* ── Load PDF ──────────────────────────────────────────────────────────── */

async function loadPdf() {
  if (!props.documentId) return

  const url = getDocumentFileUrl(props.documentId)
  const loadingTask = pdfjsLib.getDocument(url)
  pdfDoc = await loadingTask.promise
  totalPages.value = pdfDoc.numPages

  await nextTick()
  await renderAllPages()
}

async function renderAllPages() {
  if (!pdfDoc) return
  for (let i = 1; i <= pdfDoc.numPages; i++) {
    await renderPage(i)
  }
}

async function renderPage(pageNum) {
  if (!pdfDoc) return
  const page = await pdfDoc.getPage(pageNum)
  const viewport = page.getViewport({ scale: scale.value })

  const canvasId = `pdf-page-${pageNum}`
  let canvas = document.getElementById(canvasId)
  if (!canvas) return

  canvas.width = viewport.width
  canvas.height = viewport.height

  const ctx = canvas.getContext('2d')
  await page.render({ canvasContext: ctx, viewport }).promise

  canvasRefs.set(pageNum, { canvas, viewport })
}

/* ── Highlights ────────────────────────────────────────────────────────── */

const highlightStyles = ref([])

function computeHighlights() {
  const styles = []

  for (const h of props.highlights) {
    if (h.document_id !== props.documentId) continue
    const info = canvasRefs.get(h.page_no)
    if (!info) continue

    const { canvas, viewport } = info
    const sx = scale.value
    const sy = scale.value

    styles.push({
      top: `${canvas.offsetTop + h.y0 * sy}px`,
      left: `${canvas.offsetLeft + h.x0 * sx}px`,
      width: `${(h.x1 - h.x0) * sx}px`,
      height: `${(h.y1 - h.y0) * sy}px`,
      page: h.page_no,
    })
  }

  highlightStyles.value = styles
}

function scrollToFirstHighlight() {
  if (!props.highlights.length || !containerRef.value) return

  const first = props.highlights.find((h) => h.document_id === props.documentId)
  if (!first) return

  const info = canvasRefs.get(first.page_no)
  if (!info) return

  const { canvas } = info
  const scrollY = canvas.offsetTop + first.y0 * scale.value - 80
  containerRef.value.scrollTo({ top: scrollY, behavior: 'smooth' })
}

function setScale(nextScale) {
  const bounded = Math.min(maxScale, Math.max(minScale, Number(nextScale)))
  scale.value = Math.round(bounded * 100) / 100
}

function zoomIn() {
  setScale(scale.value + 0.25)
}

function zoomOut() {
  setScale(scale.value - 0.25)
}

function resetZoom() {
  setScale(1.5)
}

function onScalePresetChange(event) {
  setScale(event.target.value)
}

/* ── Page navigation ───────────────────────────────────────────────────── */

function scrollToPage(pageNum) {
  const canvas = document.getElementById(`pdf-page-${pageNum}`)
  if (!canvas || !containerRef.value) return
  containerRef.value.scrollTo({ top: canvas.offsetTop - 8, behavior: 'smooth' })
  currentPage.value = pageNum
  pageInput.value = pageNum
}

function prevPage() {
  if (currentPage.value > 1) scrollToPage(currentPage.value - 1)
}

function nextPage() {
  if (currentPage.value < totalPages.value) scrollToPage(currentPage.value + 1)
}

function onPageInputChange() {
  const n = parseInt(pageInput.value)
  if (n >= 1 && n <= totalPages.value) scrollToPage(n)
  else pageInput.value = currentPage.value
}

/* ── Keyboard shortcuts ────────────────────────────────────────────────── */

function handleKeydown(e) {
  if (!props.documentId) return
  if (e.ctrlKey && (e.key === '+' || e.key === '=')) {
    e.preventDefault()
    zoomIn()
  } else if (e.ctrlKey && e.key === '-') {
    e.preventDefault()
    zoomOut()
  } else if (e.ctrlKey && e.key === '0') {
    e.preventDefault()
    resetZoom()
  }
}

/* ── Watchers ──────────────────────────────────────────────────────────── */

watch(
  () => props.documentId,
  async () => {
    canvasRefs.clear()
    highlightStyles.value = []
    currentPage.value = 1
    pageInput.value = 1
    await loadPdf()
  },
)

watch(
  () => props.highlights,
  async () => {
    await nextTick()
    computeHighlights()
    scrollToFirstHighlight()
  },
  { deep: true },
)

watch(
  () => props.jumpHighlight,
  async (hl) => {
    if (!hl) return
    await nextTick()
    computeHighlights()
    // Scroll to the specific cited location
    const info = canvasRefs.get(hl.page_no)
    if (!info || !containerRef.value) return
    const scrollY = info.canvas.offsetTop + hl.y0 * scale.value - 80
    containerRef.value.scrollTo({ top: scrollY, behavior: 'smooth' })
    currentPage.value = hl.page_no
    pageInput.value = hl.page_no
  },
)

watch(
  () => scale.value,
  async () => {
    if (!pdfDoc) return
    await nextTick()
    await renderAllPages()
    computeHighlights()
    scrollToFirstHighlight()
  },
)

onMounted(() => {
  if (props.documentId) loadPdf()
  window.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div class="h-100 d-flex flex-column">
    <div class="d-flex align-items-center gap-2 border-bottom px-2 py-2 bg-white flex-wrap">
      <!-- Zoom controls -->
      <button type="button" class="btn btn-sm btn-outline-secondary" :disabled="!documentId" @click="zoomOut" title="Zoom out (Ctrl-)">
        −
      </button>
      <button type="button" class="btn btn-sm btn-outline-secondary" :disabled="!documentId" @click="zoomIn" title="Zoom in (Ctrl+)">
        +
      </button>
      <button type="button" class="btn btn-sm btn-outline-secondary" :disabled="!documentId" @click="resetZoom" title="Reset zoom (Ctrl+0)">
        Reset
      </button>
      <select
        class="form-select form-select-sm"
        style="max-width: 110px"
        :value="scale"
        :disabled="!documentId"
        @change="onScalePresetChange"
      >
        <option v-for="preset in scalePresets" :key="preset" :value="preset">
          {{ Math.round(preset * 100) }}%
        </option>
      </select>
      <span class="small text-muted">{{ Math.round(scale * 100) }}%</span>

      <!-- Page navigation -->
      <div class="vr mx-1"></div>
      <button type="button" class="btn btn-sm btn-outline-secondary" :disabled="!documentId || currentPage <= 1" @click="prevPage" title="Previous page">
        ‹
      </button>
      <input
        v-model.number="pageInput"
        type="number"
        min="1"
        :max="totalPages"
        class="form-control form-control-sm text-center"
        style="max-width: 60px"
        :disabled="!documentId"
        @change="onPageInputChange"
        @keydown.enter="onPageInputChange"
      />
      <span class="small text-muted">/ {{ totalPages }}</span>
      <button type="button" class="btn btn-sm btn-outline-secondary" :disabled="!documentId || currentPage >= totalPages" @click="nextPage" title="Next page">
        ›
      </button>
    </div>

    <div ref="containerRef" class="position-relative flex-grow-1 overflow-auto bg-light rounded-3 p-2">
    <div
      v-if="!documentId"
      class="d-flex h-100 align-items-center justify-content-center text-muted"
    >
      <p class="fs-6 mb-0">Select a document to view</p>
    </div>

    <div v-else class="d-flex flex-column align-items-center gap-3 p-2">
      <canvas
        v-for="page in totalPages"
        :key="page"
        :id="`pdf-page-${page}`"
        class="pdf-canvas"
      />

      <div
        v-for="(hl, idx) in highlightStyles"
        :key="idx"
        class="position-absolute"
        :style="{
          top: hl.top,
          left: hl.left,
          width: hl.width,
          height: hl.height,
          background: 'rgba(255, 200, 0, 0.25)',
          border: '2px solid rgba(220, 140, 0, 0.85)',
          borderRadius: '2px',
          pointerEvents: 'none',
          mixBlendMode: 'multiply',
        }"
      />
    </div>
    </div>
  </div>
</template>
