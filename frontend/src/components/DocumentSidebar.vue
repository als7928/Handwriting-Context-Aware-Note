<script setup>
/**
 * DocumentSidebar – lists uploaded documents with upload / delete controls.
 */
import { ref, onMounted } from 'vue'
import { listDocuments, uploadDocument, deleteDocument } from '../services/api.js'

const emit = defineEmits(['select'])

const documents = ref([])
const uploading = ref(false)
const selectedId = ref(null)

async function fetchDocuments() {
  documents.value = await listDocuments()
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  if (!file) return

  uploading.value = true
  try {
    const doc = await uploadDocument(file)
    documents.value.unshift(doc)
    selectDocument(doc.id)
  } catch (err) {
    alert('Upload failed. Please try again.')
  } finally {
    uploading.value = false
    event.target.value = ''
  }
}

async function handleDelete(id) {
  if (!confirm('Delete this document and all its data?')) return
  await deleteDocument(id)
  documents.value = documents.value.filter((d) => d.id !== id)
  if (selectedId.value === id) {
    selectedId.value = null
    emit('select', null)
  }
}

function selectDocument(id) {
  selectedId.value = id
  emit('select', id)
}

onMounted(fetchDocuments)
</script>

<template>
  <div class="d-flex h-100 flex-column">
    <div class="card-header d-flex align-items-center justify-content-between">
      <h2 class="h6 mb-0">Documents</h2>
      <label
        class="btn btn-sm btn-primary"
        :class="{ disabled: uploading }"
      >
        {{ uploading ? 'Uploading…' : '+ Upload' }}
        <input type="file" accept=".pdf" class="d-none" @change="handleUpload" />
      </label>
    </div>

    <div class="flex-grow-1 overflow-auto list-group list-group-flush">
      <div v-if="documents.length === 0" class="p-4 text-center text-muted small">
        No documents yet
      </div>

      <button
        v-for="doc in documents"
        :key="doc.id"
        @click="selectDocument(doc.id)"
        type="button"
        class="list-group-item list-group-item-action d-flex align-items-center gap-2"
        :class="{ active: selectedId === doc.id }"
      >
        <svg class="text-danger" width="28" height="28" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 1.5L18.5 9H13V3.5zM6 20V4h5v7h7v9H6z" />
        </svg>

        <div class="flex-grow-1 text-start text-truncate">
          <div class="fw-semibold text-truncate">{{ doc.filename }}</div>
          <div class="small" :class="selectedId === doc.id ? 'text-white-50' : 'text-muted'">
            {{ doc.page_count }} pages
          </div>
        </div>

        <button
          @click.stop="handleDelete(doc.id)"
          class="btn btn-sm"
          :class="selectedId === doc.id ? 'btn-outline-light' : 'btn-outline-danger'"
          title="Delete"
        >
          <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </button>
    </div>
  </div>
</template>
