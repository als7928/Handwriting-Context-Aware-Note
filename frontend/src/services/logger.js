/**
 * Lightweight logger that respects Vite's dev/prod mode.
 *
 * In development (npm run dev): DEBUG, INFO, WARN, ERROR all print.
 * In production build:          only WARN and ERROR print.
 *
 * Each message is prefixed with timestamp | level.
 */

const isDev = import.meta.env.DEV

function _fmt(level) {
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19)
  return `${ts} | ${level} |`
}

export const logger = {
  debug: (...args) => { if (isDev) console.debug(_fmt('DEBUG'), ...args) },
  info:  (...args) => { if (isDev) console.log(_fmt('INFO'),  ...args) },
  warn:  (...args) => console.warn(_fmt('WARN'),  ...args),
  error: (...args) => console.error(_fmt('ERROR'), ...args),
}
