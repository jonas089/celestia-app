import { useState } from 'react'

export function CopyButton({ value, label = 'Copy' }) {
  const [copied, setCopied] = useState(false)

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(value)
    } catch {
      // Fallback for non-secure contexts / older browsers.
      const ta = document.createElement('textarea')
      ta.value = value
      document.body.appendChild(ta)
      ta.select()
      try {
        document.execCommand('copy')
      } catch {
        /* ignore */
      }
      document.body.removeChild(ta)
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1200)
  }

  return (
    <button className="btn btn--ghost" onClick={onCopy} title="Copy to clipboard">
      {copied ? '✓ Copied' : label}
    </button>
  )
}
