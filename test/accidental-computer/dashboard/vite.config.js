import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server runs on port 3000 (required by the demo setup).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    strictPort: true,
  },
})
