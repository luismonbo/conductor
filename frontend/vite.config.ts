/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig(({ mode }) => {
  const rootDir = path.resolve(__dirname, '..')
  const env = loadEnv(mode, rootDir, '')

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: { '@': path.resolve(__dirname, './src') },
    },
    envDir: rootDir,
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq) => {
              // This proxy injects the shared key for every caller, so it also owns
              // X-Visitor-IP (the API trusts it on keyed requests): drop any copy a
              // caller sent, or they could pick their own rate-limit bucket.
              proxyReq.removeHeader('x-visitor-ip')
              if (env.HARNESS_API_KEY) {
                proxyReq.setHeader('Authorization', `Bearer ${env.HARNESS_API_KEY}`)
              }
            })
          },
        },
      },
    },
    test: {
      environment: 'happy-dom',
      globals: true,
      setupFiles: ['./src/test/setup.ts'],
      exclude: ['**/node_modules/**', '**/e2e/**'],
    },
  }
})
