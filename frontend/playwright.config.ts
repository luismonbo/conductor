import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  // NOTE: webServer requires both backend and frontend to be running externally
  // Run: uv run uvicorn harness.api.main:app --reload --app-dir src
  // Run: pnpm -C frontend dev
  // before executing: pnpm -C frontend exec playwright test
});
