import { test, expect } from '@playwright/test';

// Runs against the fake backend (HARNESS_LLM_BACKEND=fake) — zero credentials.
// The fake streams a deterministic reply, so assertions are stable.

test('message roundtrip with fake backend', async ({ page }) => {
  await page.goto('/');
  const input = page.locator('textarea[aria-label="Message"]');
  await input.fill('hello harness');
  await input.press('Enter');
  // fake backend streams a deterministic reply word-by-word
  await expect(page.getByText('Fake backend is active', { exact: false })).toBeVisible();
});

test('sidebar shows the conversation after completion', async ({ page }) => {
  await page.goto('/');
  const input = page.locator('textarea[aria-label="Message"]');
  // Unique per run: the sqlite run store persists across test runs, and a
  // repeated title would trip Playwright's strict mode.
  const marker = `sidebar check ${Date.now()}`;
  await input.fill(marker);
  await input.press('Enter');
  await expect(page.getByText('Fake backend is active', { exact: false })).toBeVisible();
  await expect(
    page.getByRole('navigation', { name: 'conversations' }).getByText(marker),
  ).toBeVisible();
});
