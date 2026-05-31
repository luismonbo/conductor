import { test, expect } from '@playwright/test';

test.describe('Chat interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for the app to be ready
    await page.waitForSelector('textarea[aria-label="Message"]', { timeout: 10_000 });
  });

  test('happy path: send message, see thinking stream, see done', async ({ page }) => {
    const textarea = page.locator('textarea[aria-label="Message"]');
    const sendButton = page.locator('button', { hasText: 'Send' });

    // Type a simple message
    await textarea.fill('Say "hello" in exactly one word.');
    await expect(sendButton).toBeEnabled();

    // Send
    await sendButton.click();

    // Input should be disabled while streaming
    await expect(textarea).toBeDisabled({ timeout: 5_000 });

    // A thinking block should appear (the .thinking-prose div)
    await expect(page.locator('.thinking-prose').first()).toBeVisible({ timeout: 30_000 });

    // Eventually the input re-enables (done state)
    await expect(textarea).toBeEnabled({ timeout: 60_000 });

    // The user message appears in the list
    await expect(page.locator('text=Say "hello" in exactly one word.')).toBeVisible();
  });

  test('cancel: send a message, cancel before done, input re-enables', async ({ page }) => {
    const textarea = page.locator('textarea[aria-label="Message"]');
    const sendButton = page.locator('button', { hasText: 'Send' });
    const cancelButton = page.locator('button', { hasText: 'Cancel' });

    await textarea.fill('Count from 1 to 100 slowly.');
    await sendButton.click();

    // Wait for Cancel button to appear
    await expect(cancelButton).toBeVisible({ timeout: 10_000 });

    // Cancel
    await cancelButton.click();

    // Input should re-enable after cancel
    await expect(textarea).toBeEnabled({ timeout: 10_000 });

    // Cancel button should be gone
    await expect(cancelButton).not.toBeVisible({ timeout: 5_000 });
  });

  test('Enter key submits message', async ({ page }) => {
    const textarea = page.locator('textarea[aria-label="Message"]');

    await textarea.fill('ping');
    await textarea.press('Enter');

    // Should start streaming (textarea disables)
    await expect(textarea).toBeDisabled({ timeout: 5_000 });
  });

  test('Shift+Enter adds newline without submitting', async ({ page }) => {
    const textarea = page.locator('textarea[aria-label="Message"]');

    await textarea.fill('line one');
    await textarea.press('Shift+Enter');
    // Textarea should still be enabled (not submitted)
    await expect(textarea).toBeEnabled();
    // Value should now contain a newline
    const value = await textarea.inputValue();
    expect(value).toContain('\n');
  });
});
