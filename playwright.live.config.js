const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/live',
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: {
    timeout: 60_000,
  },
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI
    ? [['list'], ['html', { outputFolder: 'playwright-report', open: 'never' }]]
    : 'list',
  use: {
    ...devices['Desktop Chrome'],
    baseURL: process.env.BASE_URL || 'https://panendividen.com',
    navigationTimeout: 60_000,
    actionTimeout: 15_000,
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
});
