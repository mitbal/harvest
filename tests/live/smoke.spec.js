const { test, expect } = require('@playwright/test');

const publicPages = [
  { path: '/stock_picker', heading: 'Dividend Ranking' },
  { path: '/calendar', heading: /Dividend Calendar/ },
  { path: '/best_timing', heading: 'Best Timing' },
  { path: '/porto_overview', heading: 'Portfolio Analytics' },
  { path: '/simulator', heading: 'Compounding Simulator' },
  { path: '/market_watch', heading: 'Market Heatmap' },
  { path: '/stock_comparison', heading: 'Stock Comparison' },
  { path: '/backtester', heading: 'Growth at a Discount' },
  { path: '/day_trading', heading: 'Short-Term Swing Trading Strategy Lab' },
];

async function expectHealthyStreamlitPage(page, path, heading) {
  const response = await page.goto(path, { waitUntil: 'domcontentloaded' });

  expect(response, `${path} did not return a document response`).not.toBeNull();
  expect(response.status(), `${path} returned HTTP ${response.status()}`).toBeLessThan(400);
  await expect(page.getByRole('heading', { level: 1, name: heading })).toBeVisible();
  await expect(page.locator('[data-testid="stException"]')).toHaveCount(0);
}

test('Streamlit health endpoint is ready', async ({ request }) => {
  const response = await request.get('/_stcore/health');

  expect(response.ok()).toBeTruthy();
  expect((await response.text()).trim()).toBe('ok');
});

test('home page renders the primary experience', async ({ page }) => {
  await expectHealthyStreamlitPage(
    page,
    '/',
    'Data jernih. Keputusan lebih tenang.',
  );
  await expect(page).toHaveTitle(/Panen Dividen/);
  await expect(page.getByRole('link', { name: /Mulai dari Ranking Screener/ })).toBeVisible();
});

test('stock picker opens a research section for a selected stock', async ({ page }) => {
  await expectHealthyStreamlitPage(page, '/stock_picker', 'Dividend Ranking');

  const stockSelect = page.getByRole('combobox', { name: 'Selected stock' });
  await stockSelect.click();
  await stockSelect.pressSequentially('BBCA.JK');
  await page.getByRole('option', { name: 'BBCA.JK', exact: true }).click();

  await expect(stockSelect).toHaveValue('BBCA.JK');
  await expect(
    page.getByRole('heading', { name: 'Dividend history: BBCA.JK', exact: true }),
  ).toBeVisible();

  await page.getByRole('radio', { name: 'Financials', exact: true }).click();

  await expect(page).toHaveURL(/(?:\?|&)section=financials(?:&|$)/);
  await expect(
    page.getByRole('heading', { name: 'Financial information: BBCA.JK', exact: true }),
  ).toBeVisible();
  await expect(page.locator('[data-testid="stException"]')).toHaveCount(0);
});

for (const { path, heading } of publicPages) {
  test(`${path} renders without a Streamlit exception`, async ({ page }) => {
    await expectHealthyStreamlitPage(page, path, heading);
  });
}
