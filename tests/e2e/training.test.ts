import { faker } from '@faker-js/faker'
import { prisma } from '#app/utils/db.server.ts'
import { createUser, expect, test } from '#tests/playwright-utils.ts'

test.describe('Training Flow', () => {
	let testUserId: string

	test.beforeEach(async ({ page, login }) => {
		// Create admin user
		const userData = createUser()
		const user = await prisma.user.create({
			data: {
				...userData,
				roles: { connect: [{ name: 'admin' }] },
			},
		})
		testUserId = user.id

		// Clean up any existing test data for this user
		await prisma.trainingAudio.deleteMany({
			where: { userId: testUserId }
		})

		// Login as admin user
		await login({ id: user.id })
		await page.goto('/training')
	})

	test.afterAll(async () => {
		// Clean up test data after all tests
		if (testUserId) {
			await prisma.trainingAudio.deleteMany({
				where: { userId: testUserId }
			})
		}
	})

	test('should display training interface for admin users', async ({ page }) => {
		// Check main UI elements
		await expect(page.getByRole('heading', { name: 'Training' })).toBeVisible()
		await expect(page.getByText(/upload wav files/i)).toBeVisible()
		await expect(page.getByRole('button', { name: /train model/i })).toBeVisible()
	})

	test('should handle file upload', async ({ page }) => {
		// Setup file input handling
		const testAudioPath = 'tests/fixtures/test-audio.wav'
		await page.setInputFiles('input[type="file"]', testAudioPath)

		// Verify file selection
		await expect(page.getByText(/file\(s\) selected/i)).toBeVisible()
	})

	test('should handle training process', async ({ page }) => {
		// Setup file input handling
		const testAudioPath = 'tests/fixtures/test-audio.wav'
		console.log('Attempting to upload file:', testAudioPath)
		
		// Find and verify the file input exists
		const fileInput = page.locator('input[type="file"]')
		await expect(fileInput).toBeVisible()
		console.log('File input found')
		
		// Upload the file
		await fileInput.setInputFiles(testAudioPath)
		console.log('File uploaded')
		
		// Wait a moment for any UI updates
		await page.waitForTimeout(1000)
		
		// Log the page content to see what's available
		const mainContent = await page.locator('main').textContent()
		console.log('Main content:', mainContent)
		
		// Check if file was uploaded
		const fileInput2 = page.locator('input[type="file"]')
		const files = await fileInput2.inputValue()
		console.log('File input value:', files)

		// Proceed with training if file is uploaded
		const trainButton = page.getByRole('button', { name: /train model/i })
		await expect(trainButton).toBeEnabled()
		console.log('Train button is enabled')
		
		// Start listening to network requests
		let trainingStarted = false
		page.on('request', request => {
			if (request.url().includes('/training')) {
				console.log('Training request started:', request.url())
				trainingStarted = true
			}
		})

		// Click train button and log its state
		console.log('Before click - Button text:', await trainButton.evaluate(el => el.textContent))
		
		// Start watching for navigation before clicking
		const navigationPromise = page.waitForURL(/\/training/)
		
		// Click and wait for navigation
		await trainButton.click()
		console.log('After click - Button clicked')
		await navigationPromise
		console.log('Navigation completed')
		
		// Wait for network idle to ensure all requests are complete
		await page.waitForLoadState('networkidle')
		console.log('Network is idle')

		// Wait for success indicators - either the generate button or success message
		await Promise.race([
			page.getByRole('button', { name: /generate audio/i }).waitFor({ timeout: 20000 }),
			page.getByText(/training completed|model trained/i).waitFor({ timeout: 20000 })
		])
		console.log('Found success indicator')

		// Verify generate button is present
		await expect(page.getByRole('button', { name: /generate audio/i })).toBeVisible({ timeout: 5000 })
		console.log('Generate audio button is visible')
	})
}) 