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

	test('should handle audio file upload and training', async ({ page }) => {
		// Setup file input handling
		const testAudioPath = 'tests/fixtures/test-audio.wav'
		await page.setInputFiles('input[type="file"]', testAudioPath)

		// Verify file selection
		await expect(page.getByText(/file\(s\) selected/i)).toBeVisible()

		// Click train button and wait for training completion
		const trainButton = page.getByRole('button', { name: /train model/i })
		
		// Submit the form and wait for the response
		const responsePromise = page.waitForResponse(async response => {
			if (!response.url().includes('/training')) return false
			const text = await response.text()
			return text.includes('"success":true') && text.includes('"savedFiles":[{')
		})
		await trainButton.click()
		await responsePromise
		
		// Wait for the training state to end
		await expect(page.getByRole('button', { name: /training\.\.\./i })).not.toBeVisible({ timeout: 10000 })
		
		// Verify training completion by checking for generate button
		await expect(page.getByRole('button', { name: /generate audio/i })).toBeVisible({ timeout: 10000 })
	})

	test('should generate and play audio', async ({ page }) => {
		// Upload and train first
		const testAudioPath = 'tests/fixtures/test-audio.wav'
		await page.setInputFiles('input[type="file"]', testAudioPath)
		
		// Start training
		const trainButton = page.getByRole('button', { name: /train model/i })
		
		// Submit the form and wait for the response
		const responsePromise = page.waitForResponse(async response => {
			if (!response.url().includes('/training')) return false
			const text = await response.text()
			return text.includes('"success":true') && text.includes('"savedFiles":[{')
		})
		await trainButton.click()
		await responsePromise
		
		// Wait for the training state to end
		await expect(page.getByRole('button', { name: /training\.\.\./i })).not.toBeVisible({ timeout: 10000 })
		
		// Wait for the generate button to appear
		const generateButton = page.getByRole('button', { name: /generate audio/i })
		await expect(generateButton).toBeVisible({ timeout: 10000 })
		await expect(generateButton).toBeEnabled({ timeout: 10000 })

		// Generate audio
		await generateButton.click()
		
		// Wait for the generating state
		await expect(page.getByRole('button', { name: /generating\.\.\./i })).toBeVisible()
		
		// Wait for the Generated Audio section to appear and be ready
		const generatedSection = page.locator('div.rounded-lg.border', { hasText: 'Generated Audio' }).first()
		await expect(generatedSection).toBeVisible()
		
		// Wait for the audio element to be present in this section
		await expect(generatedSection.locator('audio')).toHaveAttribute('src', /audio-uploads/)
	})

	test('should display historical samples', async ({ page }) => {
		// Create the training audio using the test user ID
		await prisma.trainingAudio.create({
			data: {
				filename: 'test.wav',
				objectKey: '/audio-uploads/test.wav',
				duration: 30,
				fileSize: 1024,
				mimeType: 'audio/wav',
				userId: testUserId,
				createdAt: new Date(),
				updatedAt: new Date(),
			},
		})

		// Navigate to the page and wait for it to be ready
		await page.goto('/training')
		await page.waitForLoadState('networkidle')
		
		// Wait for the page to be ready
		await expect(page.getByRole('heading', { name: 'Training', level: 1 })).toBeVisible()

		// Look for the Past Samples section using the exact class structure from the component
		const pastSamplesSection = page.locator('div.rounded-lg.border.p-4').filter({ hasText: 'Past Samples' })
		await expect(pastSamplesSection).toBeVisible()
		await expect(pastSamplesSection.getByRole('heading', { level: 2, name: 'Past Samples' })).toBeVisible()
		
		// Wait for the audio player to be loaded
		const audioElement = pastSamplesSection.locator('audio')
		await expect(audioElement).toBeAttached()
		await expect(audioElement).toHaveAttribute('src', '/audio-uploads/test.wav')
		await expect(audioElement).toHaveAttribute('preload', 'auto')
	})

	test('unauthorized users cannot access training', async ({ page, login }) => {
		// Create and login as non-admin user
		const userData = createUser()
		const user = await prisma.user.create({
			data: {
				...userData,
				roles: { connect: [{ name: 'user' }] },
			},
		})

		// Login as non-admin user
		await login({ id: user.id })

		// Try to access training page
		await page.goto('/training')
		await expect(page.getByText(/unauthorized/i)).toBeVisible()
		await expect(page.getByText(/you do not have access/i)).toBeVisible()
	})
}) 