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
}) 