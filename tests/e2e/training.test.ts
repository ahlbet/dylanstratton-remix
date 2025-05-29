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
		
		// Click train button and log its state
		console.log('Before click - Button text:', await trainButton.evaluate(el => el.textContent))
		
		// Start watching for navigation and network requests
		const navigationPromise = page.waitForURL(/\/training/)
		let lastResponseBody = ''
		
		// Listen for all responses to catch errors
		page.on('response', async response => {
			if (response.url().includes('/training')) {
				try {
					const contentType = response.headers()['content-type'] || ''
					if (contentType.includes('application/json')) {
						const body = await response.json()
						lastResponseBody = JSON.stringify(body, null, 2)
						console.log('Training response:', lastResponseBody)
						
						// Check for Remix error format
						if (Array.isArray(body) && body.includes('error')) {
							console.log('Detected Remix error response')
							const errorIndex = body.indexOf('error')
							if (body[errorIndex + 1] && Array.isArray(body[errorIndex + 1])) {
								const errorData = body[errorIndex + 1] as string[]
								const [errorType, ...errorDetails] = errorData
								console.log('Error type:', errorType)
								console.log('Error details:', errorDetails)
							}
						}
					} else {
						const text = await response.text()
						lastResponseBody = text
						console.log('Training response (text):', text)
					}
					console.log('Response status:', response.status())
					
					// Log response headers for debugging
					console.log('Response headers:', response.headers())
				} catch (e) {
					console.log('Error parsing response:', e)
				}
			}
		})

		// Also listen for console messages that might indicate server errors
		page.on('console', msg => {
			if (msg.type() === 'error') {
				console.log('Browser console error:', msg.text())
			}
		})

		// Click and wait for navigation
		await trainButton.click()
		console.log('After click - Button clicked')

		try {
			// Wait for navigation with error handling
			await Promise.race([
				navigationPromise,
				page.waitForSelector('[data-error-type="SanitizedError"]', { timeout: 5000 })
					.then(() => { throw new Error('Server error detected') })
			])
			console.log('Navigation completed')
			
			// Wait for network idle to ensure all requests are complete
			await page.waitForLoadState('networkidle')
			console.log('Network is idle')

			// Check for error messages
			const errorLocator = page.getByText(/error|failed/i)
			const hasError = await errorLocator.isVisible()
			
			if (hasError) {
				console.log('Found error state')
				
				// Get error details from Remix error boundary if present
				const errorBoundaryText = await page.locator('[data-error-boundary]').allTextContents()
				if (errorBoundaryText.length > 0) {
					console.log('Error boundary content:', errorBoundaryText)
				}
				
				// Get all error related text
				const errorMessages = await page.getByRole('alert').allTextContents()
				console.log('Error messages:', errorMessages)
				
				// Get form state
				const formData = await page.evaluate(() => {
					const form = document.querySelector('form')
					if (!form) return null
					const formData = new FormData(form)
					return Object.fromEntries(formData.entries())
				})
				console.log('Form data:', formData)

				// Get any server-side error details that might be in the DOM
				const serverError = await page.evaluate(() => {
					const errorElement = document.querySelector('[data-server-error]')
					return errorElement ? errorElement.textContent : null
				})
				if (serverError) {
					console.log('Server error details:', serverError)
				}
				
				// Take error screenshot
				await page.screenshot({ path: 'test-artifacts/error-state.png' })
				
				// Save page HTML
				const html = await page.content()
				await require('fs').promises.writeFile('test-artifacts/error-state.html', html)
				
				throw new Error(`Training failed: ${errorMessages.join(', ')}. Server error: ${serverError || 'Unknown'}. Last response: ${lastResponseBody}`)
			}

			// Wait for success indicators with a longer timeout for CI
			await Promise.race([
				page.getByRole('button', { name: /generate audio/i }).waitFor({ timeout: 45000 }),
				page.getByText(/training completed|model trained/i).waitFor({ timeout: 45000 })
			])
			console.log('Found success indicator')

			// Verify generate button is present
			await expect(page.getByRole('button', { name: /generate audio/i })).toBeVisible({ timeout: 5000 })
			console.log('Generate audio button is visible')
		} catch (error) {
			// If we timeout or encounter an error, capture the final state
			console.log('Final page content:', await page.textContent('body'))
			console.log('Last response received:', lastResponseBody)
			
			// Take a failure screenshot
			await page.screenshot({ path: 'test-artifacts/failure.png' })
			
			// Save page HTML for debugging
			const html = await page.content()
			await require('fs').promises.writeFile('test-artifacts/failure.html', html)
			
			throw error
		}
	})

	test.afterEach(async ({ page }, testInfo) => {
		// Capture trace only on failure
		if (testInfo.status !== testInfo.expectedStatus) {
			await page.screenshot({ path: `test-artifacts/${testInfo.title}-failure.png` })
			console.log('Test failed, saving trace and other artifacts')
		}
	})
}) 