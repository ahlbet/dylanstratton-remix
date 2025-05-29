import fs from 'node:fs/promises'
import path from 'node:path'
import { prisma } from '#app/utils/db.server.ts'
import { createUser, expect, test } from '#tests/playwright-utils.ts'

function createTestWavFile(): Buffer {
	// Audio parameters for web-compatible WAV
	const sampleRate = 44100
	const numChannels = 1
	const bitsPerSample = 16
	const duration = 0.1 // seconds (shorter duration for testing)
	const frequency = 440 // Hz (A4 note)
	
	// Calculate sizes
	const bytesPerSample = bitsPerSample / 8
	const blockAlign = numChannels * bytesPerSample
	const numSamples = Math.floor(sampleRate * duration)
	const dataSize = numSamples * blockAlign
	const headerSize = 44 // Standard WAV header size
	const fileSize = headerSize + dataSize

	// Create buffer
	const buffer = Buffer.alloc(fileSize)

	// Write WAV header
	// 1. RIFF chunk descriptor
	buffer.write('RIFF', 0)                     // ChunkID
	buffer.writeUInt32LE(fileSize - 8, 4)       // ChunkSize
	buffer.write('WAVE', 8)                     // Format

	// 2. fmt sub-chunk
	buffer.write('fmt ', 12)                    // Subchunk1ID
	buffer.writeUInt32LE(16, 16)                // Subchunk1Size (16 for PCM)
	buffer.writeUInt16LE(1, 20)                 // AudioFormat (1 for PCM)
	buffer.writeUInt16LE(numChannels, 22)       // NumChannels
	buffer.writeUInt32LE(sampleRate, 24)        // SampleRate
	buffer.writeUInt32LE(                       // ByteRate
		sampleRate * blockAlign,
		28
	)
	buffer.writeUInt16LE(blockAlign, 32)        // BlockAlign
	buffer.writeUInt16LE(bitsPerSample, 34)     // BitsPerSample

	// 3. data sub-chunk
	buffer.write('data', 36)                    // Subchunk2ID
	buffer.writeUInt32LE(dataSize, 40)          // Subchunk2Size

	// Generate audio data - a simple sine wave
	const amplitude = 0.5 // Avoid clipping
	for (let i = 0; i < numSamples; i++) {
		const t = i / sampleRate
		const value = amplitude * Math.sin(2 * Math.PI * frequency * t)
		// Convert float (-0.5 to 0.5) to 16-bit PCM (-32768 to 32767)
		const sample = Math.floor(value * 32767)
		buffer.writeInt16LE(sample, headerSize + i * bytesPerSample)
	}

	return buffer
}

test.describe('Training Flow', () => {
	let testUserId: string
	let testAudioPath: string
	let fixturesDir: string
	let filesToClean: string[] = []
	let uniqueId: string

	test.beforeAll(async () => {
		// Create unique ID for this test run to avoid conflicts in concurrent runs
		uniqueId = Date.now().toString() + '-' + Math.random().toString(36).slice(2)

		// Create test directories with unique subdirectories
		fixturesDir = path.join(process.cwd(), 'tests', 'fixtures', 'training-tests', uniqueId)
		const testArtifactsDir = path.join(process.cwd(), 'test-artifacts', uniqueId)
		const uploadsDir = path.join(process.cwd(), 'public', 'audio-uploads')
		
		await Promise.all([
			fs.mkdir(fixturesDir, { recursive: true }),
			fs.mkdir(testArtifactsDir, { recursive: true }),
			fs.mkdir(uploadsDir, { recursive: true })
		])

		// Set up test audio file path with unique name
		testAudioPath = path.join(fixturesDir, `test-audio-${uniqueId}.wav`)
		filesToClean.push(testAudioPath)
		
		try {
			// Create a proper WAV file with a sine wave tone
			const wavBuffer = createTestWavFile()
			await fs.writeFile(testAudioPath, wavBuffer)
			
			// Verify the file was created
			const stats = await fs.stat(testAudioPath)
			console.log('Created test WAV file:', {
				path: testAudioPath,
				size: stats.size,
				exists: true
			})
		} catch (error: unknown) {
			if (error instanceof Error) {
				throw error
			}
			throw new Error('Failed to create test audio file')
		}
	})

	test.beforeEach(async ({ page, login }) => {
		// Create admin user with unique email to avoid conflicts
		const userData = createUser()
		userData.email = `test-${uniqueId}-${userData.email}`
		const user = await prisma.user.create({
			data: {
				...userData,
				roles: { connect: [{ name: 'admin' }] },
			},
		})
		testUserId = user.id
		console.log('Created test user:', { id: user.id, email: userData.email, uniqueId })

		// Clean up any existing test data for this user
		await prisma.trainingAudio.deleteMany({
			where: { userId: testUserId }
		})

		// Login as admin user
		await login({ id: user.id })
		await page.goto('/training')
	})

	test.afterEach(async ({ page }, testInfo) => {
		// Clean up test data
		if (testUserId) {
			await prisma.trainingAudio.deleteMany({
				where: { userId: testUserId }
			})
		}

		// Capture artifacts on failure with unique names
		if (testInfo.status !== testInfo.expectedStatus) {
			const artifactsDir = path.join(process.cwd(), 'test-artifacts', uniqueId)
			await fs.mkdir(artifactsDir, { recursive: true })

			try {
				const timestamp = Date.now()
				await page.screenshot({ 
					path: path.join(artifactsDir, `${testInfo.title}-${timestamp}-failure.png`)
				})
				
				const html = await page.content()
				await fs.writeFile(
					path.join(artifactsDir, `${testInfo.title}-${timestamp}-failure.html`),
					html
				)
			} catch (error: unknown) {
				if (error instanceof Error) {
					console.error('Error saving test artifacts:', error)
				}
			}
		}
	})

	test.afterAll(async () => {
		// Clean up test data
		if (testUserId) {
			await prisma.trainingAudio.deleteMany({
				where: { userId: testUserId }
			})
		}

		// Clean up only our test-specific files
		await Promise.all(
			filesToClean.map(async (filePath) => {
				try {
					await fs.access(filePath)
					await fs.unlink(filePath)
					console.log('Successfully deleted test file:', filePath)
				} catch (error: unknown) {
					// Ignore errors if file doesn't exist
					if (error instanceof Error && 
						(error as NodeJS.ErrnoException).code !== 'ENOENT'
					) {
						console.error(`Error cleaning up file ${filePath}:`, error)
					}
				}
			})
		)

		// Clean up only our test-specific directory
		try {
			await fs.rm(fixturesDir, { recursive: true, force: true })
			await fs.rm(path.join(process.cwd(), 'test-artifacts', uniqueId), { recursive: true, force: true })
			console.log('Successfully cleaned up test directories:', {
				fixturesDir,
				artifactsDir: path.join(process.cwd(), 'test-artifacts', uniqueId)
			})
		} catch (error: unknown) {
			if (error instanceof Error) {
				console.error('Error cleaning up test directories:', error)
			}
		}
	})

	test('should handle multiple file uploads', async ({ page }) => {
		console.log('Starting multiple file uploads test')

		// Create second test file
		const secondTestAudioPath = path.join(fixturesDir, `test-audio-${uniqueId}-2.wav`)
		await fs.writeFile(secondTestAudioPath, createTestWavFile())
		filesToClean.push(secondTestAudioPath)

		// Verify both files exist and are readable
		const [file1Stats, file2Stats] = await Promise.all([
			fs.stat(testAudioPath),
			fs.stat(secondTestAudioPath)
		])

		console.log('Test files prepared:', {
			file1: {
				path: testAudioPath,
				size: file1Stats.size,
				exists: true
			},
			file2: {
				path: secondTestAudioPath,
				size: file2Stats.size,
				exists: true
			}
		})

		// Wait for file input to be ready
		const fileInput = page.getByTestId('file-input')
		await expect(fileInput).toBeVisible()
		await expect(fileInput).toBeEnabled()

		// Start listening for console logs
		page.on('console', msg => {
			console.log('Browser console:', msg.type(), msg.text())
		})

		// Log initial form state
		console.log('Initial form state:')
		const formState = await page.evaluate(() => {
			const form = document.querySelector('form')
			const input = document.querySelector('[data-testid="file-input"]')
			return {
				formExists: !!form,
				inputExists: !!input,
				inputType: input?.getAttribute('type'),
				inputName: input?.getAttribute('name'),
				inputAccept: input?.getAttribute('accept')
			}
		})
		console.log(formState)

		// Set files and dispatch change event
		console.log('Setting input files...')
		try {
			await fileInput.setInputFiles([testAudioPath, secondTestAudioPath])
			console.log('Files set successfully')

			// Log file input state after setting files
			const inputState = await page.evaluate(() => {
				const input = document.querySelector('[data-testid="file-input"]') as HTMLInputElement
				return {
					files: input?.files ? Array.from(input.files).map(f => ({ name: f.name, type: f.type, size: f.size })) : null,
					value: input?.value
				}
			})
			console.log('File input state after setting files:', inputState)

			// Dispatch change event
			await page.evaluate(() => {
				const input = document.querySelector('[data-testid="file-input"]')
				if (input) {
					console.log('Dispatching change event')
					const event = new Event('change', { bubbles: true, cancelable: true })
					Object.defineProperty(event, 'target', { value: input })
					input.dispatchEvent(event)
					console.log('Change event dispatched')
				} else {
					console.error('Input element not found for change event')
				}
			})

			// Wait a moment and check React state
			await page.waitForTimeout(500)
			const componentState = await page.evaluate(() => {
				const state = document.querySelector('[data-testid="component-state"]')
				return state?.textContent || 'Component state element not found'
			})
			console.log('Component state after change:', componentState)

			// Check for file count element
			console.log('Checking for file count element...')
			const fileCountExists = await page.evaluate(() => {
				return !!document.querySelector('[data-testid="file-count"]')
			})
			console.log('File count element exists:', fileCountExists)

			// Wait for file count with increased timeout and debug info
			try {
				await expect(page.getByTestId('file-count')).toBeVisible({ timeout: 5000 });
				console.log('File count element became visible')
				await expect(page.getByTestId('file-count')).toHaveText('2 files selected')
				console.log('File count text verified')
			} catch (error) {
				console.error('Failed to find file count element:', error)
				// Log the entire page content for debugging
				console.log('Current page content:', await page.content())
				throw error
			}

			// Verify train button state
			const trainButton = page.getByTestId('train-button')
			await expect(trainButton).toBeVisible()
			await expect(trainButton).toBeEnabled()

			// Submit form and wait for response
			console.log('Submitting form...')
			try {
				// Log form state before submission
				const formState = await page.evaluate(() => {
					const form = document.querySelector('form')
					const fileInput = form?.querySelector('input[type="file"]') as HTMLInputElement | null
					return {
						formExists: !!form,
						formAction: form?.getAttribute('action'),
						formMethod: form?.getAttribute('method'),
						formEnctype: form?.getAttribute('enctype'),
						fileInput: {
							exists: !!fileInput,
							files: fileInput?.files ? Array.from(fileInput.files).map(f => ({
								name: f.name,
								type: f.type,
								size: f.size
							})) : []
						}
					}
				})
				console.log('Form state before submission:', formState)

				// Start listening for all responses before clicking
				page.on('response', async response => {
					if (response.url().includes('/training')) {
						console.log('Response received:', {
							url: response.url(),
							method: response.request().method(),
							status: response.status(),
							headers: response.headers()
						})
					}
				})

				// Also listen for request failures
				page.on('requestfailed', request => {
					console.error('Request failed:', {
						url: request.url(),
						method: request.method(),
						error: request.failure()?.errorText
					})
				})

				// Log network activity
				page.on('request', request => {
					if (request.url().includes('/training')) {
						console.log('Request started:', {
							url: request.url(),
							method: request.method(),
							headers: request.headers()
						})
					}
				})

				// Submit form with explicit form submission
				console.log('Clicking train button...')
				await trainButton.click()
				console.log('Train button clicked')

				// Wait for response with detailed logging
				console.log('Waiting for response...')
				const response = await page.waitForResponse(
					response => {
						const matches = response.url().includes('/training') && 
							response.request().method() === 'POST'
						if (matches) {
							console.log('Matched response:', {
								url: response.url(),
								method: response.request().method(),
								status: response.status()
							})
						}
						return matches
					},
					{ timeout: 30000 }
				)
				console.log('Response received')

				// Log response details for debugging
				const responseStatus = response.status()
				const responseText = await response.text()
				console.log('Form submission response:', {
					status: responseStatus,
					text: responseText.slice(0, 1000), // Limit text length for logging
					headers: response.headers()
				})

				if (responseStatus !== 200) {
					throw new Error(`Unexpected response status: ${responseStatus}`)
				}

				// Wait for UI updates after successful submission
				console.log('Waiting for UI updates...')
				await Promise.all([
					page.getByTestId('past-samples').waitFor({ timeout: 10000 }),
					page.getByTestId('train-button').waitFor({ timeout: 10000 })
				])
				console.log('UI updates complete')

				// Check database state
				console.log('Checking database state...')
				const trainingFiles = await prisma.trainingAudio.findMany({
					where: { userId: testUserId }
				})
				console.log('Training files in database:', trainingFiles)

				// Wait for generate button with shorter timeout
				console.log('Waiting for generate button...')
				await expect(
					page.getByRole('button', { name: /generate audio/i })
				).toBeVisible({ timeout: 10000 })

				// Wait for past samples section
				console.log('Waiting for past-samples section...')
				await expect(page.getByTestId('past-samples')).toBeVisible({ timeout: 10000 })

				// Wait for audio elements with explicit count check
				console.log('Waiting for audio elements...')
				const audioElements = page.getByTestId('past-samples').locator('audio')
				await expect(audioElements).toHaveCount(2, { timeout: 10000 })

				// Verify audio elements
				const count = await audioElements.count()
				console.log(`Found ${count} audio elements`)
				
				for (let i = 0; i < count; i++) {
					const src = await audioElements.nth(i).getAttribute('src')
					console.log(`Audio element ${i + 1} src:`, src)
				}
			} catch (error) {
				console.error('Error during form submission:', error)
				
				// Log current page state
				console.log('Current page URL:', page.url())
				console.log('Current page content:', await page.content())
				
				// Take a screenshot
				await page.screenshot({ 
					path: `test-artifacts/form-submission-failure-${uniqueId}.png`,
					fullPage: true 
				})
				
				throw error
			}
		} catch (error) {
			console.error('Error during file upload test:', error)
			// Take a screenshot on failure
			await page.screenshot({ 
				path: `test-artifacts/file-upload-failure-${uniqueId}.png`,
				fullPage: true 
			})
			throw error
		}
	})

	test('should allow deleting training files', async ({ page }) => {
		// Upload a file
		await page.getByTestId('file-input').setInputFiles(testAudioPath)
		
		// Wait for file selection UI updates
		await expect(page.getByTestId('file-count')).toBeVisible()
		await expect(page.getByTestId('file-count')).toHaveText('1 file selected')
		
		// Train with file
		const trainButton = page.getByRole('button', { name: /train model/i })
		await expect(trainButton).toBeEnabled()
		
		// Click and wait for response
		await Promise.all([
			page.waitForResponse(response => 
				response.url().includes('/training') && 
				response.request().method() === 'POST'
			),
			trainButton.click()
		])
		
		// Wait for training completion
		await expect(page.getByRole('button', { name: /generate audio/i })).toBeVisible()
		
		// Find and click delete button for the file
		const deleteButton = page.getByRole('button', { name: /delete/i }).first()
		await expect(deleteButton).toBeVisible()
		
		// Click and wait for deletion to complete
		await Promise.all([
			page.waitForResponse(response => 
				response.url().includes('/training') && 
				response.request().method() === 'POST'
			),
			deleteButton.click()
		])
		
		// Verify file is removed from UI
		await expect(page.getByTestId('past-samples').locator('audio')).toHaveCount(0)
		
		// Verify in database
		const remainingFiles = await prisma.trainingAudio.findMany({
			where: { userId: testUserId }
		})
		expect(remainingFiles).toHaveLength(0)
	})

	test('should reject non-WAV files', async ({ page }) => {
		// Create a temporary MP3 file with unique name
		const mp3Path = path.join(fixturesDir, `test-${uniqueId}.mp3`)
		await fs.writeFile(mp3Path, Buffer.from([0xFF, 0xFB])) // Simple MP3 header
		filesToClean.push(mp3Path)

		try {
			// Upload MP3 file
			await page.getByTestId('file-input').setInputFiles(mp3Path)
			
			// Wait for error message to appear
			await expect(page.getByTestId('file-error')).toBeVisible()
			await expect(page.getByTestId('file-error')).toHaveText('Only WAV files are allowed')
			
			// Verify train button is disabled
			const trainButton = page.getByRole('button', { name: /train model/i })
			await expect(trainButton).toBeDisabled()
		} finally {
			// Clean up MP3 file
			try {
				await fs.unlink(mp3Path)
			} catch (error) {
				console.error('Error cleaning up MP3 file:', error)
			}
		}
	})

	test('should generate audio after training', async ({ page }) => {
		// Upload a file
		await page.getByTestId('file-input').setInputFiles(testAudioPath)
		
		// Wait for file selection UI updates
		await expect(page.getByTestId('file-count')).toBeVisible()
		await expect(page.getByTestId('file-count')).toHaveText('1 file selected')
		
		// Train with file
		const trainButton = page.getByRole('button', { name: /train model/i })
		await expect(trainButton).toBeEnabled()
		
		// Click and wait for response
		await Promise.all([
			page.waitForResponse(response => 
				response.url().includes('/training') && 
				response.request().method() === 'POST'
			),
			trainButton.click()
		])
		
		// Wait for training completion
		const generateButton = page.getByRole('button', { name: /generate audio/i })
		await expect(generateButton).toBeVisible()
		
		// Click generate and wait for response
		await Promise.all([
			page.waitForResponse(response => 
				response.url().includes('/training') && 
				response.request().method() === 'POST'
			),
			generateButton.click()
		])
		
		// Verify generated audio appears
		await expect(page.getByTestId('generated-audio').locator('audio')).toBeVisible()
		
		// Verify audio source is set
		const audioSrc = await page.getByTestId('generated-audio').locator('audio').getAttribute('src')
		expect(audioSrc).toMatch(/\/audio-uploads\/.*\.wav/)
	})

	test('should restrict access to non-admin users', async ({ page, login }) => {
		// Create non-admin user
		const userData = createUser()
		const user = await prisma.user.create({
			data: userData
		})
		
		// Login as non-admin
		await login({ id: user.id })
		
		// Try to access training page
		await page.goto('/training')
		
		// Verify access denied
		await expect(page.getByRole('heading', { name: 'Unauthorized' })).toBeVisible()
		await expect(page.getByText('You do not have access to this page.')).toBeVisible()
	})

	test('should display training interface for admin users', async ({ page }) => {
		// Check main UI elements
		await expect(page.getByRole('heading', { name: 'Training' })).toBeVisible()
		await expect(page.getByText(/upload wav files/i)).toBeVisible()
		await expect(page.getByRole('button', { name: /train model/i })).toBeVisible()
	})

	test('should handle file upload', async ({ page }) => {
		// Upload the test file
		await page.getByTestId('file-input').setInputFiles(testAudioPath)

		// Verify file selection
		await expect(page.getByTestId('file-count')).toBeVisible()
		await expect(page.getByTestId('file-count')).toHaveText('1 file selected')
	})

	test('should handle training process', async ({ page }) => {
		console.log('Starting training process test')
		
		// Find and verify the file input exists
		const fileInput = page.getByTestId('file-input')
		await expect(fileInput).toBeVisible()
		console.log('File input found')
		
		// Upload the file and wait for file selection to be processed
		await fileInput.setInputFiles(testAudioPath)
		console.log('File uploaded')
		
		// Wait for file count to appear, indicating successful file selection
		await expect(page.getByTestId('file-count')).toBeVisible()
		await expect(page.getByTestId('file-count')).toHaveText('1 file selected')
		console.log('File selection confirmed')
		
		// Wait for train button to be enabled
		const trainButton = page.getByTestId('train-button')
		await expect(trainButton).toBeEnabled()
		console.log('Train button is enabled')
		
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
					} else {
						const text = await response.text()
						lastResponseBody = text
						console.log('Training response (text):', text)
					}
					console.log('Response status:', response.status())
				} catch (e) {
					console.log('Error parsing response:', e)
				}
			}
		})

		// Also listen for console messages that might indicate client-side state
		page.on('console', msg => {
			console.log('Browser console:', msg.type(), msg.text())
		})

		// Click train button and wait for response
		await Promise.all([
			page.waitForResponse(response => response.url().includes('/training')),
			trainButton.click()
		])
		console.log('Train button clicked and response received')

		// Wait for either success or error state
		try {
			await Promise.race([
				page.getByRole('button', { name: /generate audio/i }).waitFor(),
				page.getByTestId('file-error').waitFor()
					.then(() => { throw new Error('File error detected') })
			])
			console.log('Training completed successfully')

			// Verify generate button is present
			await expect(page.getByRole('button', { name: /generate audio/i })).toBeVisible()
			console.log('Generate audio button is visible')
		} catch (error) {
			console.error('Training process failed:', error)
			
			// Get current page state
			const html = await page.content()
			await fs.writeFile('test-artifacts/training-failure.html', html)
			
			// Take screenshot
			await page.screenshot({ path: 'test-artifacts/training-failure.png' })
			
			// Get any error messages
			const errorMessages = await page.getByRole('alert').allTextContents()
			console.error('Error messages:', errorMessages)
			
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