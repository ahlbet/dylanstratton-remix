import fs from 'node:fs/promises'
import path from 'node:path'

import { invariantResponse } from '@epic-web/invariant'
import * as React from 'react'
import { useEffect } from 'react'
import {
	Outlet,
	useRouteError,
	Form,
	useNavigation,
	useActionData,
	useLoaderData,
} from 'react-router'

import { AudioPlayer as ClientAudioPlayer } from '#app/components/audio-player.client.tsx'
import { GeneralErrorBoundary } from '#app/components/error-boundary.tsx'
import { Icon } from '#app/components/ui/icon.tsx'
import { StatusButton } from '#app/components/ui/status-button.tsx'
import { prisma } from '#app/utils/db.server.ts'
import { requireUserWithRole } from '#app/utils/permissions.server.ts'

// Server-side only import
const getAudioDuration = async (filePath: string): Promise<number> => {
	if (process.env.NODE_ENV === 'production') {
		type AudioDurationModule = {
			getAudioDurationInSeconds: (path: string) => Promise<number>
		}
		const mod = (await import(
			'get-audio-duration'
		)) as unknown as AudioDurationModule
		return mod.getAudioDurationInSeconds(filePath)
	}
	return 0 // Mock for development
}

export async function loader({ request }: { request: Request }) {
	const userId = await requireUserWithRole(request, 'admin')
	const user = await prisma.user.findUnique({
		where: { id: userId },
		include: {
			trainingAudio: {
				orderBy: { createdAt: 'desc' },
				select: {
					id: true,
					filename: true,
					objectKey: true,
					duration: true,
					fileSize: true,
					createdAt: true,
				},
			},
		},
	})
	invariantResponse(user, 'User not found', { status: 404 })

	// Verify files exist and filter out missing ones
	const verifiedTrainingAudio = await Promise.all(
		user.trainingAudio.map(async (audio) => {
			const filePath = path.join(process.cwd(), 'public', audio.objectKey)
			try {
				await fs.access(filePath)
				return audio
			} catch {
				// Delete the database record if file is missing
				await prisma.trainingAudio.delete({ where: { id: audio.id } })
				return null
			}
		}),
	)

	// Filter out null entries (missing files)
	const filteredTrainingAudio = verifiedTrainingAudio.filter(
		(audio): audio is NonNullable<typeof audio> => audio !== null,
	)

	return { user: { ...user, trainingAudio: filteredTrainingAudio } }
}

export async function action({ request }: { request: Request }) {
	const userId = await requireUserWithRole(request, 'admin')
	const formData = await request.formData()
	const intent = formData.get('intent')

	switch (intent) {
		case 'train': {
			const files = formData.getAll('audioFiles')
			const uploadDir = path.join(process.cwd(), 'public', 'audio-uploads')

			// Validate files
			if (files.length === 0) {
				return { success: false, error: 'No files were provided' }
			}

			const validFiles = files.filter((f): f is File => {
				const file = f as unknown
				const isValid =
					file instanceof File &&
					(file.type === 'audio/wav' ||
						file.type === 'audio/wave' ||
						file.name.toLowerCase().endsWith('.wav'))
				return isValid
			})

			if (validFiles.length === 0) {
				return { success: false, error: 'No valid files were provided' }
			}

			if (validFiles.length !== files.length) {
				return { success: false, error: 'Some files were invalid' }
			}

			// Ensure upload directory exists
			await fs.mkdir(uploadDir, { recursive: true })

			// Process and save each file
			const savedFiles = []
			for (const file of validFiles) {
				try {
					const buffer = Buffer.from(await file.arrayBuffer())

					// Add index and random string to ensure unique filenames
					const timestamp = Date.now()
					const randomStr = Math.random().toString(36).substring(2, 8)
					const sanitizedName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_')
					const filename = `${timestamp}-${randomStr}-${sanitizedName}`
					const filePath = path.join(uploadDir, filename)

					// Save file to disk
					await fs.writeFile(filePath, buffer)

					// Get audio duration - use mock in development/test
					const duration =
						process.env.NODE_ENV === 'production'
							? await getAudioDuration(filePath)
							: 0

					// Save metadata to database
					const trainingAudio = await prisma.trainingAudio.create({
						data: {
							filename: file.name,
							objectKey: `/audio-uploads/${filename}`,
							duration,
							fileSize: buffer.length,
							mimeType: 'audio/wav',
							userId,
						},
					})

					savedFiles.push(trainingAudio)
				} catch (error) {
					console.error('Error processing file:', file.name, error)
					// Continue with next file instead of failing completely
				}
			}

			// Check if any files were saved successfully
			if (savedFiles.length > 0) {
				return { success: true, savedFiles }
			}

			// If no files were saved successfully, return error
			return { success: false, error: 'No files were processed successfully' }
		}
		case 'generate': {
			// For testing, we'll use one of the uploaded files
			const uploadDir = path.join(process.cwd(), 'public', 'audio-uploads')
			const files = await fs.readdir(uploadDir)

			if (files.length === 0) {
				throw new Error('No audio files available')
			}

			// Use the most recently uploaded file (based on our naming convention)
			const latestFile = files.sort().reverse()[0]
			const timestamp = new Date().toISOString()

			// Return the audio URL and timestamp
			return {
				success: true,
				audioUrl: `/audio-uploads/${latestFile}`,
				timestamp,
			}
		}
		case 'delete': {
			const audioId = formData.get('audioId')
			if (typeof audioId !== 'string') {
				throw new Error('Audio ID is required')
			}

			// Get the audio file info
			const audio = await prisma.trainingAudio.findUnique({
				where: { id: audioId },
				select: { objectKey: true, userId: true },
			})

			if (!audio) {
				throw new Error('Audio file not found')
			}

			// Ensure user owns this file
			if (audio.userId !== userId) {
				throw new Error('Not authorized')
			}

			// Delete the file from disk
			const filePath = path.join(process.cwd(), 'public', audio.objectKey)
			try {
				await fs.unlink(filePath)
			} catch {
				// Continue even if file deletion fails - file might have been manually deleted
			}

			// Delete the database record
			await prisma.trainingAudio.delete({
				where: { id: audioId },
			})

			return { success: true, deletedId: audioId }
		}
		default: {
			throw new Error('Invalid intent')
		}
	}
}

function ErrorBoundary() {
	const error = useRouteError() as any

	if (error?.status === 403) {
		return (
			<div className="container mx-auto flex h-full w-full flex-col justify-center pt-20 pb-32 text-center">
				<h3 className="text-h3">Unauthorized</h3>
				<p className="mt-2">You do not have access to this page.</p>
			</div>
		)
	}

	return <GeneralErrorBoundary />
}

export default function TrainingRoute() {
	const { user } = useLoaderData<typeof loader>()
	const navigation = useNavigation()
	const actionData = useActionData<typeof action>()
	const [selectedFiles, setSelectedFiles] = React.useState<FileList | null>(
		null,
	)
	const [fileError, setFileError] = React.useState<string | null>(null)
	const [generatedAudio, setGeneratedAudio] = React.useState<{
		url: string
		timestamp: string
	} | null>(null)

	const isPending = navigation.state === 'submitting'
	const pendingIntent = isPending
		? navigation.formData?.get('intent')?.toString()
		: null
	const pendingDeleteId = isPending
		? navigation.formData?.get('audioId')?.toString()
		: null
	const formRef = React.useRef<HTMLFormElement>(null)
	const fileInputRef = React.useRef<HTMLInputElement>(null)

	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const files = event.target.files

		if (!files?.length) {
			setSelectedFiles(null)
			setFileError(null)
			return
		}

		const fileArray = Array.from(files)
		const hasInvalidFile = fileArray.some((file) => {
			const isWav = file.type === 'audio/wav'
			return !isWav
		})

		if (hasInvalidFile) {
			setFileError('Only WAV files are allowed')
			setSelectedFiles(null)
			return
		}

		setSelectedFiles(files)
		setFileError(null)
	}

	useEffect(() => {
		if (formRef.current && fileInputRef.current) {
			if (!selectedFiles) {
				formRef.current.reset()
				fileInputRef.current.value = ''
			}
		}
	}, [selectedFiles])

	useEffect(() => {
		if (actionData?.success && actionData.audioUrl && actionData.timestamp) {
			setGeneratedAudio({
				url: actionData.audioUrl,
				timestamp: actionData.timestamp,
			})
		}
	}, [actionData])

	useEffect(() => {
		if (navigation.state === 'idle' && actionData?.success && formRef.current) {
			formRef.current.reset()
			setSelectedFiles(null)
		}
	}, [navigation.state, actionData])

	// Determine if the model is trained based on action data or existing files
	const isModelTrained = Boolean(
		(navigation.state === 'idle' &&
			actionData?.success &&
			actionData.savedFiles) ||
			(user.trainingAudio && user.trainingAudio.length > 0),
	)

	return (
		<div className="container flex min-h-[400px] flex-1 px-0 pb-12 md:px-8">
			<main className="bg-muted mx-auto px-6 py-8 md:container md:rounded-3xl">
				{/* Hidden state for testing */}
				<div
					data-testid="component-state"
					className="hidden"
					dangerouslySetInnerHTML={{
						__html: JSON.stringify({
							selectedFiles: selectedFiles
								? {
										length: selectedFiles.length,
										files: Array.from(selectedFiles).map((f) => ({
											name: f.name,
											type: f.type,
											size: f.size,
										})),
									}
								: null,
							fileError,
							isPending,
							navigationState: navigation.state,
							buttonDisabled: Boolean(!selectedFiles || fileError || isPending),
						}),
					}}
				/>

				<h1 className="text-h1">Training</h1>
				<p className="text-body-md mt-4 mb-8">
					Welcome to your training dashboard!
				</p>

				{/* Hidden navigation state for testing */}
				<div data-testid="navigation-state" className="hidden">
					{navigation.state}
				</div>

				<div className="flex flex-col gap-8">
					{/* File Upload Section */}
					<div className="rounded-lg border-2 border-dashed border-gray-300 p-6">
						<div className="flex flex-col items-center">
							<Icon name="upload" className="size-12 text-gray-400" />
							<p className="mt-4 text-sm text-gray-600">
								Upload WAV files for training
							</p>
							<Form
								ref={formRef}
								method="POST"
								encType="multipart/form-data"
								data-success={actionData?.success}
							>
								<input
									ref={fileInputRef}
									type="file"
									name="audioFiles"
									accept=".wav"
									multiple
									onChange={handleFileChange}
									className="mt-4"
									data-testid="file-input"
								/>
								{fileError && (
									<p
										className="text-destructive mt-2 text-sm"
										role="alert"
										data-testid="file-error"
									>
										{fileError}
									</p>
								)}
								{selectedFiles && !fileError && (
									<p
										className="mt-2 text-sm text-gray-500"
										data-testid="file-count"
									>
										{selectedFiles.length} file
										{selectedFiles.length !== 1 ? 's' : ''} selected
									</p>
								)}
								<div className="mt-4 flex gap-4">
									<StatusButton
										type="submit"
										name="intent"
										value="train"
										status={pendingIntent === 'train' ? 'pending' : 'idle'}
										className="w-full"
										disabled={Boolean(!selectedFiles || fileError || isPending)}
										data-testid="train-button"
									>
										<div className="flex items-center gap-2">
											<Icon name="moon" className="size-4 text-gray-700" />
											{pendingIntent === 'train'
												? 'Training...'
												: 'Train Model'}
										</div>
									</StatusButton>

									{isModelTrained && (
										<StatusButton
											type="submit"
											name="intent"
											value="generate"
											status={pendingIntent === 'generate' ? 'pending' : 'idle'}
											className="w-full"
											disabled={isPending}
										>
											<div className="flex items-center gap-2">
												<Icon name="sun" className="size-4 text-gray-700" />
												{pendingIntent === 'generate'
													? 'Generating...'
													: 'Generate Audio'}
											</div>
										</StatusButton>
									)}
								</div>
							</Form>
						</div>
					</div>

					{/* Generated Audio Player */}
					{generatedAudio && (
						<div
							className="rounded-lg border border-gray-200 p-4"
							data-testid="generated-audio"
						>
							<h2 className="mb-4 text-lg font-medium">Generated Audio</h2>
							<React.Suspense fallback={<div>Loading audio player...</div>}>
								<ClientAudioPlayer
									src={generatedAudio.url}
									timestamp={generatedAudio.timestamp}
									key={`${generatedAudio.url}-${generatedAudio.timestamp}`}
								/>
							</React.Suspense>
						</div>
					)}

					{/* Historical Training Audio Files */}
					{user.trainingAudio.length > 0 && (
						<div
							className="border-gray-20 rounded-lg border p-4"
							data-testid="past-samples"
						>
							<h2 className="mb-4 text-lg font-medium">Past Samples</h2>
							<div className="flex flex-col gap-4">
								{user.trainingAudio.map((audio) => (
									<div
										key={audio.id}
										className="flex items-center justify-between gap-4"
									>
										<React.Suspense
											fallback={<div>Loading audio player...</div>}
										>
											<ClientAudioPlayer
												src={audio.objectKey}
												timestamp={audio.createdAt.toISOString()}
											/>
										</React.Suspense>
										<Form method="POST">
											<input type="hidden" name="audioId" value={audio.id} />
											<StatusButton
												type="submit"
												name="intent"
												value="delete"
												variant="destructive"
												className="shrink-0"
												data-testid={`delete-audio-${audio.id}`}
												status={
													pendingDeleteId === audio.id ? 'pending' : 'idle'
												}
											>
												<Icon name="trash" className="size-4" />
												Delete
											</StatusButton>
										</Form>
									</div>
								))}
							</div>
						</div>
					)}
				</div>

				<Outlet />
			</main>
		</div>
	)
}

export { ErrorBoundary }
