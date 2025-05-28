import * as React from 'react'
import {
	Outlet,
	useRouteError,
	Form,
	useNavigation,
	useActionData,
} from 'react-router'
import { requireUserWithRole } from '#app/utils/permissions.server.ts'
import { prisma } from '#app/utils/db.server.ts'
import { invariantResponse } from '@epic-web/invariant'
import { GeneralErrorBoundary } from '../../components/error-boundary.tsx'
import { Icon } from '../../components/ui/icon.tsx'
import { StatusButton } from '../../components/ui/status-button.tsx'
import fs from 'node:fs/promises'
import path from 'node:path'

const ClientAudioPlayer = React.lazy(() =>
	import('../../components/audio-player.client.tsx').then((mod) => ({
		default: mod.AudioPlayer,
	})),
)

export async function loader({ request }: { request: Request }) {
	const userId = await requireUserWithRole(request, 'admin')
	const user = await prisma.user.findUnique({ where: { id: userId } })
	invariantResponse(user, 'User not found', { status: 404 })
	return { user }
}

export async function action({ request }: { request: Request }) {
	const userId = await requireUserWithRole(request, 'admin')
	const formData = await request.formData()
	const intent = formData.get('intent')

	switch (intent) {
		case 'train': {
			const files = formData.getAll('audioFiles')
			const uploadDir = path.join(process.cwd(), 'public', 'audio-uploads')

			// Ensure upload directory exists
			await fs.mkdir(uploadDir, { recursive: true })

			// Save each file
			for (const file of files) {
				if (file instanceof File) {
					const buffer = Buffer.from(await file.arrayBuffer())
					const filename = `${Date.now()}-${file.name}`
					await fs.writeFile(path.join(uploadDir, filename), buffer)
				}
			}

			return { success: true }
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

			return {
				success: true,
				audioUrl: `/audio-uploads/${latestFile}`,
				timestamp: new Date().toISOString(),
			}
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
	const navigation = useNavigation()
	const actionData = useActionData<typeof action>()
	const [selectedFiles, setSelectedFiles] = React.useState<FileList | null>(
		null,
	)
	const [isModelTrained, setIsModelTrained] = React.useState(false)
	const [generatedAudio, setGeneratedAudio] = React.useState<{
		url: string
		timestamp: string
	} | null>(null)
	const isPending = navigation.state === 'submitting'
	const pendingIntent = isPending
		? navigation.formData?.get('intent')?.toString()
		: null
	const formRef = React.useRef<HTMLFormElement>(null)

	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const files = event.target.files
		if (files && files.length > 0) {
			setSelectedFiles(files)
		}
	}

	const handleTrainSuccess = () => {
		setIsModelTrained(true)
	}

	// Handle form submission response
	React.useEffect(() => {
		if (
			navigation.state === 'idle' &&
			actionData?.success &&
			'audioUrl' in actionData &&
			'timestamp' in actionData &&
			typeof actionData.audioUrl === 'string' &&
			typeof actionData.timestamp === 'string'
		) {
			setGeneratedAudio({
				url: actionData.audioUrl,
				timestamp: actionData.timestamp,
			})
		}
	}, [navigation.state, actionData])

	return (
		<div className="container flex min-h-[400px] flex-1 px-0 pb-12 md:px-8">
			<main className="bg-muted mx-auto px-6 py-8 md:container md:rounded-3xl">
				<h1 className="text-h1">Training</h1>
				<p className="text-body-md mt-4 mb-8">
					Welcome to your training dashboard!
				</p>

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
								onSubmit={() => setSelectedFiles(null)}
							>
								<input
									type="file"
									name="audioFiles"
									accept=".wav"
									multiple
									onChange={handleFileChange}
									className="mt-4"
								/>
								{selectedFiles && (
									<p className="mt-2 text-sm text-gray-500">
										{selectedFiles.length} file(s) selected
									</p>
								)}
								<div className="mt-4 flex gap-4">
									<StatusButton
										type="submit"
										name="intent"
										value="train"
										status={pendingIntent === 'train' ? 'pending' : 'idle'}
										className="w-full"
										disabled={!selectedFiles || isPending}
										onClick={handleTrainSuccess}
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

					{/* Audio Player */}
					{generatedAudio && (
						<React.Suspense fallback={<div>Loading audio player...</div>}>
							<ClientAudioPlayer
								src={generatedAudio.url}
								timestamp={generatedAudio.timestamp}
							/>
						</React.Suspense>
					)}
				</div>

				<Outlet />
			</main>
		</div>
	)
}

export { ErrorBoundary }
