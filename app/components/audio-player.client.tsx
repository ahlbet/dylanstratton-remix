import * as React from 'react'
import { Button } from './ui/button'
import { Icon } from './ui/icon'

interface AudioPlayerProps {
	src: string
	timestamp: string
}

export function AudioPlayer({ src, timestamp }: AudioPlayerProps) {
	const [isPlaying, setIsPlaying] = React.useState(false)
	const [currentTime, setCurrentTime] = React.useState(0)
	const [duration, setDuration] = React.useState(0)
	const [isLoaded, setIsLoaded] = React.useState(false)
	const audioRef = React.useRef<HTMLAudioElement>(null)

	// Initialize audio element when component mounts
	React.useEffect(() => {
		if (audioRef.current) {
			const audio = audioRef.current

			const handleLoad = () => {
				setDuration(audio.duration)
				setIsLoaded(true)
			}

			const handleTimeUpdate = () => {
				setCurrentTime(audio.currentTime)
			}

			// Add event listeners
			audio.addEventListener('loadeddata', handleLoad)
			audio.addEventListener('durationchange', handleLoad)
			audio.addEventListener('timeupdate', handleTimeUpdate)

			// Load the audio
			void audio.load()

			// Cleanup
			return () => {
				audio.removeEventListener('loadeddata', handleLoad)
				audio.removeEventListener('durationchange', handleLoad)
				audio.removeEventListener('timeupdate', handleTimeUpdate)
			}
		}
	}, [src])

	const handlePlayPause = async () => {
		if (!audioRef.current) return

		try {
			if (isPlaying) {
				await audioRef.current.pause()
			} else {
				await audioRef.current.play()
			}
			setIsPlaying(!isPlaying)
		} catch (error) {
			console.error('Error controlling playback:', error)
		}
	}

	const formatTime = (time: number) => {
		if (!isFinite(time) || time === 0) return '0:00'
		const minutes = Math.floor(time / 60)
		const seconds = Math.floor(time % 60)
		return `${minutes}:${seconds.toString().padStart(2, '0')}`
	}

	const handleProgressBarClick = (event: React.MouseEvent<HTMLDivElement>) => {
		if (!audioRef.current || !isLoaded) return

		const progressBar = event.currentTarget
		const rect = progressBar.getBoundingClientRect()
		const x = event.clientX - rect.left
		const percent = x / rect.width
		const newTime = percent * duration

		if (audioRef.current) {
			audioRef.current.currentTime = newTime
			setCurrentTime(newTime)
		}
	}

	const progressPercent = React.useMemo(() => {
		if (!isLoaded || duration === 0) return 0
		return (currentTime / duration) * 100
	}, [currentTime, duration, isLoaded])

	return (
		<div className="mt-6 rounded-lg bg-gray-100 p-4">
			<div className="mb-2 flex items-center justify-between">
				<h3 className="text-sm font-medium">Generated Audio</h3>
				<span className="text-xs text-gray-500">
					{new Date(timestamp).toLocaleString()}
				</span>
			</div>
			<div className="flex flex-col gap-4">
				<audio
					ref={audioRef}
					src={src}
					preload="auto"
					onEnded={() => setIsPlaying(false)}
				/>
				<div
					className="h-2 w-full cursor-pointer overflow-hidden rounded-full bg-gray-200"
					onClick={handleProgressBarClick}
				>
					<div
						className="h-full bg-indigo-600 transition-all duration-100"
						style={{
							width: `${progressPercent}%`,
						}}
					/>
				</div>
				<div className="flex items-center justify-between">
					<div className="flex items-center gap-4">
						<Button
							variant="outline"
							size="sm"
							onClick={handlePlayPause}
							className="flex-shrink-0 bg-white hover:bg-gray-100"
						>
							<Icon
								name={isPlaying ? 'pause' : 'play'}
								className="size-4 text-gray-700"
							/>
						</Button>
						<span className="text-sm text-gray-600 tabular-nums">
							{formatTime(currentTime)} / {formatTime(duration)}
						</span>
					</div>
					<Button
						variant="ghost"
						size="sm"
						onClick={() => {
							const link = document.createElement('a')
							link.href = src
							link.download = `generated-audio-${timestamp}.wav`
							link.click()
						}}
					>
						<Icon name="download" className="size-4" />
					</Button>
				</div>
			</div>
		</div>
	)
}
