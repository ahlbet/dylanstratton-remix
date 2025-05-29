import type { LoaderFunctionArgs } from '@remix-run/server-runtime'
import fs from 'node:fs/promises'
import path from 'node:path'

export async function loader({ request, params }: LoaderFunctionArgs) {
	const filename = params.filename
	if (!filename) {
		throw new Response('Filename is required', { status: 400 })
	}

	const filePath = path.join(process.cwd(), 'public', 'audio-uploads', filename)

	try {
		const stat = await fs.stat(filePath)
		const fileSize = stat.size

		const rangeHeader = request.headers.get('Range')

		if (!rangeHeader) {
			// No range requested, serve the whole file
			const file = await fs.readFile(filePath)
	
			return new Response(file, {
				status: 200,
				headers: {
					'Content-Type': 'audio/wav',
					'Content-Length': fileSize.toString(),
					'Accept-Ranges': 'bytes',
					'Cache-Control': 'no-cache'
				}
			})
		}

		// Parse range header
		const matches = rangeHeader.match(/bytes=(\d*)-(\d*)/)
		if (!matches) {
			return new Response(null, {
				status: 416,
				headers: {
					'Content-Range': `bytes */${fileSize}`,
					'Accept-Ranges': 'bytes'
				}
			})
		}

		const [, startStr, endStr] = matches
		
		// Handle special cases for range requests
		let start = startStr ? parseInt(startStr, 10) : 0
		let end = endStr ? parseInt(endStr, 10) : fileSize - 1

		// Ensure start and end are within valid bounds
		start = Math.max(0, start)
		end = Math.min(fileSize - 1, end)

		// Handle case where start is greater than end or file size
		if (start >= fileSize || start > end) {
			return new Response(null, {
				status: 416,
				headers: {
					'Content-Range': `bytes */${fileSize}`,
					'Accept-Ranges': 'bytes'
				}
			})
		}

		const contentLength = end - start + 1

		// Read the requested range
		const handle = await fs.open(filePath, 'r')
		try {
			const buffer = Buffer.alloc(contentLength)
			const { bytesRead } = await handle.read(buffer, 0, contentLength, start)
			
			// Even if we read fewer bytes, we'll still return what we have
			const actualEnd = start + bytesRead - 1
			
			return new Response(buffer.slice(0, bytesRead), {
				status: 206,
				headers: {
					'Content-Type': 'audio/wav',
					'Content-Length': bytesRead.toString(),
					'Content-Range': `bytes ${start}-${actualEnd}/${fileSize}`,
					'Accept-Ranges': 'bytes',
					'Cache-Control': 'no-cache'
				}
			})
		} finally {
			await handle.close()
		}
	} catch (error) {
		if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
			throw new Response('File not found', { status: 404 })
		}
		throw error
	}
} 