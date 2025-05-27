import * as React from 'react'
import { Outlet } from 'react-router'
import { requireUserId } from '#app/utils/auth.server.ts'
import { prisma } from '#app/utils/db.server.ts'
import { invariantResponse } from '@epic-web/invariant'

export async function loader({ request }: { request: Request }) {
	const userId = await requireUserId(request)
	const user = await prisma.user.findUnique({ where: { id: userId } })
	invariantResponse(user, 'User not found', { status: 404 })
	return { user }
}

export default function TrainingRoute() {
	return (
		<div className="m-auto mt-8 mb-24 max-w-3xl">
			<div className="container">
				<main className="bg-muted mx-auto px-6 py-8 md:container md:rounded-3xl">
					<h1 className="text-h1">Training</h1>
					<p className="text-body-md mt-4">
						Welcome to your training dashboard!
					</p>
					<Outlet />
				</main>
			</div>
		</div>
	)
}
